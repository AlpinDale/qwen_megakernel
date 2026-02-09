/**
 * Fused single-kernel decode for Qwen3-0.6B on RTX 5090.
 *
 * Everything — embedding lookup, 28 transformer layers (RMSNorm, QKV, RoPE,
 * attention, O-proj, MLP), and final norm — runs inside one cooperative kernel
 * launch.  The LM head (vocab projection + argmax) is a separate
 * non-cooperative kernel launched immediately after.
 *
 * Optimized for: NVIDIA RTX 5090 (sm_120, 170 SMs, 96 MB L2)
 * Model:         Qwen/Qwen3-0.6B (bf16 weights)
 */

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

// =============================================================================
// Model constants (Qwen3-0.6B)
// =============================================================================

constexpr int WARP_SIZE = 32;
constexpr int HIDDEN_SIZE = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS = 16;
constexpr int NUM_KV_HEADS = 8;
constexpr int HEAD_DIM = 128;
constexpr int Q_SIZE = NUM_Q_HEADS * HEAD_DIM;   // 2048
constexpr int KV_SIZE = NUM_KV_HEADS * HEAD_DIM; // 1024

#ifndef LDG_NUM_BLOCKS
#define LDG_NUM_BLOCKS 128
#endif
#ifndef LDG_BLOCK_SIZE
#define LDG_BLOCK_SIZE 512
#endif
#ifndef LDG_LM_NUM_BLOCKS
#define LDG_LM_NUM_BLOCKS 1184
#endif
#ifndef LDG_LM_BLOCK_SIZE
#define LDG_LM_BLOCK_SIZE 256
#endif
#ifndef LDG_LM_ROWS_PER_WARP
#define LDG_LM_ROWS_PER_WARP 2
#endif

constexpr int LDG_NUM_WARPS = LDG_BLOCK_SIZE / WARP_SIZE;
constexpr float LDG_RMS_EPS = 1e-6f;

// LM head
constexpr int LDG_VOCAB_SIZE = 151936;

struct LDGLayerWeights {
  const __nv_bfloat16 *input_layernorm_weight;
  const __nv_bfloat16 *q_proj_weight;
  const __nv_bfloat16 *k_proj_weight;
  const __nv_bfloat16 *v_proj_weight;
  const __nv_bfloat16 *q_norm_weight;
  const __nv_bfloat16 *k_norm_weight;
  const __nv_bfloat16 *o_proj_weight;
  const __nv_bfloat16 *post_attn_layernorm_weight;
  const __nv_bfloat16 *gate_proj_weight;
  const __nv_bfloat16 *up_proj_weight;
  const __nv_bfloat16 *down_proj_weight;
};

// =============================================================================
// Atomic barrier for persistent kernel (replaces cooperative grid.sync())
// =============================================================================

struct AtomicGridSync {
  unsigned int *counter;
  unsigned int *generation;
  unsigned int nblocks;
  unsigned int local_gen;

  __device__ void sync() {
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned int my_gen = local_gen;
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      unsigned int arrived = atomicAdd(counter, 1);
      if (arrived == nblocks - 1) {
        *counter = 0;
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
        atomicAdd(generation, 1);
      } else {
        volatile unsigned int *vgen = (volatile unsigned int *)generation;
        while (*vgen <= my_gen) {
        }
      }
      local_gen = my_gen + 1;
    }
    __syncthreads();
  }
};

// =============================================================================
// Helpers
// =============================================================================

__device__ __forceinline__ float ldg_warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

constexpr float LOG2E = 1.44269504088896340736f;

__device__ __forceinline__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float ptx_rcp(float x) {
  float y;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float fast_exp(float x) {
  return ptx_exp2(x * LOG2E);
}

__device__ __forceinline__ float ldg_silu(float x) {
  return x * ptx_rcp(1.0f + fast_exp(-x));
}

// 128-bit L1-bypassing weight loads
__device__ __forceinline__ uint4 ldg_load_weight_u4(const uint4 *ptr) {
  uint4 out;
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
               : "l"(ptr));
  return out;
}

// =============================================================================
// Optimized matvec with __ldg and aggressive unrolling
// =============================================================================

__device__ void
ldg_matvec_qkv(auto &grid, const __nv_bfloat16 *__restrict__ input,
               const __nv_bfloat16 *__restrict__ norm_weight,
               const __nv_bfloat16 *__restrict__ q_weight,
               const __nv_bfloat16 *__restrict__ k_weight,
               const __nv_bfloat16 *__restrict__ v_weight,
               float *__restrict__ g_normalized, float *__restrict__ g_residual,
               float *__restrict__ q_out, float *__restrict__ k_out,
               float *__restrict__ v_out) {
  int block_id = blockIdx.x;
  int num_blocks = gridDim.x;
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  __shared__ __align__(16) float s_norm[HIDDEN_SIZE];

  // ALL blocks compute RMSNorm redundantly
  {
    __shared__ float smem_reduce[LDG_NUM_WARPS];

    float local_sum_sq = 0.0f;

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float v = __bfloat162float(__ldg(input + i));
      s_norm[i] = v;
      local_sum_sq += v * v;
    }

    // Block 0 saves residual for later use
    if (block_id == 0) {
      for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
        g_residual[i] = s_norm[i];
      }
    }

    local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
      smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
      float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
      }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float w = __bfloat162float(__ldg(norm_weight + i));
      s_norm[i] = s_norm[i] * rstd * w;
    }
    __syncthreads();
  }

  // QKV projection with vec4 and __ldg
  constexpr int TOTAL_ROWS = Q_SIZE + KV_SIZE + KV_SIZE;
  int rows_per_block = (TOTAL_ROWS + num_blocks - 1) / num_blocks;
  int row_start = block_id * rows_per_block;
  int row_end = min(row_start + rows_per_block, TOTAL_ROWS);

  for (int m_base = row_start; m_base < row_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < row_end) {
      const __nv_bfloat16 *weight_row;
      float *output_ptr;

      if (m < Q_SIZE) {
        weight_row = q_weight + m * HIDDEN_SIZE;
        output_ptr = q_out + m;
      } else if (m < Q_SIZE + KV_SIZE) {
        weight_row = k_weight + (m - Q_SIZE) * HIDDEN_SIZE;
        output_ptr = k_out + (m - Q_SIZE);
      } else {
        weight_row = v_weight + (m - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
        output_ptr = v_out + (m - Q_SIZE - KV_SIZE);
      }

      float sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
        uint4 w_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(weight_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(s_norm + k);
        float4 a2 = *reinterpret_cast<const float4 *>(s_norm + k + 4);

        sum += __bfloat162float(w_ptr[0]) * a1.x +
               __bfloat162float(w_ptr[1]) * a1.y +
               __bfloat162float(w_ptr[2]) * a1.z +
               __bfloat162float(w_ptr[3]) * a1.w +
               __bfloat162float(w_ptr[4]) * a2.x +
               __bfloat162float(w_ptr[5]) * a2.y +
               __bfloat162float(w_ptr[6]) * a2.z +
               __bfloat162float(w_ptr[7]) * a2.w;
      }

      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        *output_ptr = sum;
      }
    }
  }

  grid.sync();
}

// =============================================================================
// Attention with __ldg for KV cache + block divergence for prefetching
// =============================================================================

// Prefetch weights into L2 cache using __ldg reads
__device__ void
ldg_prefetch_weights_l2(const __nv_bfloat16 *__restrict__ weights,
                        int num_elements) {
  // Bulk L2 prefetch (Blackwell) or fallback to cached loads
  // Each thread prefetches strided elements to warm L2 cache
  float dummy = 0.0f;
  for (int i = threadIdx.x; i < num_elements; i += LDG_BLOCK_SIZE) {
    // Read but don't use - compiler won't optimize out due to volatile-like
    // __ldg
    dummy += __bfloat162float(__ldg(weights + i));
  }
  // Prevent optimization (result stored to shared but never used)
  __shared__ float s_dummy;
  if (threadIdx.x == 0)
    s_dummy = dummy;
}

__device__ void ldg_attention(
    auto &grid, float *__restrict__ q, float *__restrict__ k,
    const float *__restrict__ v, __nv_bfloat16 *__restrict__ k_cache,
    __nv_bfloat16 *__restrict__ v_cache, float *__restrict__ attn_out,
    int cache_len, int max_seq_len, float attn_scale,
    // QK norm parameters (fused to eliminate a grid.sync)
    const __nv_bfloat16 *__restrict__ q_norm_weight,
    const __nv_bfloat16 *__restrict__ k_norm_weight,
    const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table, int position,
    // Weights to prefetch during attention (for blocks not doing attention)
    const __nv_bfloat16 *__restrict__ o_weight,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const __nv_bfloat16 *__restrict__ up_weight,
    const __nv_bfloat16 *__restrict__ down_weight,
    // Lightweight flag syncs (independent of grid barrier). null = use
    // grid.sync().
    unsigned int *__restrict__ kv_flag =
        nullptr, // block 0 signals KV cache ready
    unsigned int *__restrict__ attn_flag =
        nullptr, // attention blocks signal completion
    int layer_idx = 0) {
  int block_id = blockIdx.x;
  int num_blocks = gridDim.x;
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  const int ATTN_BLOCKS = NUM_Q_HEADS; // 16 blocks for 16 Q heads
  const __nv_bfloat16 *cos_pos = cos_table + position * HEAD_DIM;
  const __nv_bfloat16 *sin_pos = sin_table + position * HEAD_DIM;

  // -- Fused QK norm: block 0 handles all K heads, attention blocks handle Q --
  // Block 0: K norm + RoPE + KV cache write (8 heads × 128 dim — trivial)
  if (block_id == 0) {
    for (int h = warp_id; h < NUM_KV_HEADS; h += LDG_NUM_WARPS) {
      float *k_head = k + h * HEAD_DIM;
      const float *v_head = v + h * HEAD_DIM;
      __nv_bfloat16 *kc =
          k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
      __nv_bfloat16 *vc =
          v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

      float ss = 0.0f;
      for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
        ss += k_head[i] * k_head[i];
      ss = ldg_warp_reduce_sum(ss);
      float sc = rsqrtf(ss / float(HEAD_DIM) + LDG_RMS_EPS);
      sc = __shfl_sync(0xffffffff, sc, 0);

      float kl[HEAD_DIM / WARP_SIZE];
#pragma unroll
      for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
        kl[j] = k_head[i] * sc * __bfloat162float(__ldg(k_norm_weight + i));
#pragma unroll
      for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
        float cv = __bfloat162float(__ldg(cos_pos + i));
        float sv = __bfloat162float(__ldg(sin_pos + i));
        int po = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
        int pi = i + po, pj = pi / WARP_SIZE;
        float pv = __shfl_sync(0xffffffff, kl[pj], pi % WARP_SIZE);
        float kf =
            (i < HEAD_DIM / 2) ? kl[j] * cv - pv * sv : pv * sv + kl[j] * cv;
        kc[i] = __float2bfloat16(kf);
        vc[i] = __float2bfloat16(v_head[i]);
      }
    }
  }
  // Attention blocks: Q norm + RoPE for own head (warp 0 only — 128 elements)
  if (block_id < ATTN_BLOCKS && block_id < NUM_Q_HEADS && warp_id == 0) {
    float *qh = q + block_id * HEAD_DIM;
    float ss = 0.0f;
    for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE)
      ss += qh[i] * qh[i];
    ss = ldg_warp_reduce_sum(ss);
    float sc = rsqrtf(ss / float(HEAD_DIM) + LDG_RMS_EPS);
    sc = __shfl_sync(0xffffffff, sc, 0);
    float ql[HEAD_DIM / WARP_SIZE];
#pragma unroll
    for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++)
      ql[j] = qh[i] * sc * __bfloat162float(__ldg(q_norm_weight + i));
#pragma unroll
    for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
      float cv = __bfloat162float(__ldg(cos_pos + i));
      float sv = __bfloat162float(__ldg(sin_pos + i));
      int po = (i < HEAD_DIM / 2) ? HEAD_DIM / 2 : -HEAD_DIM / 2;
      int pi = i + po, pj = pi / WARP_SIZE;
      float pv = __shfl_sync(0xffffffff, ql[pj], pi % WARP_SIZE);
      qh[i] = (i < HEAD_DIM / 2) ? ql[j] * cv - pv * sv : pv * sv + ql[j] * cv;
    }
  }

  // Non-attention blocks: prefetch while QK norm runs above (overlapped work)
  if (block_id >= ATTN_BLOCKS) {
    int prefetch_block_id = block_id - ATTN_BLOCKS;
    int num_prefetch_blocks = num_blocks - ATTN_BLOCKS;
    int o_blocks = num_prefetch_blocks * 2 / 11;
    int gate_blocks = num_prefetch_blocks * 3 / 11;
    int up_blocks = num_prefetch_blocks * 3 / 11;
    if (o_blocks < 1)
      o_blocks = 1;

    if (prefetch_block_id < o_blocks) {
      int total = Q_SIZE * HIDDEN_SIZE;
      int epb = (total + o_blocks - 1) / o_blocks;
      int start = prefetch_block_id * epb;
      int count = min(epb, total - start);
      if (count > 0)
        ldg_prefetch_weights_l2(o_weight + start, count);
    } else if (prefetch_block_id < o_blocks + gate_blocks) {
      int adj = prefetch_block_id - o_blocks;
      int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
      int epb = (total + gate_blocks - 1) / gate_blocks;
      int start = adj * epb;
      int count = min(epb, total - start);
      if (count > 0)
        ldg_prefetch_weights_l2(gate_weight + start, count);
    } else if (prefetch_block_id < o_blocks + gate_blocks + up_blocks) {
      int adj = prefetch_block_id - o_blocks - gate_blocks;
      int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
      int epb = (total + up_blocks - 1) / up_blocks;
      int start = adj * epb;
      int count = min(epb, total - start);
      if (count > 0)
        ldg_prefetch_weights_l2(up_weight + start, count);
    } else {
      int adj = prefetch_block_id - o_blocks - gate_blocks - up_blocks;
      int db = num_prefetch_blocks - o_blocks - gate_blocks - up_blocks;
      int total = INTERMEDIATE_SIZE * HIDDEN_SIZE;
      int epb = (total + db - 1) / db;
      int start = adj * epb;
      int count = min(epb, total - start);
      if (count > 0)
        ldg_prefetch_weights_l2(down_weight + start, count);
    }
  }

  // --- Barrier #2: KV cache must be ready before attention ---
  if (kv_flag) {
    // All attention blocks need __syncthreads() to finish QK norm writes
    if (block_id < ATTN_BLOCKS) {
      __syncthreads();
    }
    // Block 0: signal KV cache is written (after intra-block sync)
    if (block_id == 0 && threadIdx.x == 0) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      atomicExch(kv_flag, (unsigned int)(layer_idx + 1));
    }
    // Blocks 1-15: wait for KV cache
    if (block_id > 0 && block_id < ATTN_BLOCKS) {
      if (threadIdx.x == 0) {
        volatile unsigned int *vf = (volatile unsigned int *)kv_flag;
        while (*vf < (unsigned int)(layer_idx + 1)) {
        }
      }
      __syncthreads();
    }
    // Blocks 16-127: skip (they don't do attention)
  } else {
    grid.sync();
  }

  // Shared memory for cross-warp reduction of online softmax
  __shared__ float s_max_score[LDG_NUM_WARPS];
  __shared__ float s_sum_exp[LDG_NUM_WARPS];
  __shared__ float s_out_acc[LDG_NUM_WARPS][HEAD_DIM];

  // Each of the 16 attention blocks handles one Q head
  int heads_per_block = (NUM_Q_HEADS + ATTN_BLOCKS - 1) / ATTN_BLOCKS;
  int head_start = block_id * heads_per_block;
  int head_end = min(head_start + heads_per_block, NUM_Q_HEADS);

  for (int qh = head_start; qh < head_end; qh++) {
    int kv_head = qh / (NUM_Q_HEADS / NUM_KV_HEADS);
    const float *q_head = q + qh * HEAD_DIM;
    float *out_head = attn_out + qh * HEAD_DIM;

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int q_idx = lane_id * 4;
    float q_local[4];
    q_local[0] = q_head[q_idx + 0];
    q_local[1] = q_head[q_idx + 1];
    q_local[2] = q_head[q_idx + 2];
    q_local[3] = q_head[q_idx + 3];

    // Each warp processes a subset of cache positions
    for (int pos = warp_id; pos < cache_len; pos += LDG_NUM_WARPS) {
      const __nv_bfloat16 *k_pos =
          k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
      const __nv_bfloat16 *v_pos =
          v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

      // Q @ K with __ldg
      float score = 0.0f;
      uint2 k_u2 = __ldg(reinterpret_cast<const uint2 *>(k_pos + q_idx));
      __nv_bfloat16 *k_ptr = reinterpret_cast<__nv_bfloat16 *>(&k_u2);
      score += q_local[0] * __bfloat162float(k_ptr[0]) +
               q_local[1] * __bfloat162float(k_ptr[1]) +
               q_local[2] * __bfloat162float(k_ptr[2]) +
               q_local[3] * __bfloat162float(k_ptr[3]);
      score = ldg_warp_reduce_sum(score) * attn_scale;
      score = __shfl_sync(0xffffffff, score, 0);

      float old_max = max_score;
      max_score = fmaxf(max_score, score);
      float exp_diff = fast_exp(old_max - max_score);
      sum_exp = sum_exp * exp_diff + fast_exp(score - max_score);

      float weight = fast_exp(score - max_score);
      uint2 v_u2 = __ldg(reinterpret_cast<const uint2 *>(v_pos + q_idx));
      __nv_bfloat16 *v_ptr = reinterpret_cast<__nv_bfloat16 *>(&v_u2);
      out_acc[0] = out_acc[0] * exp_diff + weight * __bfloat162float(v_ptr[0]);
      out_acc[1] = out_acc[1] * exp_diff + weight * __bfloat162float(v_ptr[1]);
      out_acc[2] = out_acc[2] * exp_diff + weight * __bfloat162float(v_ptr[2]);
      out_acc[3] = out_acc[3] * exp_diff + weight * __bfloat162float(v_ptr[3]);
    }

    // Store each warp's partial results to shared memory
    if (lane_id == 0) {
      s_max_score[warp_id] = max_score;
      s_sum_exp[warp_id] = sum_exp;
    }
    int out_base = lane_id * 4;
    s_out_acc[warp_id][out_base + 0] = out_acc[0];
    s_out_acc[warp_id][out_base + 1] = out_acc[1];
    s_out_acc[warp_id][out_base + 2] = out_acc[2];
    s_out_acc[warp_id][out_base + 3] = out_acc[3];
    __syncthreads();

    // Warp 0 combines results from all warps
    if (warp_id == 0) {
      // Find global max across all warps
      float global_max = s_max_score[0];
      for (int w = 1; w < LDG_NUM_WARPS; w++) {
        if (s_max_score[w] >
            -INFINITY) { // Only consider warps that processed positions
          global_max = fmaxf(global_max, s_max_score[w]);
        }
      }

      // Rescale and sum the partial results
      float total_sum_exp = 0.0f;
      float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

      for (int w = 0; w < LDG_NUM_WARPS; w++) {
        if (s_max_score[w] >
            -INFINITY) { // Only consider warps that processed positions
          float scale = fast_exp(s_max_score[w] - global_max);
          total_sum_exp += s_sum_exp[w] * scale;

          int base = lane_id * 4;
          final_out[0] += s_out_acc[w][base + 0] * scale;
          final_out[1] += s_out_acc[w][base + 1] * scale;
          final_out[2] += s_out_acc[w][base + 2] * scale;
          final_out[3] += s_out_acc[w][base + 3] * scale;
        }
      }

      // Write final normalized output
      int base = lane_id * 4;
      out_head[base + 0] = final_out[0] / total_sum_exp;
      out_head[base + 1] = final_out[1] / total_sum_exp;
      out_head[base + 2] = final_out[2] / total_sum_exp;
      out_head[base + 3] = final_out[3] / total_sum_exp;
    }
    __syncthreads();
  }

  // --- Barrier #3: attention output must be complete before O proj ---
  if (attn_flag) {
    // Attention blocks (0-15): signal completion
    if (block_id < ATTN_BLOCKS) {
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      if (threadIdx.x == 0)
        atomicAdd(attn_flag, 1);
    }
    // Non-attention blocks: prefetch O+gate+up weights while waiting
    // (refreshes L2 so O proj + MLP reads hit L2 instead of DRAM)
    if (block_id >= ATTN_BLOCKS && threadIdx.x != 0) {
      int prefetch_id = block_id - ATTN_BLOCKS;
      int npb = num_blocks - ATTN_BLOCKS; // 112 blocks
      // Split: ~20% O proj, ~40% gate, ~40% up (proportional to size)
      int o_total = Q_SIZE * HIDDEN_SIZE;                    // ~2M
      int g_total = HIDDEN_SIZE * INTERMEDIATE_SIZE;         // ~3M
      int u_total = g_total;                                 // ~3M
      int d_total = INTERMEDIATE_SIZE * HIDDEN_SIZE;         // ~3M
      int all_total = o_total + g_total + u_total + d_total; // ~11M
      int per = (all_total + npb - 1) / npb;
      int s = prefetch_id * per, e = min(s + per, all_total);
      for (int i = s + (threadIdx.x - 1); i < e; i += LDG_BLOCK_SIZE - 1) {
        const __nv_bfloat16 *ptr;
        if (i < o_total)
          ptr = o_weight + i;
        else if (i < o_total + g_total)
          ptr = gate_weight + (i - o_total);
        else if (i < o_total + g_total + u_total)
          ptr = up_weight + (i - o_total - g_total);
        else
          ptr = down_weight + (i - o_total - g_total - u_total);
        asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
      }
    }
    // Thread 0: wait for all 16 attention heads to finish
    if (threadIdx.x == 0) {
      unsigned int target = (unsigned int)(ATTN_BLOCKS * (layer_idx + 1));
      volatile unsigned int *vf = (volatile unsigned int *)attn_flag;
      while (*vf < target) {
      }
    }
    __syncthreads();
  } else {
    grid.sync();
  }
}

// =============================================================================
// O Projection + Residual + PostNorm + MLP (all with __ldg)
// =============================================================================

__device__ void ldg_o_proj_postnorm_mlp(
    auto &grid, const __nv_bfloat16 *__restrict__ o_weight,
    const __nv_bfloat16 *__restrict__ post_norm_weight,
    const __nv_bfloat16 *__restrict__ gate_weight,
    const __nv_bfloat16 *__restrict__ up_weight,
    const __nv_bfloat16 *__restrict__ down_weight,
    const float *__restrict__ attn_out, float *__restrict__ g_residual,
    float *__restrict__ g_activations, float *__restrict__ g_mlp_intermediate,
    __nv_bfloat16 *__restrict__ hidden_out) {
  int block_id = blockIdx.x;
  int num_blocks = gridDim.x;
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  __shared__ __align__(16) float s_attn[Q_SIZE];
  __shared__ __align__(16) float s_act[HIDDEN_SIZE];
  __shared__ __align__(16) float s_mlp[INTERMEDIATE_SIZE];

  // Cache attention output for reuse across rows in this block.
  for (int i = threadIdx.x; i < Q_SIZE; i += LDG_BLOCK_SIZE) {
    s_attn[i] = attn_out[i];
  }
  __syncthreads();

  // O Projection + Residual
  int hid_per_block = (HIDDEN_SIZE + num_blocks - 1) / num_blocks;
  int hid_start = block_id * hid_per_block;
  int hid_end = min(hid_start + hid_per_block, HIDDEN_SIZE);

  for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < hid_end) {
      const __nv_bfloat16 *o_row = o_weight + m * Q_SIZE;

      float sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < Q_SIZE; k += WARP_SIZE * 8) {
        uint4 w_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(o_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(s_attn + k);
        float4 a2 = *reinterpret_cast<const float4 *>(s_attn + k + 4);

        sum += __bfloat162float(w_ptr[0]) * a1.x +
               __bfloat162float(w_ptr[1]) * a1.y +
               __bfloat162float(w_ptr[2]) * a1.z +
               __bfloat162float(w_ptr[3]) * a1.w +
               __bfloat162float(w_ptr[4]) * a2.x +
               __bfloat162float(w_ptr[5]) * a2.y +
               __bfloat162float(w_ptr[6]) * a2.z +
               __bfloat162float(w_ptr[7]) * a2.w;
      }

      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        g_activations[m] = sum + g_residual[m];
      }
    }
  }

  grid.sync();

  // ALL blocks compute post-attention RMSNorm redundantly (eliminates
  // grid.sync)
  {
    __shared__ float smem_reduce[LDG_NUM_WARPS];

    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float v = g_activations[i];
      s_act[i] = v;
      local_sum_sq += v * v;
    }

    // Block 0 saves residual for later use
    if (block_id == 0) {
      for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
        g_residual[i] = s_act[i];
      }
    }

    local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0) {
      smem_reduce[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
      float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
      }
    }
    __syncthreads();

    float rstd = smem_reduce[0];

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float w = __bfloat162float(__ldg(post_norm_weight + i));
      s_act[i] = s_act[i] * rstd * w;
    }
    __syncthreads();
  }

  // Gate + Up + SiLU
  int int_per_block = (INTERMEDIATE_SIZE + num_blocks - 1) / num_blocks;
  int int_start = block_id * int_per_block;
  int int_end = min(int_start + int_per_block, INTERMEDIATE_SIZE);

  for (int m_base = int_start; m_base < int_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < int_end) {
      const __nv_bfloat16 *gate_row = gate_weight + m * HIDDEN_SIZE;
      const __nv_bfloat16 *up_row = up_weight + m * HIDDEN_SIZE;

      float gate_sum = 0.0f, up_sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
        uint4 g_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(gate_row + k));
        uint4 u_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(up_row + k));
        __nv_bfloat16 *g_ptr = reinterpret_cast<__nv_bfloat16 *>(&g_u4);
        __nv_bfloat16 *u_ptr = reinterpret_cast<__nv_bfloat16 *>(&u_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(s_act + k);
        float4 a2 = *reinterpret_cast<const float4 *>(s_act + k + 4);

        gate_sum += __bfloat162float(g_ptr[0]) * a1.x +
                    __bfloat162float(g_ptr[1]) * a1.y +
                    __bfloat162float(g_ptr[2]) * a1.z +
                    __bfloat162float(g_ptr[3]) * a1.w +
                    __bfloat162float(g_ptr[4]) * a2.x +
                    __bfloat162float(g_ptr[5]) * a2.y +
                    __bfloat162float(g_ptr[6]) * a2.z +
                    __bfloat162float(g_ptr[7]) * a2.w;

        up_sum += __bfloat162float(u_ptr[0]) * a1.x +
                  __bfloat162float(u_ptr[1]) * a1.y +
                  __bfloat162float(u_ptr[2]) * a1.z +
                  __bfloat162float(u_ptr[3]) * a1.w +
                  __bfloat162float(u_ptr[4]) * a2.x +
                  __bfloat162float(u_ptr[5]) * a2.y +
                  __bfloat162float(u_ptr[6]) * a2.z +
                  __bfloat162float(u_ptr[7]) * a2.w;
      }

      gate_sum = ldg_warp_reduce_sum(gate_sum);
      up_sum = ldg_warp_reduce_sum(up_sum);

      if (lane_id == 0) {
        g_mlp_intermediate[m] = ldg_silu(gate_sum) * up_sum;
      }
    }
  }

  grid.sync();

  for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += LDG_BLOCK_SIZE) {
    s_mlp[i] = g_mlp_intermediate[i];
  }
  __syncthreads();

  // Down projection + residual
  const float *mlp_in = g_mlp_intermediate;
  mlp_in = s_mlp;
  for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
    int m = m_base + warp_id;

    if (m < hid_end) {
      const __nv_bfloat16 *down_row = down_weight + m * INTERMEDIATE_SIZE;

      float sum = 0.0f;
#pragma unroll 4
      for (int k = lane_id * 8; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 8) {
        uint4 d_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(down_row + k));
        __nv_bfloat16 *d_ptr = reinterpret_cast<__nv_bfloat16 *>(&d_u4);
        float4 a1 = *reinterpret_cast<const float4 *>(g_mlp_intermediate + k);
        float4 a2 =
            *reinterpret_cast<const float4 *>(g_mlp_intermediate + k + 4);

        sum += __bfloat162float(d_ptr[0]) * a1.x +
               __bfloat162float(d_ptr[1]) * a1.y +
               __bfloat162float(d_ptr[2]) * a1.z +
               __bfloat162float(d_ptr[3]) * a1.w +
               __bfloat162float(d_ptr[4]) * a2.x +
               __bfloat162float(d_ptr[5]) * a2.y +
               __bfloat162float(d_ptr[6]) * a2.z +
               __bfloat162float(d_ptr[7]) * a2.w;
      }

      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0) {
        hidden_out[m] = __float2bfloat16(sum + g_residual[m]);
      }
    }
  }

  grid.sync();
}

__global__ void ldg_lm_head_phase1(const float *__restrict__ hidden,
                                   const __nv_bfloat16 *__restrict__ weight,
                                   float *__restrict__ block_max_vals,
                                   int *__restrict__ block_max_idxs) {
  __shared__ __align__(16) float s_hidden[HIDDEN_SIZE];

  for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE) {
    s_hidden[i] = hidden[i];
  }
  __syncthreads();

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
  int row_start = blockIdx.x * rows_per_block;
  int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

  float local_max = -INFINITY;
  int local_max_idx = -1;

  int warp_stride = LDG_LM_BLOCK_SIZE / WARP_SIZE;
  int base = row_start + warp_id * LDG_LM_ROWS_PER_WARP;

  for (int m_base = base; m_base < row_end;
       m_base += warp_stride * LDG_LM_ROWS_PER_WARP) {
    int rows[LDG_LM_ROWS_PER_WARP];
    bool valid[LDG_LM_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      rows[r] = m_base + r;
      valid[r] = rows[r] < row_end;
    }

    float sum[LDG_LM_ROWS_PER_WARP];
#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      sum[r] = 0.0f;
    }

#pragma unroll 4
    for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
      float4 a1 = *reinterpret_cast<const float4 *>(s_hidden + k);
      float4 a2 = *reinterpret_cast<const float4 *>(s_hidden + k + 4);

#pragma unroll
      for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
        if (!valid[r]) {
          continue;
        }
        const __nv_bfloat16 *w_row = weight + rows[r] * HIDDEN_SIZE;
        uint4 w_u4 =
            ldg_load_weight_u4(reinterpret_cast<const uint4 *>(w_row + k));
        __nv_bfloat16 *w_ptr = reinterpret_cast<__nv_bfloat16 *>(&w_u4);

        sum[r] += __bfloat162float(w_ptr[0]) * a1.x +
                  __bfloat162float(w_ptr[1]) * a1.y +
                  __bfloat162float(w_ptr[2]) * a1.z +
                  __bfloat162float(w_ptr[3]) * a1.w +
                  __bfloat162float(w_ptr[4]) * a2.x +
                  __bfloat162float(w_ptr[5]) * a2.y +
                  __bfloat162float(w_ptr[6]) * a2.z +
                  __bfloat162float(w_ptr[7]) * a2.w;
      }
    }

#pragma unroll
    for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
      if (!valid[r]) {
        continue;
      }
      float reduced = ldg_warp_reduce_sum(sum[r]);
      if (lane_id == 0 && reduced > local_max) {
        local_max = reduced;
        local_max_idx = rows[r];
      }
    }
  }

  local_max = __shfl_sync(0xffffffff, local_max, 0);
  local_max_idx = __shfl_sync(0xffffffff, local_max_idx, 0);

  __shared__ float warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];
  __shared__ int warp_idx[LDG_LM_BLOCK_SIZE / WARP_SIZE];

  if (lane_id == 0) {
    warp_max[warp_id] = local_max;
    warp_idx[warp_id] = local_max_idx;
  }
  __syncthreads();

  if (warp_id == 0) {
    float max_val = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE)
                        ? warp_max[lane_id]
                        : -INFINITY;
    int max_idx =
        (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      float other_val = __shfl_down_sync(0xffffffff, max_val, offset);
      int other_idx = __shfl_down_sync(0xffffffff, max_idx, offset);
      if (other_val > max_val) {
        max_val = other_val;
        max_idx = other_idx;
      }
    }

    if (lane_id == 0) {
      block_max_vals[blockIdx.x] = max_val;
      block_max_idxs[blockIdx.x] = max_idx;
    }
  }
}

__global__ void ldg_lm_head_phase2(const float *__restrict__ block_max_vals,
                                   const int *__restrict__ block_max_idxs,
                                   int *__restrict__ output_token,
                                   int num_blocks) {
  __shared__ float s_max_vals[1024];
  __shared__ int s_max_idxs[1024];

  int tid = threadIdx.x;

  float local_max = -INFINITY;
  int local_idx = -1;

  for (int i = tid; i < num_blocks; i += blockDim.x) {
    float val = block_max_vals[i];
    if (val > local_max) {
      local_max = val;
      local_idx = block_max_idxs[i];
    }
  }

  s_max_vals[tid] = local_max;
  s_max_idxs[tid] = local_idx;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_max_vals[tid + s] > s_max_vals[tid]) {
        s_max_vals[tid] = s_max_vals[tid + s];
        s_max_idxs[tid] = s_max_idxs[tid + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *output_token = s_max_idxs[0];
  }
}

// =============================================================================
// Persistent (non-cooperative) decode kernel
// =============================================================================

__global__ void
__launch_bounds__(LDG_BLOCK_SIZE, 1) ldg_decode_kernel_persistent(
    const __nv_bfloat16 *__restrict__ embed_weight,
    const LDGLayerWeights *__restrict__ layer_weights,
    const __nv_bfloat16 *__restrict__ final_norm_weight,
    const __nv_bfloat16 *__restrict__ cos_table,
    const __nv_bfloat16 *__restrict__ sin_table,
    __nv_bfloat16 *__restrict__ k_cache, __nv_bfloat16 *__restrict__ v_cache,
    __nv_bfloat16 *__restrict__ hidden_buffer,
    float *__restrict__ g_activations, float *__restrict__ g_residual,
    float *__restrict__ g_q, float *__restrict__ g_k, float *__restrict__ g_v,
    float *__restrict__ g_attn_out, float *__restrict__ g_mlp_intermediate,
    float *__restrict__ g_normalized,
    unsigned int *__restrict__ barrier_counter,
    unsigned int *__restrict__ barrier_sense,
    unsigned int *__restrict__ kv_flag, unsigned int *__restrict__ attn_flag,
    int num_layers, const int *__restrict__ d_position,
    const int *__restrict__ d_token_id, int max_seq_len, float attn_scale) {
  // Read mutable params from device memory (allows CUDA graph replay)
  int position = *d_position;
  int input_token_id = *d_token_id;
  int cache_len = position + 1;
  int block_id = blockIdx.x;
  int num_blocks = gridDim.x;

  // Reset barrier counters + flags on-device
  if (block_id == 0 && threadIdx.x == 0) {
    *barrier_counter = 0;
    *barrier_sense = 0;
    atomicExch(kv_flag, 0u);
    atomicExch(attn_flag, 0u);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    unsigned int arrived = atomicAdd(barrier_counter, 1);
    if (arrived == (unsigned int)num_blocks - 1) {
      *barrier_counter = 0;
      asm volatile("fence.acq_rel.gpu;" ::: "memory");
      atomicAdd(barrier_sense, 1);
    } else {
      volatile unsigned int *vg = (volatile unsigned int *)barrier_sense;
      while (*vg == 0) {
      }
    }
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
  }
  __syncthreads();

  AtomicGridSync grid{barrier_counter, barrier_sense, (unsigned int)gridDim.x,
                      1};

  // First layer reads embed row directly (no separate embed + barrier needed)
  const __nv_bfloat16 *embed_row = embed_weight + input_token_id * HIDDEN_SIZE;

  int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

  for (int layer = 0; layer < num_layers; layer++) {
    const LDGLayerWeights &w = layer_weights[layer];
    __nv_bfloat16 *layer_k_cache = k_cache + layer * kv_cache_layer_stride;
    __nv_bfloat16 *layer_v_cache = v_cache + layer * kv_cache_layer_stride;

    // Layer 0: read directly from embedding table. Other layers: from
    // hidden_buffer.
    const __nv_bfloat16 *layer_input = (layer == 0) ? embed_row : hidden_buffer;

    ldg_matvec_qkv(grid, layer_input, w.input_layernorm_weight, w.q_proj_weight,
                   w.k_proj_weight, w.v_proj_weight, g_activations, g_residual,
                   g_q, g_k, g_v);

    ldg_attention(grid, g_q, g_k, g_v, layer_k_cache, layer_v_cache, g_attn_out,
                  cache_len, max_seq_len, attn_scale, w.q_norm_weight,
                  w.k_norm_weight, cos_table, sin_table, position,
                  w.o_proj_weight, w.gate_proj_weight, w.up_proj_weight,
                  w.down_proj_weight, kv_flag, attn_flag, layer);

    ldg_o_proj_postnorm_mlp(grid, w.o_proj_weight, w.post_attn_layernorm_weight,
                            w.gate_proj_weight, w.up_proj_weight,
                            w.down_proj_weight, g_attn_out, g_residual,
                            g_activations, g_mlp_intermediate, hidden_buffer);
  }

  // Final RMSNorm
  if (block_id == 0) {
    __shared__ float smem_reduce[LDG_NUM_WARPS];
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float v = __bfloat162float(hidden_buffer[i]);
      g_activations[i] = v;
      local_sum_sq += v * v;
    }
    local_sum_sq = ldg_warp_reduce_sum(local_sum_sq);
    if (lane_id == 0)
      smem_reduce[warp_id] = local_sum_sq;
    __syncthreads();
    if (warp_id == 0) {
      float sum = (lane_id < LDG_NUM_WARPS) ? smem_reduce[lane_id] : 0.0f;
      sum = ldg_warp_reduce_sum(sum);
      if (lane_id == 0)
        smem_reduce[0] = rsqrtf(sum / float(HIDDEN_SIZE) + LDG_RMS_EPS);
    }
    __syncthreads();
    float rstd = smem_reduce[0];
    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_BLOCK_SIZE) {
      float wt = __bfloat162float(__ldg(final_norm_weight + i));
      g_normalized[i] = g_activations[i] * rstd * wt;
    }
  }
}

// Device-side step update: copies LM head output -> next token input,
// increments position, logs output token. Uses device-side step counter
// so the kernel arg is fixed (graph-compatible).
__global__ void ldg_update_step(const int *__restrict__ lm_output,
                                int *__restrict__ d_token_id,
                                int *__restrict__ d_position,
                                int *__restrict__ output_log,
                                int *__restrict__ d_step_counter) {
  int tok = *lm_output;
  int step = *d_step_counter;
  *d_token_id = tok;
  *d_position = *d_position + 1;
  output_log[step] = tok;
  *d_step_counter = step + 1;
}

// =============================================================================
// Launch functions
// =============================================================================

static unsigned int *d_barrier_counter = nullptr;
static unsigned int *d_barrier_sense = nullptr;
static unsigned int *d_kv_flag = nullptr;
static unsigned int *d_attn_flag = nullptr;
static int *d_mutable_position = nullptr;
static int *d_mutable_token_id = nullptr;
int *h_pinned_position = nullptr;
int *h_pinned_token_id = nullptr;

static void ensure_barrier_alloc() {
  if (!d_barrier_counter) {
    cudaMalloc(&d_barrier_counter, sizeof(unsigned int));
    cudaMalloc(&d_barrier_sense, sizeof(unsigned int));
    cudaMalloc(&d_kv_flag, sizeof(unsigned int));
    cudaMalloc(&d_attn_flag, sizeof(unsigned int));
    cudaMalloc(&d_mutable_position, sizeof(int));
    cudaMalloc(&d_mutable_token_id, sizeof(int));
    cudaHostAlloc(&h_pinned_position, sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_pinned_token_id, sizeof(int), cudaHostAllocDefault);
    cudaMemset(d_barrier_counter, 0, sizeof(unsigned int));
    cudaMemset(d_barrier_sense, 0, sizeof(unsigned int));
    cudaMemset(d_kv_flag, 0, sizeof(unsigned int));
    cudaMemset(d_attn_flag, 0, sizeof(unsigned int));
  }
}

static inline void ldg_configure_kernel_attributes(); // forward decl

extern "C" void launch_ldg_decode_persistent(
    int input_token_id, int *output_token_id, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int position, int cache_len,
    int max_seq_len, float attn_scale, cudaStream_t stream) {
  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();

  // Write mutable params via pinned host memory (barriers reset on-device)
  *h_pinned_position = position;
  *h_pinned_token_id = input_token_id;
  cudaMemcpyAsync(d_mutable_position, h_pinned_position, sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_mutable_token_id, h_pinned_token_id, sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  ldg_decode_kernel_persistent<<<LDG_NUM_BLOCKS, LDG_BLOCK_SIZE, 0, stream>>>(
      (const __nv_bfloat16 *)embed_weight, layer_weights,
      (const __nv_bfloat16 *)final_norm_weight,
      (const __nv_bfloat16 *)cos_table, (const __nv_bfloat16 *)sin_table,
      (__nv_bfloat16 *)k_cache, (__nv_bfloat16 *)v_cache,
      (__nv_bfloat16 *)hidden_buffer, (float *)g_activations,
      (float *)g_residual, (float *)g_q, (float *)g_k, (float *)g_v,
      (float *)g_attn_out, (float *)g_mlp_intermediate, (float *)g_normalized,
      d_barrier_counter, d_barrier_sense, d_kv_flag, d_attn_flag, num_layers,
      d_mutable_position, d_mutable_token_id, max_seq_len, attn_scale);

  ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
      (const float *)g_normalized, (const __nv_bfloat16 *)lm_head_weight,
      (float *)block_max_vals, (int *)block_max_idxs);

  ldg_lm_head_phase2<<<1, 256, 0, stream>>>((const float *)block_max_vals,
                                            (const int *)block_max_idxs,
                                            output_token_id, LDG_LM_NUM_BLOCKS);
}

// N-step generate with NO per-step CPU sync. All steps queued back-to-back.
// The update kernel feeds output_token -> d_token_id on device between steps.
extern "C" void launch_ldg_generate_nosync(
    int first_token_id, int num_steps, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs,
    int *output_log, // device int[num_steps]: all generated tokens
    int num_layers, int start_position, int max_seq_len, float attn_scale,
    cudaStream_t stream) {

  ldg_configure_kernel_attributes();
  ensure_barrier_alloc();

  // Allocate device-side step counter
  static int *d_step_counter = nullptr;
  static int *d_output_token = nullptr;
  if (!d_step_counter) {
    cudaMalloc(&d_step_counter, sizeof(int));
    cudaMalloc(&d_output_token, sizeof(int));
  }
  cudaMemsetAsync(d_step_counter, 0, sizeof(int), stream);

  // Set initial position and token_id on device
  *h_pinned_position = start_position;
  *h_pinned_token_id = first_token_id;
  cudaMemcpyAsync(d_mutable_position, h_pinned_position, sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_mutable_token_id, h_pinned_token_id, sizeof(int),
                  cudaMemcpyHostToDevice, stream);

  // Launch all N steps back-to-back with NO CPU sync between them.
  // Each step: persistent kernel -> LM head phase 1 -> LM head phase 2 ->
  // update
  for (int step = 0; step < num_steps; step++) {
    ldg_decode_kernel_persistent<<<LDG_NUM_BLOCKS, LDG_BLOCK_SIZE, 0, stream>>>(
        (const __nv_bfloat16 *)embed_weight, layer_weights,
        (const __nv_bfloat16 *)final_norm_weight,
        (const __nv_bfloat16 *)cos_table, (const __nv_bfloat16 *)sin_table,
        (__nv_bfloat16 *)k_cache, (__nv_bfloat16 *)v_cache,
        (__nv_bfloat16 *)hidden_buffer, (float *)g_activations,
        (float *)g_residual, (float *)g_q, (float *)g_k, (float *)g_v,
        (float *)g_attn_out, (float *)g_mlp_intermediate, (float *)g_normalized,
        d_barrier_counter, d_barrier_sense, d_kv_flag, d_attn_flag, num_layers,
        d_mutable_position, d_mutable_token_id, max_seq_len, attn_scale);

    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float *)g_normalized, (const __nv_bfloat16 *)lm_head_weight,
        (float *)block_max_vals, (int *)block_max_idxs);

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float *)block_max_vals, (const int *)block_max_idxs,
        d_output_token, LDG_LM_NUM_BLOCKS);

    // Update step: feed output token back, increment position, log result
    ldg_update_step<<<1, 1, 0, stream>>>(d_output_token, d_mutable_token_id,
                                         d_mutable_position, output_log,
                                         d_step_counter);
  }
}

static inline void ldg_configure_kernel_attributes() {
  static bool configured = false;
  if (configured) {
    return;
  }
  configured = true;
  cudaFuncSetAttribute(ldg_decode_kernel_persistent,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);
  cudaFuncSetAttribute(ldg_lm_head_phase1,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);
  cudaFuncSetAttribute(ldg_lm_head_phase2,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxL1);
}
