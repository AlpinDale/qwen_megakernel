/**
 * Fused single-kernel decode for Qwen3-0.6B on RTX 5090.
 *
 * Everything — embedding lookup, 28 transformer layers (RMSNorm, QKV, RoPE,
 * attention, O-proj, MLP), and final norm — runs inside one cooperative kernel
 * launch.  The LM head (vocab projection + argmax) is a separate non-cooperative
 * kernel launched immediately after.
 *
 * Optimized for: NVIDIA RTX 5090 (sm_120, 170 SMs, 96 MB L2)
 * Model:         Qwen/Qwen3-0.6B (bf16 weights)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// =============================================================================
// Model constants (Qwen3-0.6B)
// =============================================================================

constexpr int WARP_SIZE        = 32;
constexpr int HIDDEN_SIZE      = 1024;
constexpr int INTERMEDIATE_SIZE = 3072;
constexpr int NUM_Q_HEADS      = 16;
constexpr int NUM_KV_HEADS     = 8;
constexpr int HEAD_DIM         = 128;
constexpr int Q_SIZE           = NUM_Q_HEADS  * HEAD_DIM;   // 2048
constexpr int KV_SIZE          = NUM_KV_HEADS * HEAD_DIM;   // 1024

#if defined(LDG_LM_USE_WMMA)
#include <mma.h>
using namespace nvcuda;
#endif

// Configuration (overridable via -DLDG_NUM_BLOCKS / -DLDG_BLOCK_SIZE)
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

constexpr int LDG_PHASE_COUNT = 6;
#if defined(LDG_PHASE_PROFILE)
__device__ __managed__ unsigned long long g_phase_cycles[LDG_PHASE_COUNT];
#endif

// L2 prefetch cache hints (from CUTLASS / NVIDIA docs)
constexpr uint64_t LDG_EVICT_NORMAL = 0x1000000000000000;
constexpr uint64_t LDG_EVICT_FIRST = 0x12F0000000000000;
constexpr uint64_t LDG_EVICT_LAST = 0x14F0000000000000;

#ifndef LDG_PREFETCH_CHUNK_BYTES
#define LDG_PREFETCH_CHUNK_BYTES 4096
#endif

#ifndef LDG_PREFETCH_DISTANCE
#define LDG_PREFETCH_DISTANCE 1
#endif

#ifndef LDG_SET_L1_CARVEOUT
#define LDG_SET_L1_CARVEOUT 1
#endif

struct LDGLayerWeights {
    const __nv_bfloat16* input_layernorm_weight;
    const __nv_bfloat16* q_proj_weight;
    const __nv_bfloat16* k_proj_weight;
    const __nv_bfloat16* v_proj_weight;
    const __nv_bfloat16* q_norm_weight;
    const __nv_bfloat16* k_norm_weight;
    const __nv_bfloat16* o_proj_weight;
    const __nv_bfloat16* post_attn_layernorm_weight;
    const __nv_bfloat16* gate_proj_weight;
    const __nv_bfloat16* up_proj_weight;
    const __nv_bfloat16* down_proj_weight;
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

__device__ __forceinline__ float ldg_silu(float x) {
    return x / (1.0f + __expf(-x));
}

#ifndef LDG_PREFETCH_L2_POLICY
#define LDG_PREFETCH_L2_POLICY LDG_EVICT_LAST
#endif

__device__ __forceinline__ void ldg_prefetch_l2(const void* src, int bytes, uint64_t cache_policy) {
#if defined(LDG_USE_BULK_PREFETCH)
    asm volatile(
        "cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
        :: "l"(src), "r"(bytes), "l"(cache_policy)
        : "memory"
    );
#else
    (void)src;
    (void)bytes;
    (void)cache_policy;
#endif
}

__device__ __forceinline__ void ldg_prefetch_row(const __nv_bfloat16* row, int elements) {
#if defined(LDG_USE_BULK_PREFETCH)
    const char* ptr = reinterpret_cast<const char*>(row);
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if (addr & 0xF) {
        return;
    }
    int bytes = elements * int(sizeof(__nv_bfloat16));
    for (int offset = 0; offset < bytes; offset += LDG_PREFETCH_CHUNK_BYTES) {
        int chunk = bytes - offset;
        if (chunk > LDG_PREFETCH_CHUNK_BYTES) {
            chunk = LDG_PREFETCH_CHUNK_BYTES;
        }
        chunk = (chunk / 16) * 16;
        if (chunk == 0) {
            break;
        }
        ldg_prefetch_l2(ptr + offset, chunk, LDG_PREFETCH_L2_POLICY);
    }
#else
    (void)row;
    (void)elements;
#endif
}

#if defined(LDG_O_PROJ_ASYNC_SMEM)
#ifndef LDG_O_PROJ_TILE_K
#define LDG_O_PROJ_TILE_K 256
#endif
static_assert(LDG_O_PROJ_TILE_K % (WARP_SIZE * 8) == 0, "LDG_O_PROJ_TILE_K must be multiple of 256 elements.");
static_assert(Q_SIZE % LDG_O_PROJ_TILE_K == 0, "LDG_O_PROJ_TILE_K must divide Q_SIZE.");
#endif

#if defined(LDG_LM_ASYNC_SMEM)
#ifndef LDG_LM_TILE_K
#define LDG_LM_TILE_K 256
#endif
static_assert(LDG_LM_TILE_K % (WARP_SIZE * 8) == 0, "LDG_LM_TILE_K must be multiple of 256 elements.");
static_assert(HIDDEN_SIZE % LDG_LM_TILE_K == 0, "LDG_LM_TILE_K must divide HIDDEN_SIZE.");
#endif

#if defined(LDG_O_PROJ_ASYNC_SMEM) || defined(LDG_LM_ASYNC_SMEM)
__device__ __forceinline__ void ldg_cp_async_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int smem = static_cast<unsigned int>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :: "r"(smem), "l"(gmem_ptr));
}

__device__ __forceinline__ void ldg_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void ldg_cp_async_wait() {
    asm volatile("cp.async.wait_group 0;");
}
#endif

// Cache-hinted weight loads (optional)
__device__ __forceinline__ uint2 ldg_load_weight_u2(const uint2* ptr) {
#if defined(LDG_WEIGHT_LDCS)
    uint2 out;
    asm volatile("ld.global.L1::no_allocate.v2.b32 {%0, %1}, [%2];"
                 : "=r"(out.x), "=r"(out.y) : "l"(ptr));
    return out;
#elif defined(LDG_WEIGHT_LDCA)
    uint2 out;
    asm volatile("ld.global.L1::evict_last.v2.b32 {%0, %1}, [%2];"
                 : "=r"(out.x), "=r"(out.y) : "l"(ptr));
    return out;
#else
    return __ldg(ptr);
#endif
}

__device__ __forceinline__ uint4 ldg_load_weight_u4(const uint4* ptr) {
#if defined(LDG_WEIGHT_LDCS)
    uint4 out;
    asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
    return out;
#elif defined(LDG_WEIGHT_LDCA)
    uint4 out;
    asm volatile("ld.global.L1::evict_last.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w) : "l"(ptr));
    return out;
#else
    return __ldg(ptr);
#endif
}

__device__ __forceinline__ void ldg_load_weight_u8(unsigned int out[8], const __nv_bfloat16* ptr) {
#if defined(LDG_WEIGHT_L2_EVICT_FIRST)
    asm volatile("ld.global.L1::no_allocate.L2::evict_first.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
                   "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
                 : "l"(ptr));
#elif defined(LDG_WEIGHT_L2_EVICT_LAST)
    asm volatile("ld.global.L1::no_allocate.L2::evict_last.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
                   "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
                 : "l"(ptr));
#elif defined(LDG_WEIGHT_LDCS)
    asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
                   "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
                 : "l"(ptr));
#elif defined(LDG_WEIGHT_LDCA)
    asm volatile("ld.global.L1::evict_last.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
                   "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
                 : "l"(ptr));
#else
    uint4 lo = ldg_load_weight_u4(reinterpret_cast<const uint4*>(ptr));
    uint4 hi = ldg_load_weight_u4(reinterpret_cast<const uint4*>(ptr) + 1);
    out[0] = lo.x; out[1] = lo.y; out[2] = lo.z; out[3] = lo.w;
    out[4] = hi.x; out[5] = hi.y; out[6] = hi.z; out[7] = hi.w;
#endif
}

// =============================================================================
// Optimized matvec with __ldg and aggressive unrolling
// =============================================================================

__device__ void ldg_matvec_qkv(
    cg::grid_group& grid,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ q_weight,
    const __nv_bfloat16* __restrict__ k_weight,
    const __nv_bfloat16* __restrict__ v_weight,
    float* __restrict__ g_normalized,
    float* __restrict__ g_residual,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out
) {
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

#if defined(LDG_PREFETCH_NEXT_ROWS)
        if (warp_id == 0 && lane_id == 0) {
            int next_base = m_base + LDG_NUM_WARPS * LDG_PREFETCH_DISTANCE;
            for (int w = 0; w < LDG_NUM_WARPS; w++) {
                int m_pf = next_base + w;
                if (m_pf >= row_end) {
                    break;
                }
                const __nv_bfloat16* weight_row_pf;
                if (m_pf < Q_SIZE) {
                    weight_row_pf = q_weight + m_pf * HIDDEN_SIZE;
                } else if (m_pf < Q_SIZE + KV_SIZE) {
                    weight_row_pf = k_weight + (m_pf - Q_SIZE) * HIDDEN_SIZE;
                } else {
                    weight_row_pf = v_weight + (m_pf - Q_SIZE - KV_SIZE) * HIDDEN_SIZE;
                }
                ldg_prefetch_row(weight_row_pf, HIDDEN_SIZE);
            }
        }
#endif

        if (m < row_end) {
            const __nv_bfloat16* weight_row;
            float* output_ptr;

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
#if defined(LDG_USE_UINT8)
            #pragma unroll 2
            for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {
                unsigned int w_u8[8];
                ldg_load_weight_u8(w_u8, weight_row + k);
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(w_u8);
                float4 a1 = *reinterpret_cast<const float4*>(s_norm + k);
                float4 a2 = *reinterpret_cast<const float4*>(s_norm + k + 4);
                float4 a3 = *reinterpret_cast<const float4*>(s_norm + k + 8);
                float4 a4 = *reinterpret_cast<const float4*>(s_norm + k + 12);

                sum += __bfloat162float(w_ptr[0]) * a1.x +
                       __bfloat162float(w_ptr[1]) * a1.y +
                       __bfloat162float(w_ptr[2]) * a1.z +
                       __bfloat162float(w_ptr[3]) * a1.w +
                       __bfloat162float(w_ptr[4]) * a2.x +
                       __bfloat162float(w_ptr[5]) * a2.y +
                       __bfloat162float(w_ptr[6]) * a2.z +
                       __bfloat162float(w_ptr[7]) * a2.w +
                       __bfloat162float(w_ptr[8]) * a3.x +
                       __bfloat162float(w_ptr[9]) * a3.y +
                       __bfloat162float(w_ptr[10]) * a3.z +
                       __bfloat162float(w_ptr[11]) * a3.w +
                       __bfloat162float(w_ptr[12]) * a4.x +
                       __bfloat162float(w_ptr[13]) * a4.y +
                       __bfloat162float(w_ptr[14]) * a4.z +
                       __bfloat162float(w_ptr[15]) * a4.w;
            }
#elif defined(LDG_USE_UINT4)
            #pragma unroll 4
            for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                uint4 w_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);
                float4 a1 = *reinterpret_cast<const float4*>(s_norm + k);
                float4 a2 = *reinterpret_cast<const float4*>(s_norm + k + 4);

                sum += __bfloat162float(w_ptr[0]) * a1.x +
                       __bfloat162float(w_ptr[1]) * a1.y +
                       __bfloat162float(w_ptr[2]) * a1.z +
                       __bfloat162float(w_ptr[3]) * a1.w +
                       __bfloat162float(w_ptr[4]) * a2.x +
                       __bfloat162float(w_ptr[5]) * a2.y +
                       __bfloat162float(w_ptr[6]) * a2.z +
                       __bfloat162float(w_ptr[7]) * a2.w;
            }
#else
            // Use vec4 loads with __ldg through uint2
            #pragma unroll 8
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = ldg_load_weight_u2(reinterpret_cast<const uint2*>(weight_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * s_norm[k] +
                       __bfloat162float(w_ptr[1]) * s_norm[k+1] +
                       __bfloat162float(w_ptr[2]) * s_norm[k+2] +
                       __bfloat162float(w_ptr[3]) * s_norm[k+3];
            }
#endif

            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) {
                *output_ptr = sum;
            }
        }
    }

    grid.sync();
}

// =============================================================================
// QK Norm + RoPE + KV Cache
// =============================================================================

__device__ void ldg_qk_norm_rope_cache(
    cg::grid_group& grid,
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ v,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    int position,
    int max_seq_len
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    const __nv_bfloat16* cos_pos = cos_table + position * HEAD_DIM;
    const __nv_bfloat16* sin_pos = sin_table + position * HEAD_DIM;

    int q_heads_per_block = (NUM_Q_HEADS + num_blocks - 1) / num_blocks;
    int q_head_start = block_id * q_heads_per_block;
    int q_head_end = min(q_head_start + q_heads_per_block, NUM_Q_HEADS);

    for (int h = q_head_start + warp_id; h < q_head_end; h += LDG_NUM_WARPS) {
        float* q_head = q + h * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += q_head[i] * q_head[i];
        }
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float q_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            q_local[j] = q_head[i] * scale * __bfloat162float(__ldg(q_norm_weight + i));
        }

        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));

            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, q_local[pair_j], pair_idx % WARP_SIZE);

            if (i < HEAD_DIM/2) {
                q_head[i] = q_local[j] * cos_v - pair_v * sin_v;
            } else {
                q_head[i] = pair_v * sin_v + q_local[j] * cos_v;
            }
        }
    }

    int k_heads_per_block = (NUM_KV_HEADS + num_blocks - 1) / num_blocks;
    int k_head_start = block_id * k_heads_per_block;
    int k_head_end = min(k_head_start + k_heads_per_block, NUM_KV_HEADS);

    for (int h = k_head_start + warp_id; h < k_head_end; h += LDG_NUM_WARPS) {
        float* k_head = k + h * HEAD_DIM;
        const float* v_head = v + h * HEAD_DIM;
        __nv_bfloat16* k_cache_head = k_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;
        __nv_bfloat16* v_cache_head = v_cache + h * max_seq_len * HEAD_DIM + position * HEAD_DIM;

        float sum_sq = 0.0f;
        for (int i = lane_id; i < HEAD_DIM; i += WARP_SIZE) {
            sum_sq += k_head[i] * k_head[i];
        }
        sum_sq = ldg_warp_reduce_sum(sum_sq);
        float scale = rsqrtf(sum_sq / float(HEAD_DIM) + LDG_RMS_EPS);
        scale = __shfl_sync(0xffffffff, scale, 0);

        float k_local[HEAD_DIM / WARP_SIZE];
        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            k_local[j] = k_head[i] * scale * __bfloat162float(__ldg(k_norm_weight + i));
        }

        #pragma unroll
        for (int i = lane_id, j = 0; i < HEAD_DIM; i += WARP_SIZE, j++) {
            float cos_v = __bfloat162float(__ldg(cos_pos + i));
            float sin_v = __bfloat162float(__ldg(sin_pos + i));

            int pair_offset = (i < HEAD_DIM/2) ? HEAD_DIM/2 : -HEAD_DIM/2;
            int pair_idx = i + pair_offset;
            int pair_j = pair_idx / WARP_SIZE;
            float pair_v = __shfl_sync(0xffffffff, k_local[pair_j], pair_idx % WARP_SIZE);

            float k_final;
            if (i < HEAD_DIM/2) {
                k_final = k_local[j] * cos_v - pair_v * sin_v;
            } else {
                k_final = pair_v * sin_v + k_local[j] * cos_v;
            }
            k_head[i] = k_final;
            k_cache_head[i] = __float2bfloat16(k_final);
            v_cache_head[i] = __float2bfloat16(v_head[i]);
        }
    }

    grid.sync();
}

// =============================================================================
// Attention with __ldg for KV cache + block divergence for prefetching
// =============================================================================

// Prefetch weights into L2 cache using __ldg reads
__device__ void ldg_prefetch_weights_l2(
    const __nv_bfloat16* __restrict__ weights,
    int num_elements
) {
    // Bulk L2 prefetch (Blackwell) or fallback to cached loads
#if defined(LDG_USE_BULK_PREFETCH)
    if (threadIdx.x == 0) {
        ldg_prefetch_row(weights, num_elements);
    }
#else
    // Each thread prefetches strided elements to warm L2 cache
    float dummy = 0.0f;
    for (int i = threadIdx.x; i < num_elements; i += LDG_BLOCK_SIZE) {
        // Read but don't use - compiler won't optimize out due to volatile-like __ldg
        dummy += __bfloat162float(__ldg(weights + i));
    }
    // Prevent optimization (result stored to shared but never used)
    __shared__ float s_dummy;
    if (threadIdx.x == 0) s_dummy = dummy;
#endif
}

__device__ void ldg_attention(
    cg::grid_group& grid,
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ attn_out,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    // Weights to prefetch during attention (for blocks not doing attention)
    const __nv_bfloat16* __restrict__ o_weight,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    const __nv_bfloat16* __restrict__ down_weight
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // Only first NUM_Q_HEADS blocks do attention work
    // Remaining blocks prefetch post-attention weights to warm L2 cache
    const int ATTN_BLOCKS = NUM_Q_HEADS;  // 16 blocks for 16 Q heads

    if (block_id >= ATTN_BLOCKS) {
        // Distribute prefetch work across non-attention blocks
        // Weight sizes: O=2M, gate=3M, up=3M, down=3M (total 11M elements = 22 MB)
        // 96 MB L2 can hold all of it
        int prefetch_block_id = block_id - ATTN_BLOCKS;
        int num_prefetch_blocks = num_blocks - ATTN_BLOCKS;

        // Distribute proportionally: O gets 2/11, gate/up/down get 3/11 each
        int o_blocks = num_prefetch_blocks * 2 / 11;
        int gate_blocks = num_prefetch_blocks * 3 / 11;
        int up_blocks = num_prefetch_blocks * 3 / 11;
        // down_blocks gets the remainder
        if (o_blocks < 1) o_blocks = 1;

        if (prefetch_block_id < o_blocks) {
            // O projection: Q_SIZE x HIDDEN_SIZE = 2M elements
            int total = Q_SIZE * HIDDEN_SIZE;
            int elems_per_block = (total + o_blocks - 1) / o_blocks;
            int start = prefetch_block_id * elems_per_block;
            int count = min(elems_per_block, total - start);
            if (count > 0) ldg_prefetch_weights_l2(o_weight + start, count);
        }
        else if (prefetch_block_id < o_blocks + gate_blocks) {
            // Gate projection: HIDDEN_SIZE x INTERMEDIATE_SIZE = 3M elements
            int adjusted_id = prefetch_block_id - o_blocks;
            int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
            int elems_per_block = (total + gate_blocks - 1) / gate_blocks;
            int start = adjusted_id * elems_per_block;
            int count = min(elems_per_block, total - start);
            if (count > 0) ldg_prefetch_weights_l2(gate_weight + start, count);
        }
        else if (prefetch_block_id < o_blocks + gate_blocks + up_blocks) {
            // Up projection: same as gate
            int adjusted_id = prefetch_block_id - o_blocks - gate_blocks;
            int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
            int elems_per_block = (total + up_blocks - 1) / up_blocks;
            int start = adjusted_id * elems_per_block;
            int count = min(elems_per_block, total - start);
            if (count > 0) ldg_prefetch_weights_l2(up_weight + start, count);
        }
        else {
            // Down projection: INTERMEDIATE_SIZE x HIDDEN_SIZE = 3M elements
            int adjusted_id = prefetch_block_id - o_blocks - gate_blocks - up_blocks;
            int down_blocks = num_prefetch_blocks - o_blocks - gate_blocks - up_blocks;
            int total = INTERMEDIATE_SIZE * HIDDEN_SIZE;
            int elems_per_block = (total + down_blocks - 1) / down_blocks;
            int start = adjusted_id * elems_per_block;
            int count = min(elems_per_block, total - start);
            if (count > 0) ldg_prefetch_weights_l2(down_weight + start, count);
        }

        grid.sync();
        return;
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
        const float* q_head = q + qh * HEAD_DIM;
        float* out_head = attn_out + qh * HEAD_DIM;

        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float out_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#if defined(LDG_ATTENTION_VEC4)
        int q_idx = lane_id * 4;
        float q_local[4];
        q_local[0] = q_head[q_idx + 0];
        q_local[1] = q_head[q_idx + 1];
        q_local[2] = q_head[q_idx + 2];
        q_local[3] = q_head[q_idx + 3];
#endif

        // Each warp processes a subset of cache positions
        for (int pos = warp_id; pos < cache_len; pos += LDG_NUM_WARPS) {
            const __nv_bfloat16* k_pos = k_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;
            const __nv_bfloat16* v_pos = v_cache + kv_head * max_seq_len * HEAD_DIM + pos * HEAD_DIM;

            // Q @ K with __ldg
            float score = 0.0f;
#if defined(LDG_ATTENTION_VEC4)
            uint2 k_u2 = __ldg(reinterpret_cast<const uint2*>(k_pos + q_idx));
            __nv_bfloat16* k_ptr = reinterpret_cast<__nv_bfloat16*>(&k_u2);
            score += q_local[0] * __bfloat162float(k_ptr[0]) +
                     q_local[1] * __bfloat162float(k_ptr[1]) +
                     q_local[2] * __bfloat162float(k_ptr[2]) +
                     q_local[3] * __bfloat162float(k_ptr[3]);
#else
            for (int d = lane_id; d < HEAD_DIM; d += WARP_SIZE) {
                score += q_head[d] * __bfloat162float(__ldg(k_pos + d));
            }
#endif
            score = ldg_warp_reduce_sum(score) * attn_scale;
            score = __shfl_sync(0xffffffff, score, 0);

            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_diff = __expf(old_max - max_score);
            sum_exp = sum_exp * exp_diff + __expf(score - max_score);

            float weight = __expf(score - max_score);
#if defined(LDG_ATTENTION_VEC4)
            uint2 v_u2 = __ldg(reinterpret_cast<const uint2*>(v_pos + q_idx));
            __nv_bfloat16* v_ptr = reinterpret_cast<__nv_bfloat16*>(&v_u2);
            out_acc[0] = out_acc[0] * exp_diff + weight * __bfloat162float(v_ptr[0]);
            out_acc[1] = out_acc[1] * exp_diff + weight * __bfloat162float(v_ptr[1]);
            out_acc[2] = out_acc[2] * exp_diff + weight * __bfloat162float(v_ptr[2]);
            out_acc[3] = out_acc[3] * exp_diff + weight * __bfloat162float(v_ptr[3]);
#else
            #pragma unroll
            for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                out_acc[j] = out_acc[j] * exp_diff + weight * __bfloat162float(__ldg(v_pos + d));
            }
#endif
        }

        // Store each warp's partial results to shared memory
        if (lane_id == 0) {
            s_max_score[warp_id] = max_score;
            s_sum_exp[warp_id] = sum_exp;
        }
#if defined(LDG_ATTENTION_VEC4)
        int out_base = lane_id * 4;
        s_out_acc[warp_id][out_base + 0] = out_acc[0];
        s_out_acc[warp_id][out_base + 1] = out_acc[1];
        s_out_acc[warp_id][out_base + 2] = out_acc[2];
        s_out_acc[warp_id][out_base + 3] = out_acc[3];
#else
        #pragma unroll
        for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
            s_out_acc[warp_id][d] = out_acc[j];
        }
#endif
        __syncthreads();

        // Warp 0 combines results from all warps
        if (warp_id == 0) {
            // Find global max across all warps
            float global_max = s_max_score[0];
            for (int w = 1; w < LDG_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {  // Only consider warps that processed positions
                    global_max = fmaxf(global_max, s_max_score[w]);
                }
            }

            // Rescale and sum the partial results
            float total_sum_exp = 0.0f;
            float final_out[4] = {0.0f, 0.0f, 0.0f, 0.0f};

            for (int w = 0; w < LDG_NUM_WARPS; w++) {
                if (s_max_score[w] > -INFINITY) {  // Only consider warps that processed positions
                    float scale = __expf(s_max_score[w] - global_max);
                    total_sum_exp += s_sum_exp[w] * scale;

#if defined(LDG_ATTENTION_VEC4)
                    int base = lane_id * 4;
                    final_out[0] += s_out_acc[w][base + 0] * scale;
                    final_out[1] += s_out_acc[w][base + 1] * scale;
                    final_out[2] += s_out_acc[w][base + 2] * scale;
                    final_out[3] += s_out_acc[w][base + 3] * scale;
#else
                    #pragma unroll
                    for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                        final_out[j] += s_out_acc[w][d] * scale;
                    }
#endif
                }
            }

            // Write final normalized output
#if defined(LDG_ATTENTION_VEC4)
            int base = lane_id * 4;
            out_head[base + 0] = final_out[0] / total_sum_exp;
            out_head[base + 1] = final_out[1] / total_sum_exp;
            out_head[base + 2] = final_out[2] / total_sum_exp;
            out_head[base + 3] = final_out[3] / total_sum_exp;
#else
            #pragma unroll
            for (int d = lane_id, j = 0; d < HEAD_DIM; d += WARP_SIZE, j++) {
                out_head[d] = final_out[j] / total_sum_exp;
            }
#endif
        }
        __syncthreads();
    }

    grid.sync();
}

// =============================================================================
// O Projection + Residual + PostNorm + MLP (all with __ldg)
// =============================================================================

__device__ void ldg_o_proj_postnorm_mlp(
    cg::grid_group& grid,
    const __nv_bfloat16* __restrict__ o_weight,
    const __nv_bfloat16* __restrict__ post_norm_weight,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    const __nv_bfloat16* __restrict__ down_weight,
    const float* __restrict__ attn_out,
    float* __restrict__ g_residual,
    float* __restrict__ g_activations,
    float* __restrict__ g_mlp_intermediate,
    __nv_bfloat16* __restrict__ hidden_out
) {
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    __shared__ __align__(16) float s_attn[Q_SIZE];
    __shared__ __align__(16) float s_act[HIDDEN_SIZE];
#if defined(LDG_MLP_SMEM)
    __shared__ __align__(16) float s_mlp[INTERMEDIATE_SIZE];
#endif
#if defined(LDG_O_PROJ_ASYNC_SMEM)
    __shared__ __align__(16) __nv_bfloat16 s_o_wt[2][LDG_NUM_WARPS][LDG_O_PROJ_TILE_K];
#endif

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

#if defined(LDG_PREFETCH_NEXT_ROWS)
        if (warp_id == 0 && lane_id == 0) {
            int next_base = m_base + LDG_NUM_WARPS * LDG_PREFETCH_DISTANCE;
            for (int w = 0; w < LDG_NUM_WARPS; w++) {
                int m_pf = next_base + w;
                if (m_pf >= hid_end) {
                    break;
                }
                const __nv_bfloat16* o_row_pf = o_weight + m_pf * Q_SIZE;
                ldg_prefetch_row(o_row_pf, Q_SIZE);
            }
        }
#endif

        if (m < hid_end) {
            const __nv_bfloat16* o_row = o_weight + m * Q_SIZE;

            float sum = 0.0f;
#if defined(LDG_O_PROJ_ASYNC_SMEM)
            __nv_bfloat16* smem_tile0 = &s_o_wt[0][warp_id][0];
            __nv_bfloat16* smem_tile1 = &s_o_wt[1][warp_id][0];
            int stage = 0;

            for (int kk = lane_id * 8; kk < LDG_O_PROJ_TILE_K; kk += WARP_SIZE * 8) {
                ldg_cp_async_16(smem_tile0 + kk, o_row + kk);
            }
            ldg_cp_async_commit();

            for (int k_base = 0; k_base < Q_SIZE; k_base += LDG_O_PROJ_TILE_K) {
                ldg_cp_async_wait();
                __syncwarp();

                int next_k = k_base + LDG_O_PROJ_TILE_K;
                int next_stage = stage ^ 1;
                if (next_k < Q_SIZE) {
                    __nv_bfloat16* smem_next = (next_stage == 0) ? smem_tile0 : smem_tile1;
                    for (int kk = lane_id * 8; kk < LDG_O_PROJ_TILE_K; kk += WARP_SIZE * 8) {
                        ldg_cp_async_16(smem_next + kk, o_row + next_k + kk);
                    }
                    ldg_cp_async_commit();
                }

                const __nv_bfloat16* smem_curr = (stage == 0) ? smem_tile0 : smem_tile1;
                #pragma unroll 4
                for (int k = lane_id * 8; k < LDG_O_PROJ_TILE_K; k += WARP_SIZE * 8) {
                    uint4 w_u4 = *reinterpret_cast<const uint4*>(smem_curr + k);
                    __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);
                    float4 a1 = *reinterpret_cast<const float4*>(s_attn + k_base + k);
                    float4 a2 = *reinterpret_cast<const float4*>(s_attn + k_base + k + 4);

                    sum += __bfloat162float(w_ptr[0]) * a1.x +
                           __bfloat162float(w_ptr[1]) * a1.y +
                           __bfloat162float(w_ptr[2]) * a1.z +
                           __bfloat162float(w_ptr[3]) * a1.w +
                           __bfloat162float(w_ptr[4]) * a2.x +
                           __bfloat162float(w_ptr[5]) * a2.y +
                           __bfloat162float(w_ptr[6]) * a2.z +
                           __bfloat162float(w_ptr[7]) * a2.w;
                }

                stage = next_stage;
            }
#elif defined(LDG_O_PROJ_USE_UINT8)
            #pragma unroll 2
            for (int k = lane_id * 16; k < Q_SIZE; k += WARP_SIZE * 16) {
                unsigned int w_u8[8];
                ldg_load_weight_u8(w_u8, o_row + k);
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(w_u8);
                float4 a1 = *reinterpret_cast<const float4*>(s_attn + k);
                float4 a2 = *reinterpret_cast<const float4*>(s_attn + k + 4);
                float4 a3 = *reinterpret_cast<const float4*>(s_attn + k + 8);
                float4 a4 = *reinterpret_cast<const float4*>(s_attn + k + 12);

                sum += __bfloat162float(w_ptr[0]) * a1.x +
                       __bfloat162float(w_ptr[1]) * a1.y +
                       __bfloat162float(w_ptr[2]) * a1.z +
                       __bfloat162float(w_ptr[3]) * a1.w +
                       __bfloat162float(w_ptr[4]) * a2.x +
                       __bfloat162float(w_ptr[5]) * a2.y +
                       __bfloat162float(w_ptr[6]) * a2.z +
                       __bfloat162float(w_ptr[7]) * a2.w +
                       __bfloat162float(w_ptr[8]) * a3.x +
                       __bfloat162float(w_ptr[9]) * a3.y +
                       __bfloat162float(w_ptr[10]) * a3.z +
                       __bfloat162float(w_ptr[11]) * a3.w +
                       __bfloat162float(w_ptr[12]) * a4.x +
                       __bfloat162float(w_ptr[13]) * a4.y +
                       __bfloat162float(w_ptr[14]) * a4.z +
                       __bfloat162float(w_ptr[15]) * a4.w;
            }
#elif defined(LDG_USE_UINT8)
            #pragma unroll 2
            for (int k = lane_id * 16; k < Q_SIZE; k += WARP_SIZE * 16) {
                unsigned int w_u8[8];
                ldg_load_weight_u8(w_u8, o_row + k);
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(w_u8);
                float4 a1 = *reinterpret_cast<const float4*>(s_attn + k);
                float4 a2 = *reinterpret_cast<const float4*>(s_attn + k + 4);
                float4 a3 = *reinterpret_cast<const float4*>(s_attn + k + 8);
                float4 a4 = *reinterpret_cast<const float4*>(s_attn + k + 12);

                sum += __bfloat162float(w_ptr[0]) * a1.x +
                       __bfloat162float(w_ptr[1]) * a1.y +
                       __bfloat162float(w_ptr[2]) * a1.z +
                       __bfloat162float(w_ptr[3]) * a1.w +
                       __bfloat162float(w_ptr[4]) * a2.x +
                       __bfloat162float(w_ptr[5]) * a2.y +
                       __bfloat162float(w_ptr[6]) * a2.z +
                       __bfloat162float(w_ptr[7]) * a2.w +
                       __bfloat162float(w_ptr[8]) * a3.x +
                       __bfloat162float(w_ptr[9]) * a3.y +
                       __bfloat162float(w_ptr[10]) * a3.z +
                       __bfloat162float(w_ptr[11]) * a3.w +
                       __bfloat162float(w_ptr[12]) * a4.x +
                       __bfloat162float(w_ptr[13]) * a4.y +
                       __bfloat162float(w_ptr[14]) * a4.z +
                       __bfloat162float(w_ptr[15]) * a4.w;
            }
#elif defined(LDG_USE_UINT4)
            #pragma unroll 4
            for (int k = lane_id * 8; k < Q_SIZE; k += WARP_SIZE * 8) {
                uint4 w_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4*>(o_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);
                float4 a1 = *reinterpret_cast<const float4*>(s_attn + k);
                float4 a2 = *reinterpret_cast<const float4*>(s_attn + k + 4);

                sum += __bfloat162float(w_ptr[0]) * a1.x +
                       __bfloat162float(w_ptr[1]) * a1.y +
                       __bfloat162float(w_ptr[2]) * a1.z +
                       __bfloat162float(w_ptr[3]) * a1.w +
                       __bfloat162float(w_ptr[4]) * a2.x +
                       __bfloat162float(w_ptr[5]) * a2.y +
                       __bfloat162float(w_ptr[6]) * a2.z +
                       __bfloat162float(w_ptr[7]) * a2.w;
            }
#else
            #pragma unroll 8
            for (int k = lane_id * 4; k < Q_SIZE; k += WARP_SIZE * 4) {
                uint2 w_u2 = ldg_load_weight_u2(reinterpret_cast<const uint2*>(o_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum += __bfloat162float(w_ptr[0]) * s_attn[k] +
                       __bfloat162float(w_ptr[1]) * s_attn[k+1] +
                       __bfloat162float(w_ptr[2]) * s_attn[k+2] +
                       __bfloat162float(w_ptr[3]) * s_attn[k+3];
            }
#endif

            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) {
                g_activations[m] = sum + g_residual[m];
            }
        }
    }

    grid.sync();

    // ALL blocks compute post-attention RMSNorm redundantly (eliminates grid.sync)
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

#if defined(LDG_PREFETCH_NEXT_ROWS)
        if (warp_id == 0 && lane_id == 0) {
            int next_base = m_base + LDG_NUM_WARPS * LDG_PREFETCH_DISTANCE;
            for (int w = 0; w < LDG_NUM_WARPS; w++) {
                int m_pf = next_base + w;
                if (m_pf >= int_end) {
                    break;
                }
                const __nv_bfloat16* gate_row_pf = gate_weight + m_pf * HIDDEN_SIZE;
                const __nv_bfloat16* up_row_pf = up_weight + m_pf * HIDDEN_SIZE;
                ldg_prefetch_row(gate_row_pf, HIDDEN_SIZE);
                ldg_prefetch_row(up_row_pf, HIDDEN_SIZE);
            }
        }
#endif

        if (m < int_end) {
            const __nv_bfloat16* gate_row = gate_weight + m * HIDDEN_SIZE;
            const __nv_bfloat16* up_row = up_weight + m * HIDDEN_SIZE;

            float gate_sum = 0.0f, up_sum = 0.0f;
#if defined(LDG_MLP_USE_UINT8)
            #pragma unroll 2
            for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {
                unsigned int g_u8[8];
                unsigned int u_u8[8];
                ldg_load_weight_u8(g_u8, gate_row + k);
                ldg_load_weight_u8(u_u8, up_row + k);
                __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(g_u8);
                __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(u_u8);
                float4 a1 = *reinterpret_cast<const float4*>(s_act + k);
                float4 a2 = *reinterpret_cast<const float4*>(s_act + k + 4);
                float4 a3 = *reinterpret_cast<const float4*>(s_act + k + 8);
                float4 a4 = *reinterpret_cast<const float4*>(s_act + k + 12);

                gate_sum += __bfloat162float(g_ptr[0]) * a1.x +
                            __bfloat162float(g_ptr[1]) * a1.y +
                            __bfloat162float(g_ptr[2]) * a1.z +
                            __bfloat162float(g_ptr[3]) * a1.w +
                            __bfloat162float(g_ptr[4]) * a2.x +
                            __bfloat162float(g_ptr[5]) * a2.y +
                            __bfloat162float(g_ptr[6]) * a2.z +
                            __bfloat162float(g_ptr[7]) * a2.w +
                            __bfloat162float(g_ptr[8]) * a3.x +
                            __bfloat162float(g_ptr[9]) * a3.y +
                            __bfloat162float(g_ptr[10]) * a3.z +
                            __bfloat162float(g_ptr[11]) * a3.w +
                            __bfloat162float(g_ptr[12]) * a4.x +
                            __bfloat162float(g_ptr[13]) * a4.y +
                            __bfloat162float(g_ptr[14]) * a4.z +
                            __bfloat162float(g_ptr[15]) * a4.w;

                up_sum += __bfloat162float(u_ptr[0]) * a1.x +
                          __bfloat162float(u_ptr[1]) * a1.y +
                          __bfloat162float(u_ptr[2]) * a1.z +
                          __bfloat162float(u_ptr[3]) * a1.w +
                          __bfloat162float(u_ptr[4]) * a2.x +
                          __bfloat162float(u_ptr[5]) * a2.y +
                          __bfloat162float(u_ptr[6]) * a2.z +
                          __bfloat162float(u_ptr[7]) * a2.w +
                          __bfloat162float(u_ptr[8]) * a3.x +
                          __bfloat162float(u_ptr[9]) * a3.y +
                          __bfloat162float(u_ptr[10]) * a3.z +
                          __bfloat162float(u_ptr[11]) * a3.w +
                          __bfloat162float(u_ptr[12]) * a4.x +
                          __bfloat162float(u_ptr[13]) * a4.y +
                          __bfloat162float(u_ptr[14]) * a4.z +
                          __bfloat162float(u_ptr[15]) * a4.w;
            }
#elif defined(LDG_USE_UINT8)
            #pragma unroll 2
            for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {
                unsigned int g_u8[8];
                unsigned int u_u8[8];
                ldg_load_weight_u8(g_u8, gate_row + k);
                ldg_load_weight_u8(u_u8, up_row + k);
                __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(g_u8);
                __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(u_u8);
                float4 a1 = *reinterpret_cast<const float4*>(s_act + k);
                float4 a2 = *reinterpret_cast<const float4*>(s_act + k + 4);
                float4 a3 = *reinterpret_cast<const float4*>(s_act + k + 8);
                float4 a4 = *reinterpret_cast<const float4*>(s_act + k + 12);

                gate_sum += __bfloat162float(g_ptr[0]) * a1.x +
                            __bfloat162float(g_ptr[1]) * a1.y +
                            __bfloat162float(g_ptr[2]) * a1.z +
                            __bfloat162float(g_ptr[3]) * a1.w +
                            __bfloat162float(g_ptr[4]) * a2.x +
                            __bfloat162float(g_ptr[5]) * a2.y +
                            __bfloat162float(g_ptr[6]) * a2.z +
                            __bfloat162float(g_ptr[7]) * a2.w +
                            __bfloat162float(g_ptr[8]) * a3.x +
                            __bfloat162float(g_ptr[9]) * a3.y +
                            __bfloat162float(g_ptr[10]) * a3.z +
                            __bfloat162float(g_ptr[11]) * a3.w +
                            __bfloat162float(g_ptr[12]) * a4.x +
                            __bfloat162float(g_ptr[13]) * a4.y +
                            __bfloat162float(g_ptr[14]) * a4.z +
                            __bfloat162float(g_ptr[15]) * a4.w;

                up_sum += __bfloat162float(u_ptr[0]) * a1.x +
                          __bfloat162float(u_ptr[1]) * a1.y +
                          __bfloat162float(u_ptr[2]) * a1.z +
                          __bfloat162float(u_ptr[3]) * a1.w +
                          __bfloat162float(u_ptr[4]) * a2.x +
                          __bfloat162float(u_ptr[5]) * a2.y +
                          __bfloat162float(u_ptr[6]) * a2.z +
                          __bfloat162float(u_ptr[7]) * a2.w +
                          __bfloat162float(u_ptr[8]) * a3.x +
                          __bfloat162float(u_ptr[9]) * a3.y +
                          __bfloat162float(u_ptr[10]) * a3.z +
                          __bfloat162float(u_ptr[11]) * a3.w +
                          __bfloat162float(u_ptr[12]) * a4.x +
                          __bfloat162float(u_ptr[13]) * a4.y +
                          __bfloat162float(u_ptr[14]) * a4.z +
                          __bfloat162float(u_ptr[15]) * a4.w;
            }
#elif defined(LDG_USE_UINT4)
            #pragma unroll 4
            for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
                uint4 g_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4*>(gate_row + k));
                uint4 u_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4*>(up_row + k));
                __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(&g_u4);
                __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(&u_u4);
                float4 a1 = *reinterpret_cast<const float4*>(s_act + k);
                float4 a2 = *reinterpret_cast<const float4*>(s_act + k + 4);

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
#else
            #pragma unroll 8
            for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
                uint2 g_u2 = ldg_load_weight_u2(reinterpret_cast<const uint2*>(gate_row + k));
                uint2 u_u2 = ldg_load_weight_u2(reinterpret_cast<const uint2*>(up_row + k));
                __nv_bfloat16* g_ptr = reinterpret_cast<__nv_bfloat16*>(&g_u2);
                __nv_bfloat16* u_ptr = reinterpret_cast<__nv_bfloat16*>(&u_u2);

                gate_sum += __bfloat162float(g_ptr[0]) * s_act[k] +
                            __bfloat162float(g_ptr[1]) * s_act[k+1] +
                            __bfloat162float(g_ptr[2]) * s_act[k+2] +
                            __bfloat162float(g_ptr[3]) * s_act[k+3];

                up_sum += __bfloat162float(u_ptr[0]) * s_act[k] +
                          __bfloat162float(u_ptr[1]) * s_act[k+1] +
                          __bfloat162float(u_ptr[2]) * s_act[k+2] +
                          __bfloat162float(u_ptr[3]) * s_act[k+3];
            }
#endif

            gate_sum = ldg_warp_reduce_sum(gate_sum);
            up_sum = ldg_warp_reduce_sum(up_sum);

            if (lane_id == 0) {
                g_mlp_intermediate[m] = ldg_silu(gate_sum) * up_sum;
            }
        }
    }

    grid.sync();

#if defined(LDG_MLP_SMEM)
    for (int i = threadIdx.x; i < INTERMEDIATE_SIZE; i += LDG_BLOCK_SIZE) {
        s_mlp[i] = g_mlp_intermediate[i];
    }
    __syncthreads();
#endif

    // Down projection + residual
    const float* mlp_in = g_mlp_intermediate;
#if defined(LDG_MLP_SMEM)
    mlp_in = s_mlp;
#endif
    for (int m_base = hid_start; m_base < hid_end; m_base += LDG_NUM_WARPS) {
        int m = m_base + warp_id;

#if defined(LDG_PREFETCH_NEXT_ROWS)
        if (warp_id == 0 && lane_id == 0) {
            int next_base = m_base + LDG_NUM_WARPS * LDG_PREFETCH_DISTANCE;
            for (int w = 0; w < LDG_NUM_WARPS; w++) {
                int m_pf = next_base + w;
                if (m_pf >= hid_end) {
                    break;
                }
                const __nv_bfloat16* down_row_pf = down_weight + m_pf * INTERMEDIATE_SIZE;
                ldg_prefetch_row(down_row_pf, INTERMEDIATE_SIZE);
            }
        }
#endif

        if (m < hid_end) {
            const __nv_bfloat16* down_row = down_weight + m * INTERMEDIATE_SIZE;

            float sum = 0.0f;
#if defined(LDG_MLP_USE_UINT8)
            #pragma unroll 2
            for (int k = lane_id * 16; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 16) {
                unsigned int d_u8[8];
                ldg_load_weight_u8(d_u8, down_row + k);
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(d_u8);
                float4 a1 = *reinterpret_cast<const float4*>(mlp_in + k);
                float4 a2 = *reinterpret_cast<const float4*>(mlp_in + k + 4);
                float4 a3 = *reinterpret_cast<const float4*>(mlp_in + k + 8);
                float4 a4 = *reinterpret_cast<const float4*>(mlp_in + k + 12);

                sum += __bfloat162float(d_ptr[0]) * a1.x +
                       __bfloat162float(d_ptr[1]) * a1.y +
                       __bfloat162float(d_ptr[2]) * a1.z +
                       __bfloat162float(d_ptr[3]) * a1.w +
                       __bfloat162float(d_ptr[4]) * a2.x +
                       __bfloat162float(d_ptr[5]) * a2.y +
                       __bfloat162float(d_ptr[6]) * a2.z +
                       __bfloat162float(d_ptr[7]) * a2.w +
                       __bfloat162float(d_ptr[8]) * a3.x +
                       __bfloat162float(d_ptr[9]) * a3.y +
                       __bfloat162float(d_ptr[10]) * a3.z +
                       __bfloat162float(d_ptr[11]) * a3.w +
                       __bfloat162float(d_ptr[12]) * a4.x +
                       __bfloat162float(d_ptr[13]) * a4.y +
                       __bfloat162float(d_ptr[14]) * a4.z +
                       __bfloat162float(d_ptr[15]) * a4.w;
            }
#elif defined(LDG_USE_UINT8)
            #pragma unroll 2
            for (int k = lane_id * 16; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 16) {
                unsigned int d_u8[8];
                ldg_load_weight_u8(d_u8, down_row + k);
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(d_u8);
                float4 a1 = *reinterpret_cast<const float4*>(mlp_in + k);
                float4 a2 = *reinterpret_cast<const float4*>(mlp_in + k + 4);
                float4 a3 = *reinterpret_cast<const float4*>(g_mlp_intermediate + k + 8);
                float4 a4 = *reinterpret_cast<const float4*>(g_mlp_intermediate + k + 12);

                sum += __bfloat162float(d_ptr[0]) * a1.x +
                       __bfloat162float(d_ptr[1]) * a1.y +
                       __bfloat162float(d_ptr[2]) * a1.z +
                       __bfloat162float(d_ptr[3]) * a1.w +
                       __bfloat162float(d_ptr[4]) * a2.x +
                       __bfloat162float(d_ptr[5]) * a2.y +
                       __bfloat162float(d_ptr[6]) * a2.z +
                       __bfloat162float(d_ptr[7]) * a2.w +
                       __bfloat162float(d_ptr[8]) * a3.x +
                       __bfloat162float(d_ptr[9]) * a3.y +
                       __bfloat162float(d_ptr[10]) * a3.z +
                       __bfloat162float(d_ptr[11]) * a3.w +
                       __bfloat162float(d_ptr[12]) * a4.x +
                       __bfloat162float(d_ptr[13]) * a4.y +
                       __bfloat162float(d_ptr[14]) * a4.z +
                       __bfloat162float(d_ptr[15]) * a4.w;
            }
#elif defined(LDG_USE_UINT4)
            #pragma unroll 4
            for (int k = lane_id * 8; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 8) {
                uint4 d_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4*>(down_row + k));
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u4);
                float4 a1 = *reinterpret_cast<const float4*>(g_mlp_intermediate + k);
                float4 a2 = *reinterpret_cast<const float4*>(g_mlp_intermediate + k + 4);

                sum += __bfloat162float(d_ptr[0]) * a1.x +
                       __bfloat162float(d_ptr[1]) * a1.y +
                       __bfloat162float(d_ptr[2]) * a1.z +
                       __bfloat162float(d_ptr[3]) * a1.w +
                       __bfloat162float(d_ptr[4]) * a2.x +
                       __bfloat162float(d_ptr[5]) * a2.y +
                       __bfloat162float(d_ptr[6]) * a2.z +
                       __bfloat162float(d_ptr[7]) * a2.w;
            }
#else
            #pragma unroll 8
            for (int k = lane_id * 4; k < INTERMEDIATE_SIZE; k += WARP_SIZE * 4) {
                uint2 d_u2 = ldg_load_weight_u2(reinterpret_cast<const uint2*>(down_row + k));
                __nv_bfloat16* d_ptr = reinterpret_cast<__nv_bfloat16*>(&d_u2);

                sum += __bfloat162float(d_ptr[0]) * mlp_in[k] +
                       __bfloat162float(d_ptr[1]) * mlp_in[k+1] +
                       __bfloat162float(d_ptr[2]) * mlp_in[k+2] +
                       __bfloat162float(d_ptr[3]) * mlp_in[k+3];
            }
#endif

            sum = ldg_warp_reduce_sum(sum);
            if (lane_id == 0) {
                hidden_out[m] = __float2bfloat16(sum + g_residual[m]);
            }
        }
    }

    grid.sync();
}

// =============================================================================
// Main Kernel
// =============================================================================

__global__ void __launch_bounds__(LDG_BLOCK_SIZE, 1)
ldg_decode_kernel(
    int input_token_id,
    const __nv_bfloat16* __restrict__ embed_weight,
    const LDGLayerWeights* __restrict__ layer_weights,
    const __nv_bfloat16* __restrict__ final_norm_weight,
    const __nv_bfloat16* __restrict__ cos_table,
    const __nv_bfloat16* __restrict__ sin_table,
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ hidden_buffer,
    float* __restrict__ g_activations,
    float* __restrict__ g_residual,
    float* __restrict__ g_q,
    float* __restrict__ g_k,
    float* __restrict__ g_v,
    float* __restrict__ g_attn_out,
    float* __restrict__ g_mlp_intermediate,
    float* __restrict__ g_normalized,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale
) {
    cg::grid_group grid = cg::this_grid();
    int block_id = blockIdx.x;
    int num_blocks = gridDim.x;
    bool is_prof = (block_id == 0 && threadIdx.x == 0);
#if defined(LDG_PHASE_PROFILE)
    if (is_prof) {
        #pragma unroll
        for (int i = 0; i < LDG_PHASE_COUNT; i++) {
            g_phase_cycles[i] = 0;
        }
    }
    grid.sync();
    unsigned long long t0 = 0;
    unsigned long long t1 = 0;
#endif

    // Embedding lookup with __ldg
    const __nv_bfloat16* embed_row = embed_weight + input_token_id * HIDDEN_SIZE;
#if defined(LDG_PHASE_PROFILE)
    if (is_prof) {
        t0 = clock64();
    }
#endif
    for (int i = block_id * LDG_BLOCK_SIZE + threadIdx.x; i < HIDDEN_SIZE; i += num_blocks * LDG_BLOCK_SIZE) {
        hidden_buffer[i] = __ldg(embed_row + i);
    }
    grid.sync();
#if defined(LDG_PHASE_PROFILE)
    if (is_prof) {
        t1 = clock64();
        g_phase_cycles[0] += (t1 - t0);
    }
#endif

    int kv_cache_layer_stride = NUM_KV_HEADS * max_seq_len * HEAD_DIM;

    for (int layer = 0; layer < num_layers; layer++) {
        const LDGLayerWeights& w = layer_weights[layer];
        __nv_bfloat16* layer_k_cache = k_cache + layer * kv_cache_layer_stride;
        __nv_bfloat16* layer_v_cache = v_cache + layer * kv_cache_layer_stride;

#if defined(LDG_PHASE_PROFILE)
        if (is_prof) {
            t0 = clock64();
        }
#endif
        ldg_matvec_qkv(
            grid, hidden_buffer, w.input_layernorm_weight,
            w.q_proj_weight, w.k_proj_weight, w.v_proj_weight,
            g_activations, g_residual, g_q, g_k, g_v
        );
#if defined(LDG_PHASE_PROFILE)
        if (is_prof) {
            t1 = clock64();
            g_phase_cycles[1] += (t1 - t0);
        }
#endif

#if defined(LDG_PHASE_PROFILE)
        if (is_prof) {
            t0 = clock64();
        }
#endif
        ldg_qk_norm_rope_cache(
            grid, g_q, g_k, g_v,
            w.q_norm_weight, w.k_norm_weight,
            cos_table, sin_table,
            layer_k_cache, layer_v_cache,
            position, max_seq_len
        );
#if defined(LDG_PHASE_PROFILE)
        if (is_prof) {
            t1 = clock64();
            g_phase_cycles[2] += (t1 - t0);
        }
#endif

#if defined(LDG_PHASE_PROFILE)
        if (is_prof) {
            t0 = clock64();
        }
#endif
        ldg_attention(
            grid, g_q, layer_k_cache, layer_v_cache, g_attn_out,
            cache_len, max_seq_len, attn_scale,
            w.o_proj_weight, w.gate_proj_weight, w.up_proj_weight,
            w.down_proj_weight
        );
#if defined(LDG_PHASE_PROFILE)
        if (is_prof) {
            t1 = clock64();
            g_phase_cycles[3] += (t1 - t0);
        }
#endif

#if defined(LDG_PHASE_PROFILE)
        if (is_prof) {
            t0 = clock64();
        }
#endif
        ldg_o_proj_postnorm_mlp(
            grid, w.o_proj_weight, w.post_attn_layernorm_weight,
            w.gate_proj_weight, w.up_proj_weight, w.down_proj_weight,
            g_attn_out, g_residual, g_activations, g_mlp_intermediate,
            hidden_buffer
        );
#if defined(LDG_PHASE_PROFILE)
        if (is_prof) {
            t1 = clock64();
            g_phase_cycles[4] += (t1 - t0);
        }
#endif
    }

    // Final RMSNorm
#if defined(LDG_PHASE_PROFILE)
    if (is_prof) {
        t0 = clock64();
    }
#endif
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
            float wt = __bfloat162float(__ldg(final_norm_weight + i));
            g_normalized[i] = g_activations[i] * rstd * wt;
        }
    }
#if defined(LDG_PHASE_PROFILE)
    if (is_prof) {
        t1 = clock64();
        g_phase_cycles[5] += (t1 - t0);
    }
#endif
}

// =============================================================================
// LM Head (same structure)
// =============================================================================

// Kernel to compute full logits (for KL divergence measurement)
__global__ void ldg_lm_head_logits(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ logits
) {
    __shared__ __align__(16) float s_hidden[HIDDEN_SIZE];
#if defined(LDG_LM_ASYNC_SMEM)
    __shared__ __align__(16) __nv_bfloat16 s_lm_wt[2][LDG_LM_BLOCK_SIZE / WARP_SIZE][LDG_LM_ROWS_PER_WARP][LDG_LM_TILE_K];
#endif

    for (int i = threadIdx.x; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE) {
        s_hidden[i] = hidden[i];
    }
    __syncthreads();

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

    int warp_stride = LDG_LM_BLOCK_SIZE / WARP_SIZE;
    int base = row_start + warp_id * LDG_LM_ROWS_PER_WARP;

    for (int m_base = base; m_base < row_end; m_base += warp_stride * LDG_LM_ROWS_PER_WARP) {
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

#if defined(LDG_LM_ASYNC_SMEM)
        __nv_bfloat16* smem_tile0 = &s_lm_wt[0][warp_id][0][0];
        __nv_bfloat16* smem_tile1 = &s_lm_wt[1][warp_id][0][0];
        int stage = 0;

        #pragma unroll
        for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
            if (!valid[r]) {
                continue;
            }
            const __nv_bfloat16* w_row = weight + rows[r] * HIDDEN_SIZE;
            for (int kk = lane_id * 8; kk < LDG_LM_TILE_K; kk += WARP_SIZE * 8) {
                ldg_cp_async_16(smem_tile0 + r * LDG_LM_TILE_K + kk, w_row + kk);
            }
        }
        ldg_cp_async_commit();

        for (int k_base = 0; k_base < HIDDEN_SIZE; k_base += LDG_LM_TILE_K) {
            ldg_cp_async_wait();
            __syncwarp();

            int next_k = k_base + LDG_LM_TILE_K;
            int next_stage = stage ^ 1;
            if (next_k < HIDDEN_SIZE) {
                __nv_bfloat16* smem_next = (next_stage == 0) ? smem_tile0 : smem_tile1;
                #pragma unroll
                for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                    if (!valid[r]) {
                        continue;
                    }
                    const __nv_bfloat16* w_row = weight + rows[r] * HIDDEN_SIZE;
                    for (int kk = lane_id * 8; kk < LDG_LM_TILE_K; kk += WARP_SIZE * 8) {
                        ldg_cp_async_16(smem_next + r * LDG_LM_TILE_K + kk, w_row + next_k + kk);
                    }
                }
                ldg_cp_async_commit();
            }

            const __nv_bfloat16* smem_curr = (stage == 0) ? smem_tile0 : smem_tile1;
            #pragma unroll
            for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                if (!valid[r]) {
                    continue;
                }
                const __nv_bfloat16* w_tile = smem_curr + r * LDG_LM_TILE_K;
                #pragma unroll 4
                for (int k = lane_id * 8; k < LDG_LM_TILE_K; k += WARP_SIZE * 8) {
                    uint4 w_u4 = *reinterpret_cast<const uint4*>(w_tile + k);
                    __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);
                    float4 a1 = *reinterpret_cast<const float4*>(s_hidden + k_base + k);
                    float4 a2 = *reinterpret_cast<const float4*>(s_hidden + k_base + k + 4);

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

            stage = next_stage;
        }
#elif defined(LDG_USE_UINT8)
        #pragma unroll 2
        for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {
            float4 a1 = *reinterpret_cast<const float4*>(s_hidden + k);
            float4 a2 = *reinterpret_cast<const float4*>(s_hidden + k + 4);
            float4 a3 = *reinterpret_cast<const float4*>(s_hidden + k + 8);
            float4 a4 = *reinterpret_cast<const float4*>(s_hidden + k + 12);

            #pragma unroll
            for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                if (!valid[r]) {
                    continue;
                }
                const __nv_bfloat16* w_row = weight + rows[r] * HIDDEN_SIZE;
                unsigned int w_u8[8];
                ldg_load_weight_u8(w_u8, w_row + k);
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(w_u8);

                sum[r] += __bfloat162float(w_ptr[0]) * a1.x +
                          __bfloat162float(w_ptr[1]) * a1.y +
                          __bfloat162float(w_ptr[2]) * a1.z +
                          __bfloat162float(w_ptr[3]) * a1.w +
                          __bfloat162float(w_ptr[4]) * a2.x +
                          __bfloat162float(w_ptr[5]) * a2.y +
                          __bfloat162float(w_ptr[6]) * a2.z +
                          __bfloat162float(w_ptr[7]) * a2.w +
                          __bfloat162float(w_ptr[8]) * a3.x +
                          __bfloat162float(w_ptr[9]) * a3.y +
                          __bfloat162float(w_ptr[10]) * a3.z +
                          __bfloat162float(w_ptr[11]) * a3.w +
                          __bfloat162float(w_ptr[12]) * a4.x +
                          __bfloat162float(w_ptr[13]) * a4.y +
                          __bfloat162float(w_ptr[14]) * a4.z +
                          __bfloat162float(w_ptr[15]) * a4.w;
            }
        }
#elif defined(LDG_USE_UINT4)
        #pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            float4 a1 = *reinterpret_cast<const float4*>(s_hidden + k);
            float4 a2 = *reinterpret_cast<const float4*>(s_hidden + k + 4);

            #pragma unroll
            for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                if (!valid[r]) {
                    continue;
                }
                const __nv_bfloat16* w_row = weight + rows[r] * HIDDEN_SIZE;
                uint4 w_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4*>(w_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

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
#else
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            float a0 = s_hidden[k];
            float a1 = s_hidden[k + 1];
            float a2 = s_hidden[k + 2];
            float a3 = s_hidden[k + 3];

            #pragma unroll
            for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                if (!valid[r]) {
                    continue;
                }
                const __nv_bfloat16* w_row = weight + rows[r] * HIDDEN_SIZE;
                uint2 w_u2 = ldg_load_weight_u2(reinterpret_cast<const uint2*>(w_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum[r] += __bfloat162float(w_ptr[0]) * a0 +
                          __bfloat162float(w_ptr[1]) * a1 +
                          __bfloat162float(w_ptr[2]) * a2 +
                          __bfloat162float(w_ptr[3]) * a3;
            }
        }
#endif

        #pragma unroll
        for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
            if (!valid[r]) {
                continue;
            }
            float reduced = ldg_warp_reduce_sum(sum[r]);
            if (lane_id == 0) {
                logits[rows[r]] = reduced;
            }
        }
    }
}

#if defined(LDG_LM_USE_WMMA)
__global__ void ldg_lm_head_phase1_wmma(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
    __shared__ __align__(16) __nv_bfloat16 s_hidden_bf16[HIDDEN_SIZE];
    __shared__ __align__(16) __nv_bfloat16 sB[16 * 8];
    __shared__ float sC[(LDG_LM_BLOCK_SIZE / WARP_SIZE) * 32 * 8];
    __shared__ float warp_max[LDG_LM_BLOCK_SIZE / WARP_SIZE];
    __shared__ int warp_idx[LDG_LM_BLOCK_SIZE / WARP_SIZE];

    int tid = threadIdx.x;
    for (int i = tid; i < HIDDEN_SIZE; i += LDG_LM_BLOCK_SIZE) {
        s_hidden_bf16[i] = __float2bfloat16(hidden[i]);
    }
    __syncthreads();

    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warps_per_block = LDG_LM_BLOCK_SIZE / WARP_SIZE;

    int rows_per_block = (LDG_VOCAB_SIZE + gridDim.x - 1) / gridDim.x;
    int row_start = blockIdx.x * rows_per_block;
    int row_end = min(row_start + rows_per_block, LDG_VOCAB_SIZE);

    int tile_row = row_start + warp_id * 32;

    wmma::fragment<wmma::accumulator, 32, 8, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k0 = 0; k0 < HIDDEN_SIZE; k0 += 16) {
        if (tid < 16 * 8) {
            int row = tid / 8;
            int col = tid % 8;
            sB[row * 8 + col] = s_hidden_bf16[k0 + row];
        }
        __syncthreads();

        if (warp_id < warps_per_block) {
            const __nv_bfloat16* w_ptr = weight + tile_row * HIDDEN_SIZE + k0;
            wmma::fragment<wmma::matrix_a, 32, 8, 16, __nv_bfloat16, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 32, 8, 16, __nv_bfloat16, wmma::row_major> b_frag;

            wmma::load_matrix_sync(a_frag, w_ptr, HIDDEN_SIZE);
            wmma::load_matrix_sync(b_frag, sB, 8);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    if (warp_id < warps_per_block) {
        float* c_ptr = sC + warp_id * 32 * 8;
        wmma::store_matrix_sync(c_ptr, c_frag, 8, wmma::mem_row_major);
    }
    __syncthreads();

    float local_max = -INFINITY;
    int local_idx = -1;
    if (warp_id < warps_per_block) {
        int row = tile_row + lane_id;
        if (row < row_end) {
            float val = sC[warp_id * 32 * 8 + lane_id * 8 + 0];
            local_max = val;
            local_idx = row;
        }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other_val = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_val > local_max) {
            local_max = other_val;
            local_idx = other_idx;
        }
    }

    if (lane_id == 0) {
        warp_max[warp_id] = local_max;
        warp_idx[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = (lane_id < warps_per_block) ? warp_max[lane_id] : -INFINITY;
        int max_idx = (lane_id < warps_per_block) ? warp_idx[lane_id] : -1;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
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
#endif

__global__ void ldg_lm_head_phase1(
    const float* __restrict__ hidden,
    const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ block_max_vals,
    int* __restrict__ block_max_idxs
) {
    __shared__ __align__(16) float s_hidden[HIDDEN_SIZE];
#if defined(LDG_LM_ASYNC_SMEM)
    __shared__ __align__(16) __nv_bfloat16 s_lm_wt[2][LDG_LM_BLOCK_SIZE / WARP_SIZE][LDG_LM_ROWS_PER_WARP][LDG_LM_TILE_K];
#endif

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

    for (int m_base = base; m_base < row_end; m_base += warp_stride * LDG_LM_ROWS_PER_WARP) {
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

#if defined(LDG_LM_ASYNC_SMEM)
        __nv_bfloat16* smem_tile0 = &s_lm_wt[0][warp_id][0][0];
        __nv_bfloat16* smem_tile1 = &s_lm_wt[1][warp_id][0][0];
        int stage = 0;

        #pragma unroll
        for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
            if (!valid[r]) {
                continue;
            }
            const __nv_bfloat16* w_row = weight + rows[r] * HIDDEN_SIZE;
            for (int kk = lane_id * 8; kk < LDG_LM_TILE_K; kk += WARP_SIZE * 8) {
                ldg_cp_async_16(smem_tile0 + r * LDG_LM_TILE_K + kk, w_row + kk);
            }
        }
        ldg_cp_async_commit();

        for (int k_base = 0; k_base < HIDDEN_SIZE; k_base += LDG_LM_TILE_K) {
            ldg_cp_async_wait();
            __syncwarp();

            int next_k = k_base + LDG_LM_TILE_K;
            int next_stage = stage ^ 1;
            if (next_k < HIDDEN_SIZE) {
                __nv_bfloat16* smem_next = (next_stage == 0) ? smem_tile0 : smem_tile1;
                #pragma unroll
                for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                    if (!valid[r]) {
                        continue;
                    }
                    const __nv_bfloat16* w_row = weight + rows[r] * HIDDEN_SIZE;
                    for (int kk = lane_id * 8; kk < LDG_LM_TILE_K; kk += WARP_SIZE * 8) {
                        ldg_cp_async_16(smem_next + r * LDG_LM_TILE_K + kk, w_row + next_k + kk);
                    }
                }
                ldg_cp_async_commit();
            }

            const __nv_bfloat16* smem_curr = (stage == 0) ? smem_tile0 : smem_tile1;
            #pragma unroll
            for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                if (!valid[r]) {
                    continue;
                }
                const __nv_bfloat16* w_tile = smem_curr + r * LDG_LM_TILE_K;
                #pragma unroll 4
                for (int k = lane_id * 8; k < LDG_LM_TILE_K; k += WARP_SIZE * 8) {
                    uint4 w_u4 = *reinterpret_cast<const uint4*>(w_tile + k);
                    __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);
                    float4 a1 = *reinterpret_cast<const float4*>(s_hidden + k_base + k);
                    float4 a2 = *reinterpret_cast<const float4*>(s_hidden + k_base + k + 4);

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

            stage = next_stage;
        }
#elif defined(LDG_USE_UINT8)
        #pragma unroll 2
        for (int k = lane_id * 16; k < HIDDEN_SIZE; k += WARP_SIZE * 16) {
            float4 a1 = *reinterpret_cast<const float4*>(s_hidden + k);
            float4 a2 = *reinterpret_cast<const float4*>(s_hidden + k + 4);
            float4 a3 = *reinterpret_cast<const float4*>(s_hidden + k + 8);
            float4 a4 = *reinterpret_cast<const float4*>(s_hidden + k + 12);

            #pragma unroll
            for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                if (!valid[r]) {
                    continue;
                }
                const __nv_bfloat16* w_row = weight + rows[r] * HIDDEN_SIZE;
                unsigned int w_u8[8];
                ldg_load_weight_u8(w_u8, w_row + k);
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(w_u8);

                sum[r] += __bfloat162float(w_ptr[0]) * a1.x +
                          __bfloat162float(w_ptr[1]) * a1.y +
                          __bfloat162float(w_ptr[2]) * a1.z +
                          __bfloat162float(w_ptr[3]) * a1.w +
                          __bfloat162float(w_ptr[4]) * a2.x +
                          __bfloat162float(w_ptr[5]) * a2.y +
                          __bfloat162float(w_ptr[6]) * a2.z +
                          __bfloat162float(w_ptr[7]) * a2.w +
                          __bfloat162float(w_ptr[8]) * a3.x +
                          __bfloat162float(w_ptr[9]) * a3.y +
                          __bfloat162float(w_ptr[10]) * a3.z +
                          __bfloat162float(w_ptr[11]) * a3.w +
                          __bfloat162float(w_ptr[12]) * a4.x +
                          __bfloat162float(w_ptr[13]) * a4.y +
                          __bfloat162float(w_ptr[14]) * a4.z +
                          __bfloat162float(w_ptr[15]) * a4.w;
            }
        }
#elif defined(LDG_USE_UINT4)
        #pragma unroll 4
        for (int k = lane_id * 8; k < HIDDEN_SIZE; k += WARP_SIZE * 8) {
            float4 a1 = *reinterpret_cast<const float4*>(s_hidden + k);
            float4 a2 = *reinterpret_cast<const float4*>(s_hidden + k + 4);

            #pragma unroll
            for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                if (!valid[r]) {
                    continue;
                }
                const __nv_bfloat16* w_row = weight + rows[r] * HIDDEN_SIZE;
                uint4 w_u4 = ldg_load_weight_u4(reinterpret_cast<const uint4*>(w_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u4);

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
#else
        #pragma unroll 8
        for (int k = lane_id * 4; k < HIDDEN_SIZE; k += WARP_SIZE * 4) {
            float a0 = s_hidden[k];
            float a1 = s_hidden[k + 1];
            float a2 = s_hidden[k + 2];
            float a3 = s_hidden[k + 3];

            #pragma unroll
            for (int r = 0; r < LDG_LM_ROWS_PER_WARP; r++) {
                if (!valid[r]) {
                    continue;
                }
                const __nv_bfloat16* w_row = weight + rows[r] * HIDDEN_SIZE;
                uint2 w_u2 = ldg_load_weight_u2(reinterpret_cast<const uint2*>(w_row + k));
                __nv_bfloat16* w_ptr = reinterpret_cast<__nv_bfloat16*>(&w_u2);

                sum[r] += __bfloat162float(w_ptr[0]) * a0 +
                          __bfloat162float(w_ptr[1]) * a1 +
                          __bfloat162float(w_ptr[2]) * a2 +
                          __bfloat162float(w_ptr[3]) * a3;
            }
        }
#endif

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
        float max_val = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_max[lane_id] : -INFINITY;
        int max_idx = (lane_id < LDG_LM_BLOCK_SIZE / WARP_SIZE) ? warp_idx[lane_id] : -1;

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

__global__ void ldg_lm_head_phase2(
    const float* __restrict__ block_max_vals,
    const int* __restrict__ block_max_idxs,
    int* __restrict__ output_token,
    int num_blocks
) {
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
// Launch function
// =============================================================================

static inline void ldg_configure_kernel_attributes() {
#if LDG_SET_L1_CARVEOUT
    static bool configured = false;
    if (configured) {
        return;
    }
    configured = true;
    cudaFuncSetAttribute(ldg_decode_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(ldg_lm_head_phase1, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(ldg_lm_head_phase2, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(ldg_lm_head_logits, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
#if defined(LDG_LM_USE_WMMA)
    cudaFuncSetAttribute(ldg_lm_head_phase1_wmma, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
#endif
#endif
}

extern "C" void launch_ldg_decode(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
) {
    ldg_configure_kernel_attributes();
    void* kernel_args[] = {
        (void*)&input_token_id,
        (void*)&embed_weight,
        (void*)&layer_weights,
        (void*)&final_norm_weight,
        (void*)&cos_table,
        (void*)&sin_table,
        (void*)&k_cache,
        (void*)&v_cache,
        (void*)&hidden_buffer,
        (void*)&g_activations,
        (void*)&g_residual,
        (void*)&g_q,
        (void*)&g_k,
        (void*)&g_v,
        (void*)&g_attn_out,
        (void*)&g_mlp_intermediate,
        (void*)&g_normalized,
        (void*)&num_layers,
        (void*)&position,
        (void*)&cache_len,
        (void*)&max_seq_len,
        (void*)&attn_scale
    };

    cudaLaunchCooperativeKernel(
        (void*)ldg_decode_kernel,
        dim3(LDG_NUM_BLOCKS),
        dim3(LDG_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );

    #if defined(LDG_LM_USE_WMMA)
    ldg_lm_head_phase1_wmma<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );
    #else
    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );
    #endif

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS
    );
}

// Launch function with per-kernel timings (ms) written to timings_out[0..2].
extern "C" void launch_ldg_decode_profile(
    int input_token_id,
    int* output_token_id,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    float* timings_out,
    float* phase_timings_out,
    cudaStream_t stream
) {
    ldg_configure_kernel_attributes();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    void* kernel_args[] = {
        (void*)&input_token_id,
        (void*)&embed_weight,
        (void*)&layer_weights,
        (void*)&final_norm_weight,
        (void*)&cos_table,
        (void*)&sin_table,
        (void*)&k_cache,
        (void*)&v_cache,
        (void*)&hidden_buffer,
        (void*)&g_activations,
        (void*)&g_residual,
        (void*)&g_q,
        (void*)&g_k,
        (void*)&g_v,
        (void*)&g_attn_out,
        (void*)&g_mlp_intermediate,
        (void*)&g_normalized,
        (void*)&num_layers,
        (void*)&position,
        (void*)&cache_len,
        (void*)&max_seq_len,
        (void*)&attn_scale
    };

    cudaEventRecord(start, stream);
    cudaLaunchCooperativeKernel(
        (void*)ldg_decode_kernel,
        dim3(LDG_NUM_BLOCKS),
        dim3(LDG_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    if (timings_out) {
        cudaEventElapsedTime(&timings_out[0], start, stop);
    }
#if defined(LDG_PHASE_PROFILE)
    if (phase_timings_out) {
        static int clock_rate_khz = -1;
        if (clock_rate_khz < 0) {
            cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, 0);
        }
        for (int i = 0; i < LDG_PHASE_COUNT; i++) {
            phase_timings_out[i] = static_cast<float>(g_phase_cycles[i]) / static_cast<float>(clock_rate_khz);
        }
    }
#else
    if (phase_timings_out) {
        for (int i = 0; i < LDG_PHASE_COUNT; i++) {
            phase_timings_out[i] = 0.0f;
        }
    }
#endif

    cudaEventRecord(start, stream);
    #if defined(LDG_LM_USE_WMMA)
    ldg_lm_head_phase1_wmma<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );
    #else
    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );
    #endif
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    if (timings_out) {
        cudaEventElapsedTime(&timings_out[1], start, stop);
    }

    cudaEventRecord(start, stream);
    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS
    );
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    if (timings_out) {
        cudaEventElapsedTime(&timings_out[2], start, stop);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Launch function that also outputs full logits (for KL divergence)
extern "C" void launch_ldg_decode_with_logits(
    int input_token_id,
    int* output_token_id,
    float* logits_output,
    const void* embed_weight,
    const LDGLayerWeights* layer_weights,
    const void* final_norm_weight,
    const void* lm_head_weight,
    const void* cos_table,
    const void* sin_table,
    void* k_cache,
    void* v_cache,
    void* hidden_buffer,
    void* g_activations,
    void* g_residual,
    void* g_q,
    void* g_k,
    void* g_v,
    void* g_attn_out,
    void* g_mlp_intermediate,
    void* g_normalized,
    void* block_max_vals,
    void* block_max_idxs,
    int num_layers,
    int position,
    int cache_len,
    int max_seq_len,
    float attn_scale,
    cudaStream_t stream
) {
    ldg_configure_kernel_attributes();
    void* kernel_args[] = {
        (void*)&input_token_id,
        (void*)&embed_weight,
        (void*)&layer_weights,
        (void*)&final_norm_weight,
        (void*)&cos_table,
        (void*)&sin_table,
        (void*)&k_cache,
        (void*)&v_cache,
        (void*)&hidden_buffer,
        (void*)&g_activations,
        (void*)&g_residual,
        (void*)&g_q,
        (void*)&g_k,
        (void*)&g_v,
        (void*)&g_attn_out,
        (void*)&g_mlp_intermediate,
        (void*)&g_normalized,
        (void*)&num_layers,
        (void*)&position,
        (void*)&cache_len,
        (void*)&max_seq_len,
        (void*)&attn_scale
    };

    cudaLaunchCooperativeKernel(
        (void*)ldg_decode_kernel,
        dim3(LDG_NUM_BLOCKS),
        dim3(LDG_BLOCK_SIZE),
        kernel_args,
        0,
        stream
    );

    // Compute full logits
    ldg_lm_head_logits<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        logits_output
    );

    // Also compute argmax for the token output
    #if defined(LDG_LM_USE_WMMA)
    ldg_lm_head_phase1_wmma<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );
    #else
    ldg_lm_head_phase1<<<LDG_LM_NUM_BLOCKS, LDG_LM_BLOCK_SIZE, 0, stream>>>(
        (const float*)g_normalized,
        (const __nv_bfloat16*)lm_head_weight,
        (float*)block_max_vals,
        (int*)block_max_idxs
    );
    #endif

    ldg_lm_head_phase2<<<1, 256, 0, stream>>>(
        (const float*)block_max_vals,
        (const int*)block_max_idxs,
        output_token_id,
        LDG_LM_NUM_BLOCKS
    );
}
