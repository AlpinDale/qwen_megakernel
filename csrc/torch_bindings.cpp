#include <Python.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/library.h>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME)                                               \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() {                                     \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT,                 \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module);                                           \
  }

struct LDGLayerWeights {
  const void *input_layernorm_weight;
  const void *q_proj_weight;
  const void *k_proj_weight;
  const void *v_proj_weight;
  const void *q_norm_weight;
  const void *k_norm_weight;
  const void *o_proj_weight;
  const void *post_attn_layernorm_weight;
  const void *gate_proj_weight;
  const void *up_proj_weight;
  const void *down_proj_weight;
};

extern "C" void launch_ldg_decode_persistent(
    int input_token_id, int *output_token_id, const void *embed_weight,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int position, int cache_len,
    int max_seq_len, float attn_scale, cudaStream_t stream);

void decode(torch::Tensor output_token, int64_t input_token_id,
            torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
            torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
            torch::Tensor cos_table, torch::Tensor sin_table,
            torch::Tensor k_cache, torch::Tensor v_cache,
            torch::Tensor hidden_buffer, torch::Tensor activations,
            torch::Tensor residual, torch::Tensor q, torch::Tensor k,
            torch::Tensor v, torch::Tensor attn_out,
            torch::Tensor mlp_intermediate, torch::Tensor normalized,
            torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
            int64_t num_layers, int64_t position, int64_t max_seq_len,
            double attn_scale) {
  launch_ldg_decode_persistent(
      static_cast<int>(input_token_id),
      static_cast<int *>(output_token.data_ptr()), embed_weight.data_ptr(),
      reinterpret_cast<const LDGLayerWeights *>(
          layer_weights_packed.data_ptr()),
      final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
      cos_table.data_ptr(), sin_table.data_ptr(), k_cache.data_ptr(),
      v_cache.data_ptr(), hidden_buffer.data_ptr(), activations.data_ptr(),
      residual.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
      attn_out.data_ptr(), mlp_intermediate.data_ptr(), normalized.data_ptr(),
      block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
      static_cast<int>(num_layers), static_cast<int>(position),
      static_cast<int>(position + 1), static_cast<int>(max_seq_len),
      static_cast<float>(attn_scale),
      c10::cuda::getCurrentCUDAStream().stream());
}

extern "C" void launch_ldg_generate_nosync(
    int first_token_id, int num_steps,
    const void *embed_weight, const LDGLayerWeights *layer_weights,
    const void *final_norm_weight, const void *lm_head_weight,
    const void *cos_table, const void *sin_table,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int *output_log,
    int num_layers, int start_position, int max_seq_len, float attn_scale,
    cudaStream_t stream);

torch::Tensor generate_nosync(
    int64_t first_token_id, int64_t num_steps,
    torch::Tensor embed_weight, torch::Tensor layer_weights_packed,
    torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
    torch::Tensor cos_table, torch::Tensor sin_table,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor hidden_buffer, torch::Tensor activations,
    torch::Tensor residual, torch::Tensor q, torch::Tensor k,
    torch::Tensor v, torch::Tensor attn_out,
    torch::Tensor mlp_intermediate, torch::Tensor normalized,
    torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
    int64_t num_layers, int64_t start_position, int64_t max_seq_len,
    double attn_scale) {

  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto output_log = torch::empty({num_steps}, torch::dtype(torch::kInt32).device(torch::kCUDA));

  launch_ldg_generate_nosync(
      static_cast<int>(first_token_id), static_cast<int>(num_steps),
      embed_weight.data_ptr(),
      reinterpret_cast<const LDGLayerWeights *>(layer_weights_packed.data_ptr()),
      final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
      cos_table.data_ptr(), sin_table.data_ptr(),
      k_cache.data_ptr(), v_cache.data_ptr(),
      hidden_buffer.data_ptr(), activations.data_ptr(), residual.data_ptr(),
      q.data_ptr(), k.data_ptr(), v.data_ptr(), attn_out.data_ptr(),
      mlp_intermediate.data_ptr(), normalized.data_ptr(),
      block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
      static_cast<int *>(output_log.data_ptr()),
      static_cast<int>(num_layers), static_cast<int>(start_position),
      static_cast<int>(max_seq_len), static_cast<float>(attn_scale),
      stream);

  // Single sync at end -- all N steps already queued
  cudaStreamSynchronize(stream);
  return output_log;
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("decode(Tensor output_token, int input_token_id, "
          "Tensor embed_weight, Tensor layer_weights_packed, "
          "Tensor final_norm_weight, Tensor lm_head_weight, "
          "Tensor cos_table, Tensor sin_table, "
          "Tensor k_cache, Tensor v_cache, "
          "Tensor hidden_buffer, Tensor activations, Tensor residual, "
          "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
          "Tensor mlp_intermediate, Tensor normalized, "
          "Tensor block_max_vals, Tensor block_max_idxs, "
          "int num_layers, int position, int max_seq_len, "
          "float attn_scale) -> ()");
  ops.impl("decode", torch::kCUDA, &decode);

  ops.def("generate_nosync(int first_token_id, int num_steps, "
          "Tensor embed_weight, Tensor layer_weights_packed, "
          "Tensor final_norm_weight, Tensor lm_head_weight, "
          "Tensor cos_table, Tensor sin_table, "
          "Tensor k_cache, Tensor v_cache, "
          "Tensor hidden_buffer, Tensor activations, Tensor residual, "
          "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
          "Tensor mlp_intermediate, Tensor normalized, "
          "Tensor block_max_vals, Tensor block_max_idxs, "
          "int num_layers, int start_position, int max_seq_len, "
          "float attn_scale) -> Tensor");
  ops.impl("generate_nosync", torch::kCUDA, &generate_nosync);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
