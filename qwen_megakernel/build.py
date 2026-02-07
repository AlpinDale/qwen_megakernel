"""JIT compilation of the megakernel CUDA extension."""

import os
from torch.utils.cpp_extension import load

_module = None
_DIR = os.path.dirname(os.path.abspath(__file__))
_CSRC = os.path.join(_DIR, "../csrc")

# RTX 5090 (sm_120) tuning flags.
KERNEL_FLAGS = [
    "-DLDG_NUM_BLOCKS=128",
    "-DLDG_BLOCK_SIZE=512",
    "-DLDG_LM_NUM_BLOCKS=1184",
    "-DLDG_LM_BLOCK_SIZE=256",
    "-DLDG_LM_ROWS_PER_WARP=2",
    "-DLDG_USE_UINT4",
    "-DLDG_ATTENTION_VEC4",
    "-DLDG_WEIGHT_LDCS",
    "-DLDG_MLP_SMEM",
]

CUDA_FLAGS = [
    "-O3", "--use_fast_math", "-std=c++17", "--expt-relaxed-constexpr",
    "-arch=sm_120a",
    f"-I{_CSRC}",
] + KERNEL_FLAGS


def get_extension():
    """Build (or return cached) the megakernel extension. Triggers torch.ops.qwen_megakernel_C.*"""
    global _module
    if _module is not None:
        return _module

    _module = load(
        name="qwen_megakernel_C",
        sources=[
            os.path.join(_CSRC, "torch_bindings.cpp"),
            os.path.join(_CSRC, "kernel.cu"),
        ],
        extra_cuda_cflags=CUDA_FLAGS,
        extra_cflags=[f"-I{_CSRC}"],
        verbose=False,
    )
    return _module
