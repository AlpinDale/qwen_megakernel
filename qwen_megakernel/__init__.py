"""Qwen Megakernel — single-kernel Qwen3-0.6B decode for RTX 5090."""

from qwen_megakernel.build import get_extension as _get_ext

_get_ext()

from qwen_megakernel.model import load_weights, Decoder, generate  # noqa: E402

__all__ = ["load_weights", "Decoder", "generate"]
