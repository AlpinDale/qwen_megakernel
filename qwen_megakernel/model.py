"""Weight loading and high-level decode API for Qwen3-0.6B."""

import math
import struct

import torch

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM      # 2048
KV_SIZE = 8 * HEAD_DIM      # 1024
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 151936

_decode = torch.ops.qwen_megakernel_C.decode


def load_weights(model_name="Qwen/Qwen3-0.6B"):
    """Load Qwen3-0.6B weights from HuggingFace into GPU tensors."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    state = model.state_dict()

    # RoPE tables
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    # Per-layer weight list (11 tensors per layer, flattened)
    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        layer_weights.extend([
            state[p + "input_layernorm.weight"].contiguous(),
            state[p + "self_attn.q_proj.weight"].contiguous(),
            state[p + "self_attn.k_proj.weight"].contiguous(),
            state[p + "self_attn.v_proj.weight"].contiguous(),
            state[p + "self_attn.q_norm.weight"].contiguous(),
            state[p + "self_attn.k_norm.weight"].contiguous(),
            state[p + "self_attn.o_proj.weight"].contiguous(),
            state[p + "post_attention_layernorm.weight"].contiguous(),
            state[p + "mlp.gate_proj.weight"].contiguous(),
            state[p + "mlp.up_proj.weight"].contiguous(),
            state[p + "mlp.down_proj.weight"].contiguous(),
        ])

    weights = dict(
        embed_weight=state["model.embed_tokens.weight"].contiguous(),
        layer_weights=layer_weights,
        final_norm_weight=state["model.norm.weight"].contiguous(),
        lm_head_weight=state["lm_head.weight"].contiguous(),
        cos_table=cos_table,
        sin_table=sin_table,
    )

    del model
    torch.cuda.empty_cache()
    return weights, tokenizer


def _pack_layer_weights(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    """Pack 11-tensor-per-layer flat list into a device blob of LDGLayerWeights structs."""
    ptr_size = 8  # 64-bit pointers
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    t = torch.frombuffer(buf, dtype=torch.uint8).cuda()
    return t


class Decoder:
    """Stateful decoder wrapping the Qwen Megakernel torch ops."""

    def __init__(self, weights=None, tokenizer=None, model_name="Qwen/Qwen3-0.6B"):
        if weights is None:
            weights, tokenizer = load_weights(model_name)
        self.tokenizer = tokenizer
        self._position = 0

        # Keep references so tensors stay alive (prevents GC of weight memory).
        self._weights = weights

        # Model weights (read-only, shared across calls)
        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])

        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        # KV cache
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda")
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers (single-token decode)
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden    = torch.empty(HIDDEN_SIZE, **bf16)
        self._act       = torch.empty(HIDDEN_SIZE, **f32)
        self._res       = torch.empty(HIDDEN_SIZE, **f32)
        self._q         = torch.empty(Q_SIZE, **f32)
        self._k         = torch.empty(KV_SIZE, **f32)
        self._v         = torch.empty(KV_SIZE, **f32)
        self._attn_out  = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out  = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def step(self, token_id: int) -> int:
        """Decode one token. Returns the next token id."""
        _decode(
            self._out_token, token_id,
            self._embed_weight, self._layer_weights_packed,
            self._final_norm_weight, self._lm_head_weight,
            self._cos_table, self._sin_table,
            self._k_cache, self._v_cache,
            self._hidden, self._act, self._res,
            self._q, self._k, self._v, self._attn_out,
            self._mlp_inter, self._norm_out,
            self._bmax_vals, self._bmax_idxs,
            NUM_LAYERS, self._position, MAX_SEQ_LEN, self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        self.reset()
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        for tid in ids[:-1]:
            self.step(tid)
        out, tok = [], ids[-1]
        eos = self.tokenizer.eos_token_id
        for _ in range(max_tokens):
            tok = self.step(tok)
            if tok == eos:
                break
            out.append(tok)
        return self.tokenizer.decode(out, skip_special_tokens=True)


def generate(prompt: str, max_tokens: int = 100) -> str:
    """One-shot convenience: load model, generate, return text."""
    return Decoder().generate(prompt, max_tokens)
