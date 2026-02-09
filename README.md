## Qwen 0.6B Megakernel for RTX 5090

This megakernel is aggressively optimized for Qwen3-0.6B (bf16) shapes to be run on an RTX 5090. It still needs a lot of work, and is only really fast in specific scenarios.


| Backend      | tok/s  | ms/tok | Speedup |
|--------------|--------|--------|---------|
| PyTorch (HF) | 123.3  | 8.11   | 1.00x   |
| Megakernel   | 1008.3 | 0.99   | 8.17x   |


To use this:

```bash
uv pip install -r requirements.txt
python -m qwen_megakernel.bench
```

Not tested on any other GPU, and likely won't run or work. Needs at least CUDA 12.8.


### Credits
Based on Elliot Arledge's [MegaQwen](https://github.com/Infatoshi/MegaQwen) for the RTX 3090 GPU.
