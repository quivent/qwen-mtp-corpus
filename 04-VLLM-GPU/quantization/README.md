# quantization/ — AutoAWQ for Qwen3.5

Copied verbatim from `qwen-ops/quantization/autoawq-qwen35/`.

- `qwen3_5.py` — `Qwen3_5AWQForCausalLM` model class (dual full-attention /
  GDN linear-attention handling, QKV fusion only for attention layers).
- `inject_mtp_weights.py` — post-quantization MTP head injector (AWQ drops MTP
  on save).
- `__init__.py.patch`, `auto.py.patch`, `base.py.patch`, `quantizer.py.patch` —
  patches to register/run this class in stock AutoAWQ.

See [`../quantization.md`](../quantization.md) for the full write-up and the
AWQ-vs-GPTQ results. The GPTQ RTN quantizer for the DeltaNet draft model lives
at `../optimizations/scripts/quantize_deltanet.py`, and the llmcompressor conv1d
fix at `../patches/llmcompressor-conv1d.patch`.
