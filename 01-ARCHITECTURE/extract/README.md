# MTP weight-extraction scripts

Pull the 15-tensor Qwen3.5-27B MTP head out of a raw HF checkpoint, apply the Qwen3.5 RMSNorm
`+1.0` shift, quantize the large matrices to 4-bit (group_size=64), and write a standalone
`safetensors` file. Needed because **every framework strips the MTP tensors on load**.

Full narrative + step-by-step: `../weight-extraction.md`.

| File | Origin (verbatim copy) | Use when |
|---|---|---|
| `extract_weights.py` | `mlx-qwen-mtp/src/extract_weights.py` (byte-identical to `qwen-ops/mlx/extract_weights.py`) | Standard `mlx-community/Qwen3.5-27B` flow; auto-discovers HF cache or takes `--model-path`; importable `extract_mtp_weights()`. numpy-backed safetensors (fp16/fp32 shards). |
| `extract_mtp_huihui.py` | `qwen-ops/validation/extract_mtp_huihui.py` | Source ships **bfloat16** shards (uses `mx.load`, not numpy). Positional CLI: `<fp16_hf_dir> <output.safetensors>`. Named for the huihui-abliterated repo; works on any vanilla HF dir. |

Both require MLX (`import mlx.core`). They differ only in safetensors loader backend (numpy vs
`mx.load`) and CLI shape. Output is compatible with `mlx-qwen-mtp` / parallel-mtp-voting
`load_mtp()`.

Quantization detail: 2-D matrices with `size > 1024` → 4-bit `mx.quantize(group_size=64, bits=4)`
(`.weight`/`.scales`/`.biases`); everything else (norms, small tensors) kept bf16. Resulting head
≈ 265 MB.
