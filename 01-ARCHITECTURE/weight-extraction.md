# Extracting the MTP Head from a Qwen3.5-27B Checkpoint

> **The recurring problem:** every framework strips the MTP weights on load. mlx-lm,
> `transformers`, and vLLM all have model classes with no `mtp` field, so the 15 MTP tensors
> are silently dropped during `load`/`convert`. Before you can do *any* speculative decoding,
> you have to pull the MTP tensors out of the raw safetensors shards yourself. This doc is how.

## Why this is necessary

> "First MTP inference implementation for Qwen3.5 in Python. **Every other framework strips
> MTP weights on load.** We reverse-engineered the architecture and built working inference."
> — `mlx-qwen-mtp/README.md`

The HF checkpoint *does* contain the trained MTP head (under `model.mtp.*`). The problem is
purely on the consumer side:
- **mlx-lm**: `mlx_lm.convert` / `mlx_lm.load` instantiate a Qwen3.5 Python model class that has
  no `mtp` submodule, so those keys never get assigned and are discarded.
- **llama.cpp**: `convert_hf_to_gguf.py` had `if name.startswith("mtp"): return` (see
  `tensor-layout-hf-to-gguf.md`).
- **vLLM / transformers**: same shape — no MTP module on the standard path.

So the extraction step is framework-agnostic in spirit: read the index, find the `mtp.*` keys,
pull them from the shards, apply the Qwen3.5 norm convention, (optionally) quantize, and write a
standalone `mtp_weights.safetensors`.

## The HF tensor names you're extracting

```
model.mtp.pre_fc_norm_hidden.weight
model.mtp.pre_fc_norm_embedding.weight
model.mtp.fc.weight                                  # eh_proj, 10240 -> 5120
model.mtp.layers.0.input_layernorm.weight
model.mtp.layers.0.self_attn.{q,k,v,o}_proj.weight
model.mtp.layers.0.self_attn.{q,k}_norm.weight
model.mtp.layers.0.post_attention_layernorm.weight
model.mtp.layers.0.mlp.{gate,up,down}_proj.weight
model.mtp.norm.weight                                # final norm before shared lm_head
```
(15 tensors; full shapes in `mtp-head-anatomy.md`.)

## The two critical extraction steps

### 1. The RMSNorm `+1.0` shift
Qwen3.5 stores RMSNorm weights as `w` where the computation is `x · (1 + w)`. mlx-lm applies the
`+1.0` shift in its Qwen3.5 *sanitize* hook **for the main model only** — and since it strips
the MTP tensors, the shift never reaches them. So **you must add `1.0` to every 1-D MTP norm
weight yourself** before the weights are usable. Both scripts shift these 7 suffixes:
```
.input_layernorm.weight   .post_attention_layernorm.weight
.q_norm.weight            .k_norm.weight            .norm.weight
.pre_fc_norm_embedding.weight   .pre_fc_norm_hidden.weight
```
Forgetting this gives wrong (but non-crashing) norms → garbage drafts.

### 2. Quantize the large matrices (optional, for 4-bit serving)
2-D weight matrices with `size > 1024` are quantized to **4-bit, group_size=64**
(`mx.quantize(..., group_size=64, bits=4)`), emitting `.weight`/`.scales`/`.biases` triplets;
small tensors and norms are kept as **bfloat16**. This matches the `mlx-community/Qwen3.5-27B-4bit`
quantization so the extracted head drops into the 4-bit model. The resulting head is ~**265 MB**
(MLX README) for the single block.

## The scripts (copied into `extract/`)

| Script | Origin | Input model | bf16 shards? | Invocation |
|---|---|---|---|---|
| `extract_weights.py` | `mlx-qwen-mtp/src/extract_weights.py` (identical copy in `qwen-ops/mlx/extract_weights.py`) | defaults to HF cache `~/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots`, or `--model-path` | loads via `safetensors.safe_open(framework="numpy")` | `python -m src.extract_weights -o mtp_weights.safetensors [--model-path DIR]` |
| `extract_mtp_huihui.py` | `qwen-ops/validation/extract_mtp_huihui.py` | **positional** fp16 HF dir (vanilla *or* abliterated, e.g. "huihui") | loads via `mx.load` (numpy safetensors can't parse bf16) | `python3 extract_mtp_huihui.py <fp16_hf_dir> <output.safetensors>` |

Both: read `model.safetensors.index.json` → `weight_map` filtered to keys containing `"mtp"` →
load only the needed shards → `+1.0` norm shift → 4-bit quantize large matrices → write one
safetensors file, printing per-tensor shapes/dtypes and MB before/after.

### Which to use
- **`extract_weights.py`** — the "parametrized extractor". Use for the standard
  `mlx-community/Qwen3.5-27B` flow; it can auto-discover the HF cache snapshot or take an
  explicit `--model-path`, and exposes `extract_mtp_weights(output_path=..., model_path=...)` as
  an importable function (this is what `mlx-qwen-mtp/README.md`'s Quick Start calls).
- **`extract_mtp_huihui.py`** — use when your source repo ships **bfloat16** shards (the numpy
  safetensors backend chokes on bf16; this one uses `mx.load`). Named for the
  *huihui abliterated* Qwen3.5-27B but works on any vanilla fp16/bf16 HF dir. Output is
  compatible with `mlx-qwen-mtp` / parallel-mtp-voting `load_mtp()`.

> The two scripts are near-identical in logic (same index parsing, same norm suffixes, same
> quantize params); the only real differences are the loader backend (numpy vs `mx.load`) and
> the CLI shape (function/`--flags` vs positional args). Kept both because each handles a
> distinct real-world case (HF-cache + numpy shards vs explicit-dir + bf16 shards).

## End-to-end (MLX)

```python
# 1) extract once
from src.extract_weights import extract_mtp_weights
extract_mtp_weights(model_path="mlx-community/Qwen3.5-27B-4bit",
                    output_path="src/mtp_weights.safetensors")

# 2) load main model, patch DeltaNet kernels, attach MTP head, generate
import mlx_lm
from src import patch_model, mtp_generate, load_mtp
model, tokenizer = mlx_lm.load("mlx-community/Qwen3.5-27B-4bit")
patch_model(model)                                   # fused Metal kernels (MLX domain)
mtp_head = load_mtp(model, weights_path="src/mtp_weights.safetensors")
out = mtp_generate(model, tokenizer, prompt="...", max_tokens=256, mtp_head=mtp_head)
```

`load_mtp` (in `mtp_head.py`) maps the extracted HF keys to the `MTPHead` module fields and
handles the quantized `.weight`/`.scales`/`.biases` triplets. (Full module: `mtp-head-anatomy.md`.)

## For llama.cpp
You do **not** run these scripts for the llama.cpp path — there the MTP tensors are kept inside
the GGUF by the fixed `convert_hf_to_gguf.py` (they land as `blk.64.nextn.*`). These extraction
scripts are the **MLX / vLLM** route, where the head is a separate `safetensors` file loaded
alongside the base model. Same tensors, two packaging strategies. See `tensor-layout-hf-to-gguf.md`
for the GGUF route.

---

## Related

- [`mtp-head-anatomy.md`](mtp-head-anatomy.md) — full module + shapes of the 15 tensors you extract here.
- [`tensor-layout-hf-to-gguf.md`](tensor-layout-hf-to-gguf.md) — the alternate (GGUF/llama.cpp) packaging, where extraction is unnecessary.
- [`extract/`](extract/) — the two copied extraction scripts (`extract_weights.py`, `extract_mtp_huihui.py`).
- [`../03-MLX/mtp-self-speculative-mlx.md`](../03-MLX/mtp-self-speculative-mlx.md) — the MLX inference path the extracted head feeds into.
- [`../04-VLLM-GPU/quantization.md`](../04-VLLM-GPU/quantization.md) — the vLLM 4-bit serving route that also loads the head as a separate file.

---

### Provenance
- `mlx-qwen-mtp/src/extract_weights.py` (= `qwen-ops/mlx/extract_weights.py`) → `extract/extract_weights.py`
- `qwen-ops/validation/extract_mtp_huihui.py` → `extract/extract_mtp_huihui.py`
- `mlx-qwen-mtp/README.md` — "every framework strips MTP on load", Quick Start, 265 MB head
- `mlx-qwen-mtp/src/mtp_head.py` — `load_mtp`, weight_map, `+1.0` shift rationale
