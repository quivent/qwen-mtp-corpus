# W4A16 Quantization for Qwen3.5-27B (AWQ vs GPTQ)

How to quantize the hybrid DeltaNet+attention model to 4-bit while keeping the
MTP head usable for speculative decoding. Covers the dual-layer-type handling,
the post-quantization MTP weight injection, the conv1d/llmcompressor fix, and
the measured AWQ-vs-GPTQ results.

## Headline finding: GPTQ > AWQ for MTP serving

On RTX 5090 (vLLM 0.19.0, MTP=5):

| Metric | GPTQ W4A16 | AWQ W4A16 |
|---|---:|---:|
| Single 256 tok | **151 tok/s** | 77 tok/s |
| MTP acceptance | **51%** | 31% |
| Batch=4 aggregate | **347 tok/s** | 313 tok/s |
| Model size | 19.5 GB | 18.6 GB |

**Why:** GPTQ's Hessian-optimal rounding preserves MTP-head quality better than
AWQ's activation-aware approach, so the draft head keeps a higher acceptance
rate and the spec-decode throughput is higher. **For MTP-enabled serving, GPTQ
is recommended.** AWQ may win for non-MTP workloads (and it does win batch
throughput in one preset: `vllm-serve.sh` quotes AWQ 403 vs GPTQ 347 at batch=4
on the cyankiwi-AWQ-textonly vs Huihui-GPTQ models — the comparison there is
across different base models, so read it as "both ~quant-equivalent for batch").

## The hard part: two layer types in one model

The model has two structurally different decoder layers, and the quantizer must
branch on which one it is sees:

- **`full_attention` layers** (16): standard `self_attn` with `q_proj` (2× width
  due to gating), `k_proj`, `v_proj`, `o_proj`, plus the SwiGLU MLP.
- **`linear_attention` (GDN/DeltaNet) layers** (48): `linear_attn` with
  `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`, plus a
  `conv1d` + gated-delta-rule core that has no clean linear "prev_op" chain.

### AutoAWQ model class (`quantization/qwen3_5.py`)

`Qwen3_5AWQForCausalLM` handles both. Key decisions:

```python
modules_to_not_convert = ["visual", "mtp", "in_proj_b", "in_proj_a"]
```

- **`visual`** — vision encoder kept at full precision (or stripped entirely).
- **`mtp`** — the MTP head is kept at full precision (AWQ would otherwise
  degrade it; and AWQ drops it on save anyway — see injection below).
- **`in_proj_b`, `in_proj_a`** — the small GDN gating projections have **48
  out_features, not divisible by pack_num=8**, so they cannot be packed into the
  4-bit layout. Excluded from both quantization and AWQ scaling.

`get_layers_for_scaling()` returns different scaling groups per layer type:
- Full-attention: `input_layernorm → q/k/v_proj`; `v_proj → o_proj` (only when
  `v_proj.shape == o_proj.shape`, since `q_proj` is 2× width); MLP gate/up; MLP
  down.
- Linear-attention: `input_layernorm → in_proj_qkv, in_proj_z` (skip
  `in_proj_a/b`); **out_proj scaling skipped** (no clean prev_op after conv1d +
  gated delta rule); MLP gate/up; MLP down.

`fuse_layers()` fuses QKV only for full-attention layers; GDN layers are left
as-is (conv1d + gated delta rule is incompatible with standard QKV fusion).

`move_embed()` handles the VL nesting: for `ConditionalGeneration` models the
layers live at `model.model.language_model.layers`, and it creates a
`model.model.rotary_emb` alias so the quantizer's hardcoded
`self.model.model.rotary_emb(...)` path (quantizer.py line 162) still works.

### The 4 AutoAWQ patches (`quantization/*.patch`)

To register and run this model class in stock AutoAWQ:

| Patch | What it does |
|---|---|
| `__init__.py.patch` | `from .qwen3_5 import Qwen3_5AWQForCausalLM` in `awq/models/__init__.py`. |
| `auto.py.patch` | Adds `"qwen3_5": Qwen3_5AWQForCausalLM` to the auto-mapping dict. |
| `base.py.patch` | Maps `qwen3_5 → AutoModelForImageTextToText` (it's a VL model) and adds that class to the `use_cache` exclusion. |
| `quantizer.py.patch` | (1) Supports the nested `language_model.rotary_emb` path; (2) for small layers like GDN `in_proj_b` (48 out_features), falls back `oc_batch_size = org_w_shape[0]` so the `% oc_batch_size == 0` assert doesn't fire. |

## Post-quantization MTP weight injection (`quantization/inject_mtp_weights.py`)

AutoAWQ **drops the MTP head during save** (it's in `modules_to_not_convert`).
Without the MTP tensors, vLLM cannot do MTP speculative decoding. The injector
copies them back:

1. Extract all tensors matching `^mtp` from the **source** (unquantized) model
   (15 MTP tensors — see below).
2. Save them to `model_mtp.safetensors` in the quantized model dir.
3. Add each key to `model.safetensors.index.json` `weight_map`; bump
   `metadata.total_size`.
4. Verify byte-equality of saved vs source tensors.
5. Ensure `config.json` (or `text_config`) has `mtp_num_hidden_layers` (copied
   from source) so vLLM recognizes the MTP head.

The 15 MTP tensor keys (from `microgreens/mtp_clone.py`):
```
mtp.fc.weight
mtp.layers.0.input_layernorm.weight
mtp.layers.0.mlp.{down,gate,up}_proj.weight
mtp.layers.0.post_attention_layernorm.weight
mtp.layers.0.self_attn.{k,q}_norm.weight
mtp.layers.0.self_attn.{k,o,q,v}_proj.weight
mtp.norm.weight
mtp.pre_fc_norm_embedding.weight
mtp.pre_fc_norm_hidden.weight
```

Usage: `python inject_mtp_weights.py <source_model> <quantized_model>`.

## The conv1d / llmcompressor fix (`patches/llmcompressor-conv1d.patch`)

When using **llmcompressor** (the GPTQ/compressed-tensors path) instead of
AutoAWQ, it tries to import `Conv1D` from the wrong transformers module. One-line
fix:

```diff
- from transformers.modeling_utils import Conv1D as TransformerConv1D
+ from transformers.pytorch_utils import Conv1D as TransformerConv1D
```

Newer transformers moved `Conv1D` to `pytorch_utils`; without this,
llmcompressor's module-type detection throws and quantization of the GDN
conv1d-bearing layers fails.

## A hand-rolled GPTQ (RTN) quantizer (`optimizations/scripts/quantize_deltanet.py`)

For the dedicated **DeltaNet draft model** (`Qwen3.5-27B-DeltaNet-draft`), a
CPU-only Round-To-Nearest GPTQ quantizer produces vLLM-loadable W4A16
safetensors:
- 4-bit, group_size=128, symmetric.
- Quantizes `in_proj_{qkv,z,a,b}`, `out_proj`, MLP `{gate,up,down}_proj` — but
  **only when `in_features % 128 == 0`** (so the 48-wide GDN projections are
  skipped, same constraint as AWQ).
- Skips `lm_head` and `embed_tokens`.
- GPTQ pack format: `qweight [in/8, out] int32`, `scales [in/group, out] fp16`,
  `qzeros [in/group, out/8] int32`; symmetric zero_point = 8.
- Writes `quantization_config` (`quant_method: gptq, desc_act: false`) into
  `config.json`. Reports a ~42.7 GB → 4-bit compression ratio.

## Models on disk / on HF (provenance)

| Model | Quant | Size | Notes |
|---|---|---|---|
| `j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-textonly` | AWQ/compressed-tensors W4A16 | ~19.1 GB | **Canonical GH200 model.** Vision stripped, stock working MTP. ~186 tok/s baseline. |
| `j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-retrained-mtp` | AWQ W4A16 | — | **DO NOT USE.** Abandoned retrained draft head, broken weight mapping → ~0% MTP acceptance → ~3× slowdown. |
| `j-a-a-a-y/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16` | GPTQ W4A16 | 19.5 GB | Abliterated; best RTX 5090 single-request (151 tok/s, 51% accept). |
| `j-a-a-a-y/Huihui-Qwen3.5-27B-abliterated-{AWQ,CT}-W4A16` | AWQ / compressed-tensors | — | Abliterated variants. |
| `cyankiwi/Qwen3.5-27B-AWQ-4bit` | AWQ | — | Original community AWQ, still has vision. |

> Vision-strip and stale-index gotchas for these models are in
> [`deployment.md`](deployment.md).

## Related

- [`deployment.md`](deployment.md) — serving the quantized models; vision-strip + stale-index gotchas.
- [`vllm-patches-and-strategies.md`](vllm-patches-and-strategies.md) — `llmcompressor-conv1d.patch`, `quantize_deltanet.py`, INT8-embedding patch.
- [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) — the two layer types (48 DeltaNet / 16 attention) the quantizer branches on.
- [`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md) — the MTP head tensors that injection copies back.
- [`README.md`](README.md) — the RTX 5090 / GH200 results tables.
- [`../00-OVERVIEW/glossary.md`](../00-OVERVIEW/glossary.md) — W4A16, AWQ, GPTQ, compressed-tensors.

---
*Provenance: `qwen-ops/research/benchmarks/autoawq-benchmarks.md`,
`qwen-ops/quantization/autoawq-qwen35/*` (qwen3_5.py, inject_mtp_weights.py, the
4 .patch files), `qwen-ops/vllm/patches/llmcompressor-conv1d.patch`,
`qwen-ops/vllm/scripts/quantize_deltanet.py`,
`qwen-ops/vllm/microgreens/mtp_clone.py` (MTP key list).*
