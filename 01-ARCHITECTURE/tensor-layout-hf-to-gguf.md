# Tensor Archaeology — HuggingFace → GGUF for the Qwen3.5 MTP Head

> The canonical mapping of *which weight lives where* across the HF checkpoint, the GGUF file,
> and the llama.cpp in-memory layer struct — plus the five-part converter/loader/classifier fix
> that was needed before a real Qwen3.5-27B MTP GGUF would load and run. This is the deepest
> "tensor archaeology" doc in the corpus; if a tensor name confuses you, it is explained here.

## TL;DR

Out of the box, `convert_hf_to_gguf.py` **silently strips every MTP tensor** (`if name.startswith("mtp"): return`). Even after you teach the converter to keep them, four more
things are wrong: the block count doesn't include the MTP layer, the tensors are classified as
one-shot output tensors instead of per-layer repeating tensors, the loader doesn't allocate the
`nextn.*` slots, and the `post_attention_layernorm` lands in the `ffn_norm` slot (not where the
loader looked). Five coordinated fixes (six commits in the diff) make it work end-to-end.

## The on-disk GGUF layout (production, N=1)

MTP tensors live under `blk.64.nextn.*` (template `blk.%d.nextn.*` in `llama-arch.cpp`). The
MTP block reuses the *standard* per-layer attention/FFN slots (`attn_q`, `ffn_gate`, …) for its
transformer body, and adds four `nextn`-specific slots:

```
blk.64.attn_norm.weight          [5120]            # = HF layers.0.input_layernorm
blk.64.attn_q.weight             [12288, 5120]     # gated Q (Q‖gate, double width)
blk.64.attn_k.weight             [1024, 5120]      # GQA
blk.64.attn_v.weight             [1024, 5120]
blk.64.attn_output.weight        [5120, 6144]
blk.64.attn_q_norm.weight        [256]             # head_dim
blk.64.attn_k_norm.weight        [256]
blk.64.ffn_norm.weight           [5120]            # = HF post_attention_layernorm  (!! see below)
blk.64.ffn_gate.weight           [17408, 5120]
blk.64.ffn_up.weight             [17408, 5120]
blk.64.ffn_down.weight           [5120, 17408]
blk.64.nextn.eh_proj.weight      [5120, 10240]     # 2*n_embd -> n_embd  (embed-first concat)
blk.64.nextn.enorm.weight        [5120]            # = HF mtp.pre_fc_norm_embedding
blk.64.nextn.hnorm.weight        [5120]            # = HF mtp.pre_fc_norm_hidden
blk.64.nextn.shared_head_norm.weight  [5120]       # = HF mtp.norm (final norm)
# optional, ABSENT for Qwen3.5-27B (mtp_use_dedicated_embeddings = false):
# blk.64.nextn.embed_tokens.weight        [5120, 151936]  -> falls back to main tok_embd
# blk.64.nextn.shared_head_head.weight    [5120, 151936]  -> falls back to main output.weight
```

KV metadata added: `qwen3.nextn_predict_layers = 1` (key `%s.nextn_predict_layers`,
`LLM_KV_NEXTN_PREDICT_LAYERS`). That is the *only* new metadata; everything else is reused.

> **Two competing shape conventions appear in the sources.** The production `qwen-mtp-tensors`
> README and the MLX code give the head as `24 heads · head_dim 256` ⇒ `attn_q [12288,5120]`,
> `attn_q_norm [256]`. The *N=4 generalization* design doc (`qwen-mtp-research/docs/tensor-layout.md`)
> instead writes `attn_q [5120,5120]`, `attn_q_norm [128]` (a `40 heads · head_dim 128`, square-Q
> assumption). The latter is an earlier/hypothetical sketch for a not-yet-trained N=4 head, **not**
> the real Qwen3.5-27B head. Use `24/4/256` for the real model. (Logged in the ledger.)

### HF → GGUF name mapping table (the canonical map)

| HuggingFace (`model.mtp.*`) | GGUF | Shape (HF `[out,in]`) |
|---|---|---|
| `mtp.layers.0.input_layernorm.weight` | `blk.64.attn_norm.weight` | `[5120]` |
| `mtp.layers.0.post_attention_layernorm.weight` | `blk.64.ffn_norm.weight` | `[5120]` |
| `mtp.layers.0.self_attn.q_proj.weight` | `blk.64.attn_q.weight` | `[12288, 5120]` |
| `mtp.layers.0.self_attn.k_proj.weight` | `blk.64.attn_k.weight` | `[1024, 5120]` |
| `mtp.layers.0.self_attn.v_proj.weight` | `blk.64.attn_v.weight` | `[1024, 5120]` |
| `mtp.layers.0.self_attn.o_proj.weight` | `blk.64.attn_output.weight` | `[5120, 6144]` |
| `mtp.layers.0.self_attn.q_norm.weight` | `blk.64.attn_q_norm.weight` | `[256]` |
| `mtp.layers.0.self_attn.k_norm.weight` | `blk.64.attn_k_norm.weight` | `[256]` |
| `mtp.layers.0.mlp.gate_proj.weight` | `blk.64.ffn_gate.weight` | `[17408, 5120]` |
| `mtp.layers.0.mlp.up_proj.weight` | `blk.64.ffn_up.weight` | `[17408, 5120]` |
| `mtp.layers.0.mlp.down_proj.weight` | `blk.64.ffn_down.weight` | `[5120, 17408]` |
| `mtp.fc.weight` | `blk.64.nextn.eh_proj.weight` | `[5120, 10240]` |
| `mtp.pre_fc_norm_embedding.weight` | `blk.64.nextn.enorm.weight` | `[5120]` |
| `mtp.pre_fc_norm_hidden.weight` | `blk.64.nextn.hnorm.weight` | `[5120]` |
| `mtp.norm.weight` | `blk.64.nextn.shared_head_norm.weight` | `[5120]` |
| `mtp.shared_head.head.weight` (if present) | (falls back to main `output.weight`) | `[151936, 5120]` |

### DeltaNet (GDN) per-layer tensors (the backbone's 48 recurrent layers)

The MTP head reuses the *attention*-layer slots, but the 48 DeltaNet layers carry their own
`ssm_*` tensor set. For completeness (these are the GGUF tensors of layers 0–47, minus the 16
full-attention layers), the `QWEN35` arch registers (`llama-arch.cpp` `LLM_ARCH_QWEN35` list):

| GGUF tensor (`blk.%d.*`) | enum | Role |
|---|---|---|
| `ssm_in` | `LLM_TENSOR_SSM_IN` | fused input projection → QKVZ (built by `build_qkvz` in `qwen35.cpp`) |
| `ssm_conv1d` | `LLM_TENSOR_SSM_CONV1D` | causal depthwise conv1d kernel (kernel size 4) |
| `ssm_beta` | `LLM_TENSOR_SSM_BETA` | β (delta-rule write strength), sigmoid-gated |
| `ssm_alpha` | `LLM_TENSOR_SSM_ALPHA` | α (decay logits) |
| `ssm_dt` | `LLM_TENSOR_SSM_DT` | dt bias added to α before softplus |
| `ssm_a` | `LLM_TENSOR_SSM_A_NOSCAN` (`blk.%d.ssm_a`) | the precomputed `-A_log.exp()` gate scale (see transform below) |
| `ssm_norm` | `LLM_TENSOR_SSM_NORM` | gated RMSNorm on the recurrence output (`build_norm_gated` with `z`) |
| `ssm_out` | `LLM_TENSOR_SSM_OUT` | output projection |

> **Converter weight transforms (diff `01`, `modify_tensors`).** Two DeltaNet weights are
> *transformed at convert time*, not just renamed:
> - `*.A_log` → `data_torch = -torch.exp(data_torch)` — the GGUF `ssm_a` stores `-exp(A_log)`
>   directly, so the runtime gate is just `softplus(α+dt) * ssm_a` (no exp at inference; see
>   `qwen35.cpp:272` `gate = mul(alpha_softplus, ssm_a)  // -A_log.exp() * softplus`).
> - `*.dt_bias` → (dt-bias handling; the diff hunk truncates here but the runtime adds `ssm_dt`
>   to α before softplus at `qwen35.cpp:268`).
>
> The dense full-attention layers (16 of them) and the MTP head (layer 64) carry **no** `ssm_*`
> tensors; the `is_recurrent(i)` mask selects which layers get the GDN vs attention tensor set.
> (Recurrent-state shapes for these layers: see "DeltaNet recurrent state" in `qwen35-hybrid-architecture.md`.)

The converter implements the MTP-head non-`layers` remap explicitly:
```python
remapper = {
    "mtp.fc":                    f"model.layers.{n_base}.eh_proj",          # n_base = 64
    "mtp.pre_fc_norm_embedding": f"model.layers.{n_base}.enorm",
    "mtp.pre_fc_norm_hidden":    f"model.layers.{n_base}.hnorm",
    "mtp.norm":                  f"model.layers.{n_base}.shared_head.norm",
}
# and: name.replace("mtp.layers.0", f"model.layers.{n_base}")  for the transformer body
```

## The bug and the five (six-commit) fixes

The bug, verbatim from the source:
```python
def modify_tensors(self, data_torch, name, bid):
    if name.startswith("mtp"):
        return  # <-- the bug: MTP tensors silently dropped
```

Two commits carry the fix: `53075d24c` (load/convert/classify, diff 01) and `83babcae7`
(graph builder, diff 02). Both are copied verbatim into `tensor-diffs/`. The five logical fixes:

| # | File | Fix |
|---|---|---|
| 1 | `gguf-py/gguf/constants.py` | Add `FFN_NORM` + the 6 `NEXTN_*` tensor entries to the `QWEN35` MODEL_TENSORS list so the converter's `tensor_map` accepts the remapped MTP names. Strictly additive. |
| 2 | `convert_hf_to_gguf.py` (`Qwen3NextModel.__init__`) | Override `__init__` to bump `self.block_count += mtp_num_hidden_layers` after `super().__init__`, then **rebuild `self.tensor_map`** so block index 64 is in range (`get_tensor_name_map` otherwise rejects it). Gated on `n_mtp > 0` so the 80B-A3B MoE path is unaffected. Also `add_nextn_predict_layers(1)`, and rewrite `modify_tensors` to remap `mtp.*` instead of dropping it. |
| 3 | `src/llama-arch.cpp` | (a) In the `QWEN35` tensor-names list replace `ATTN_POST_NORM` with `FFN_NORM` so `tn()` resolves. (b) Add the 6 `NEXTN_*` names. (c) **Reclassify** the 6 `NEXTN_*` entries from `LLM_TENSOR_LAYER_OUTPUT` → `LLM_TENSOR_LAYER_REPEATING`: their template is `blk.%d.nextn.*`, so they are per-layer, not global output tensors. Misclassification made `create_tensor`'s layer-index sanity check reject them. |
| 4 | `src/llama-model.cpp` (loader) | Read `LLM_KV_NEXTN_PREDICT_LAYERS` (optional). Fix the recurrent mask so the MTP layer at `n_base` is **never** marked recurrent (`(i < n_base) && ((i+1) % full_attn_interval != 0)`). Extend the type switch: `(n_layer - nextn_predict_layers) == 64 → LLM_TYPE_27B`. Add a `layer.nextn.*` allocation block (eh_proj, enorm, hnorm, shared_head_norm **required**; embed_tokens, shared_head_head **TENSOR_NOT_REQUIRED**) gated on `i >= n_layer - nextn_predict_layers`. **Load `attn_post_norm` from the `FFN_NORM` slot**, not `ATTN_POST_NORM` (the post-attn norm naming fix; applied to both QWEN35 and QWEN35MOE). |
| 5 | `src/llama-hparams.h` / loader | Store `nextn_predict_layers` (default 0) in `llama_hparams`. |
| (6) | `src/models/qwen35.cpp` + `src/llama-context.cpp` | Graph side: rewrite `build_mtp_head` to mirror the working main path (use `build_attn` instead of manual K·Q/softmax/`ggml_repeat_4d`; 3D views of `Qcur_full`; **switch RoPE from `ggml_rope_ext` to `ggml_rope_multi`** with `rope_sections` and allocate `inp_pos_zero` length **4**; fall back to `model.output`/`model.tok_embd` when `shared_head_head` is null). `mtp_graph_compute` uploads the full length-4 `inp_pos_zero`, not a single int32. |

## The three discoveries (named gotchas)

### 1. `post_attention_layernorm` → `ffn_norm` slot (not `attn_post_norm`)
HF's `post_attention_layernorm` is emitted by `convert_hf_to_gguf.py` as `blk.N.ffn_norm.weight`
per standard Llama-family tensor mapping. But the graph builder reads it via the
`layer.attn_post_norm` field, and the loader originally created that field from the
`ATTN_POST_NORM` GGUF slot — which is empty. Result: a **null tensor** and a misleading
"tensor not found" error pointing at the wrong field. Fix: the QWEN35 loader must create
`layer.attn_post_norm` from the **`FFN_NORM`** slot. This was a pre-existing upstream bug — the
QWEN35 loader had never been exercised against a real converted GGUF. (Same fix for QWEN35MOE.)

### 2. MRoPE `inp_pos_zero` must be a 4-element tensor
Qwen3.5 uses MRoPE; the full-attention path calls `ggml_rope_multi` with a 4-element
`rope_sections` array. The MTP graph's position-0 marker `inp_pos_zero` must therefore be a
**length-4** tensor (one entry per rope section) for `ggml_rope_multi` — **not** the length-1
tensor that `ggml_rope_ext` would take. Getting it wrong fails an assertion deep inside the RoPE
op. (Note: the *earlier* graph diff `02` still used `ggml_rope_ext` + length-1, reasoning "at
pos 0 RoPE is identity so sections don't matter"; the *later* load diff `01` Fix 5 supersedes
this with `ggml_rope_multi` + length-4 to match the main path and pass the assertion. The
length-4 version is the resolved/current behavior.)

### 3. Dedicated-embeddings null pointer (`mtp_use_dedicated_embeddings: false`)
Qwen3.5-27B has `mtp_use_dedicated_embeddings: false` → the MTP head's LM projection uses the
**main model's `output.weight`**, not a dedicated `shared_head_head`. Initial loader/graph code
assumed the dedicated tensor existed and crashed with a null pointer. Fix: load
`nextn.shared_head_head` and `nextn.embed_tokens` as `TENSOR_NOT_REQUIRED`, and in the graph
**fall back to `model.output` (or `model.tok_embd` if tied)** when null.

## Known remaining issue (carried forward from commit `53075d24c`)
The MTP head runs end-to-end and produces plausible drafts, but the **spec-decode rollback path
hits backend scheduling errors** around recurrent-memory position management
(`find_slot: non-consecutive token position`) during verify. That is a separate subsystem — the
llama.cpp / theory domains own the rollback fix (this is the same family as "the bug that ate the
session" documented in the research/llamacpp domains).

## Generalizing to N>1 (forward-looking, design only)
The loader **already** supports `nextn_predict_layers > 1`: it iterates
`i >= n_layer - nextn_predict_layers` and loads per-layer `nextn.*` for each. The only reason it
ends at 1 today is the converter writes one MTP block and the upstream checkpoint only has one.
For an N=4 per-position-head design the GGUF would carry `blk.64..blk.67.nextn.*` and metadata
`qwen3.nextn_predict_layers = 4`; the converter loop would be
`gguf_block_idx = n_main_layers + k` for `k in 0..N-1`. Storage ≈ **210 MB/head Q4_K_M**,
**~840 MB for N=4** (+5.25% on a ~16 GB model). Quantization needs no special-casing —
`blk.N.nextn.*` quantize as ordinary block tensors. (Full design: `qwen-mtp-research/docs/per-position-heads.md`, distilled in the THEORY domain.)

---

## Related

- [`mtp-head-anatomy.md`](mtp-head-anatomy.md) — the 15 MTP tensors mapped here, and what the head computes with them.
- [`qwen35-hybrid-architecture.md`](qwen35-hybrid-architecture.md) — the recurrent mask, `full_attention_interval`, MRoPE, and DeltaNet state shapes referenced throughout.
- [`tensor-diffs/`](tensor-diffs/) — the verbatim converter/loader (`01`) and graph (`02`) diffs this doc annotates.
- [`../02-LLAMACPP/infrastructure-patches.md`](../02-LLAMACPP/infrastructure-patches.md) — patch 03 is these same five fixes as a llama.cpp commit.
- [`../02-LLAMACPP/source/`](../02-LLAMACPP/source/) — `llama-arch.cpp`, `llama-model.cpp`, `qwen35.cpp` (the files patched here).
- [`../05-THEORY-AND-DESIGNS/per-position-heads-design.md`](../05-THEORY-AND-DESIGNS/per-position-heads-design.md) — the N>1 generalization this layout already supports.

---

### Provenance
- `qwen-mtp-tensors/README.md` (= `qwen-ops/llamacpp/tensor-mapping/README.md`) — the bug, the 5 fixes, the 3 discoveries, the production HF→GGUF map
- `qwen-mtp-tensors/diffs/01-qwen35-tensor-load.diff` (commit `53075d24c`) → `tensor-diffs/01-qwen35-tensor-load.diff` — converter/loader/classifier source
- `qwen-mtp-tensors/diffs/02-qwen35-graph-tensors.diff` (commit `83babcae7`) → `tensor-diffs/02-qwen35-graph-tensors.diff` — graph builder source (earlier RoPE-ext variant)
- `qwen-mtp-research/docs/tensor-layout.md` (= `qwen-ops/research/findings/tensor-layout.md`) — N=4 generalization, storage estimates, `enorm/hnorm` GGUF naming, loader line refs (`llama-model.cpp:1789`, `:5493-5532`)
- `llama-mtp/src/llama-arch.{h,cpp}` — `LLM_TENSOR_NEXTN_*` enum + `blk.%d.nextn.*` templates + `LLM_KV_NEXTN_PREDICT_LAYERS`
- `llama-mtp/src/llama-hparams.h` — `nextn_predict_layers`, `rope_sections[4]`, `recurrent_layer_arr`
