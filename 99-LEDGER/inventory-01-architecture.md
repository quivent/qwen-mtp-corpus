# Inventory — Agent A (01-ARCHITECTURE)

Domain: Qwen3.5-27B architecture, MTP head anatomy, HF→GGUF tensor mapping, weight extraction.
Owner folder: `MTP/01-ARCHITECTURE/`.

## Files produced
- `MTP/01-ARCHITECTURE/qwen35-hybrid-architecture.md`
- `MTP/01-ARCHITECTURE/mtp-head-anatomy.md`
- `MTP/01-ARCHITECTURE/tensor-layout-hf-to-gguf.md`
- `MTP/01-ARCHITECTURE/weight-extraction.md`
- `MTP/01-ARCHITECTURE/tensor-diffs/README.md`
- `MTP/01-ARCHITECTURE/extract/README.md`
- `MTP/01-ARCHITECTURE/tensor-diffs/01-qwen35-tensor-load.diff` (copied artifact)
- `MTP/01-ARCHITECTURE/tensor-diffs/02-qwen35-graph-tensors.diff` (copied artifact)
- `MTP/01-ARCHITECTURE/extract/extract_weights.py` (copied artifact)
- `MTP/01-ARCHITECTURE/extract/extract_mtp_huihui.py` (copied artifact)

## Source files consumed

| Source file | Absorbed into | Status |
|---|---|---|
| `qwen-mtp-tensors/README.md` | `tensor-layout-hf-to-gguf.md`, `qwen35-hybrid-architecture.md`, `mtp-head-anatomy.md` | distilled (canonical "tensor archaeology" source) |
| `qwen-ops/llamacpp/tensor-mapping/README.md` | (same) | duplicate-of `qwen-mtp-tensors/README.md` (verified byte-identical via `diff`) |
| `qwen-mtp-tensors/diffs/01-qwen35-tensor-load.diff` | `tensor-layout-hf-to-gguf.md` + copied to `tensor-diffs/` | distilled + copied |
| `qwen-mtp-tensors/diffs/02-qwen35-graph-tensors.diff` | `tensor-layout-hf-to-gguf.md`, `mtp-head-anatomy.md` + copied to `tensor-diffs/` | distilled + copied |
| `qwen-ops/llamacpp/tensor-mapping/01-qwen35-tensor-load.diff` | (same as 01 above) | duplicate-of `qwen-mtp-tensors/diffs/01...` (byte-identical) |
| `qwen-ops/llamacpp/tensor-mapping/02-qwen35-graph-tensors.diff` | (same as 02 above) | duplicate-of `qwen-mtp-tensors/diffs/02...` (byte-identical) |
| `qwen-mtp-research/docs/tensor-layout.md` | `tensor-layout-hf-to-gguf.md` (N=4 generalization, storage est., enorm/hnorm naming, loader line refs) | distilled |
| `qwen-ops/research/findings/tensor-layout.md` | (same) | duplicate-of `qwen-mtp-research/docs/tensor-layout.md` (byte-identical) |
| `mlx-qwen-mtp/README.md` | `mtp-head-anatomy.md` (15 tensors, shapes), `qwen35-hybrid-architecture.md`, `weight-extraction.md` | distilled (MLX perf/journey numbers belong to MLX domain — only arch sections taken here) |
| `mlx-qwen-mtp/src/mtp_head.py` | `mtp-head-anatomy.md` (forward computation, dims, weight_map, chaining), `weight-extraction.md` (`load_mtp`) | distilled |
| `qwen-ops/mlx/mtp_head.py` | (same) | duplicate-of `mlx-qwen-mtp/src/mtp_head.py` (same module; confirmed identical config defaults) |
| `qwen-ops/mlx/extract_weights.py` | `weight-extraction.md` + copied to `extract/` | distilled + copied |
| `mlx-qwen-mtp/src/extract_weights.py` | (same) | duplicate-of `qwen-ops/mlx/extract_weights.py` (byte-identical via `diff`); copied this canonical one |
| `qwen-ops/validation/extract_mtp_huihui.py` | `weight-extraction.md` + copied to `extract/` | distilled + copied |
| `llama-mtp/src/llama-arch.h` | `tensor-layout-hf-to-gguf.md`, `mtp-head-anatomy.md` | distilled (read, not copied — fork referenced by path) |
| `llama-mtp/src/llama-arch.cpp` | `tensor-layout-hf-to-gguf.md` (`blk.%d.nextn.*` templates, KV key, classifier) | distilled (read, not copied) |
| `llama-mtp/src/llama-hparams.h` | `qwen35-hybrid-architecture.md`, `tensor-layout-hf-to-gguf.md` (`nextn_predict_layers`, `rope_sections[4]`, `recurrent_layer_arr`) | distilled (read, not copied) |
| `llama-mtp/src/models/qwen35.cpp` | `mtp-head-anatomy.md` (graph build, MTP branch, il_mtp), `qwen35-hybrid-architecture.md` | distilled (read, not copied — MTP source file also owned by Agent B for full copy) |
| `llama-mtp/src/llama-model.cpp` | `tensor-layout-hf-to-gguf.md` (loader fixes, type detection, recurrent mask), `qwen35-hybrid-architecture.md` | distilled (read, not copied — grep only) |
| `qwen-mtp-research/docs/per-position-heads.md` | `qwen35-hybrid-architecture.md` (vocab math), `tensor-layout-hf-to-gguf.md` (N>1 generalization) | partial — full distillation is Agent E (THEORY) territory; only arch-fact slivers taken |
| `qwen-mtp-research/README.md` | `qwen35-hybrid-architecture.md` (hybrid framing), `mtp-head-anatomy.md` (single-head, chaining) | partial — "the bug"/"the recipe"/methodology are Agent B/E; only architecture facts taken |
| `qwen-ops/research/findings/modal-mtp-precision-divergence.md` | `qwen35-hybrid-architecture.md` (3:1 pattern, 77 MB state snapshot, DeltaNet-only correctness) | partial — full doc is Agent D (vLLM/modal) territory; only arch facts taken |
| `qwen-ops/vllm/microgreens/mtp_diversity_train.py` | `qwen35-hybrid-architecture.md`, `mtp-head-anatomy.md` (config values: 5120/17408/24/4/256) | partial — config-value cross-check only; training code is Agent D |
| `qwen-ops/vllm/optimizations/sibling_sequential.py` | (same config cross-check) | partial — Agent D owns full file |
| `qwen-ops/vllm/optimizations/selective_state_snapshot.py` | `qwen35-hybrid-architecture.md` (DeltaNet linear dims) | partial — Agent D owns full file |
| `qwen-ops/research/findings/mlx-reference.md` | (referenced for "single head, chained" confirmation) | deferred — Agent C (MLX) canonical home; cross-checked only |
| `qwen-ops/research/findings/TIMELINE.md`, `DISPATCH_BARRIER_PROFILING.md`, `BANDWIDTH_ANALYSIS.md` | (skimmed for "48 DeltaNet/16 attn" confirmation) | deferred — Agent C (MLX journey) territory |
| `MTP/03-MLX/src/fused_gdn.py` | `qwen35-hybrid-architecture.md` ("DeltaNet recurrent state" — **authoritative** runtime ssm/conv state shapes) | distilled (arch sliver; full kernel-fusion analysis is Agent C/MLX) |
| `MTP/03-MLX/src/generate.py` | (cross-check; identical GDN state alloc to fused_gdn.py) | deferred — Agent C owns; arch shape cross-checked |
| `llama-mtp/src/models/qwen3next.cpp` | `qwen35-hybrid-architecture.md` (GDN state `reshape_4d`, ssm dim mapping — cross-check vs qwen35.cpp) | distilled (read, not copied — Agent B owns full copy) |
| `llama-mtp/src/llama-hparams.cpp` | `qwen35-hybrid-architecture.md` (`n_embd_s`/`n_embd_r` state-size formulas) | distilled (read; arch slice) |
| `modal-mtp/README.md` | `qwen35-hybrid-architecture.md` (state `[48,128,128]`, 77/150 MB) | partial — Agent D (modal/vLLM) owns full doc; arch state facts taken + reconciled |
| `modal-mtp/patches/02-draft-mode-and-state.patch` | (snapshot/restore conv+ssm state confirmation) | partial — Agent D owns; arch cross-check only |
| `recurrent-rollback/README.md`, `recurrent-rollback/docs/TECHNIQUE.md` | `qwen35-hybrid-architecture.md` (state-shape **correction**: `(14,256,256)` → `(48,128,128)`) | partial — Agent E (theory/rollback) owns full technique; arch shape adjudicated + corrected here |
| `qwen-ops/vllm/optimizations/selective_state_snapshot.py` | `qwen35-hybrid-architecture.md` (**authoritative** ssm/conv shapes, `num_value_heads=48`, conv_kernel=4) | partial — Agent D owns full file; arch shapes taken |
| `qwen-ops/vllm/optimizations/deltanet_adjuster.py` | `qwen35-hybrid-architecture.md` (`linear_num_key_heads=16`; flagged placeholder `num_value_heads=32`) | partial — Agent D owns; arch config cross-check |

## Duplications collapsed (one canonical home each)
1. **HF→GGUF tensor-mapping README**: `qwen-mtp-tensors/README.md` ≡ `qwen-ops/llamacpp/tensor-mapping/README.md` → canonical home `tensor-layout-hf-to-gguf.md`.
2. **N=4 tensor-layout doc**: `qwen-mtp-research/docs/tensor-layout.md` ≡ `qwen-ops/research/findings/tensor-layout.md` → folded into `tensor-layout-hf-to-gguf.md` (generalization section).
3. **The two `.diff` files**: `qwen-mtp-tensors/diffs/*` ≡ `qwen-ops/llamacpp/tensor-mapping/*.diff` → one copy in `tensor-diffs/`.
4. **`extract_weights.py`**: `mlx-qwen-mtp/src/` ≡ `qwen-ops/mlx/` → one copy in `extract/`.
5. **`mtp_head.py`**: `mlx-qwen-mtp/src/` ≡ `qwen-ops/mlx/` → distilled once into `mtp-head-anatomy.md` (file copy itself is Agent C / MLX `src/` territory).

All "≡" confirmed with `diff` (byte-identical) except `mtp_head.py` (confirmed identical config
defaults + module).

## Contradictions & evolving threads

1. **Attention head shape: `24/4/256` vs `40/8/128`.**
   - **Resolved/canonical (real Qwen3.5-27B MTP head):** `num_heads=24, num_kv_heads=4, head_dim=256` → `attn_q [12288,5120]`, `attn_q_norm [256]`. Sourced from the working `mtp_head.py`, the MLX README's explicit shapes (q 5120→12288, o 6144→5120), and three vLLM configs that read the real `text_config`.
   - **Earlier/hypothetical (NOT the real model):** the N=4 generalization doc `qwen-mtp-research/docs/tensor-layout.md` uses `attn_q [5120,5120]`, `attn_q_norm [128]` (a `40 heads · head_dim 128`, square-Q sketch) for a *not-yet-trained* N=4 head, and `per-position-heads.md` writes "GQA: n_head=40, n_head_kv=8, head_dim=128". These are design placeholders, not measured values. Documented as such in `tensor-layout-hf-to-gguf.md` and `qwen35-hybrid-architecture.md`.

2. **MTP prediction offset: "+1" vs "+2".**
   - `qwen-mtp-tensors/README.md`: head predicts "the +1 token from the layer-63 hidden state".
   - `mlx-qwen-mtp/README.md`: head predicts "token t+2 from hidden at t and embedding of token t+1".
   - **Resolved:** same operation, two reference points. Given `h_t` + embedding of the next committed token (`t+1`), it outputs logits for the token after (`t+2` w.r.t. hidden, `+1` w.r.t. the embedding input). Not a contradiction. Explained in `mtp-head-anatomy.md`.

3. **MTP-head RoPE: `ggml_rope_ext` (len-1 pos) vs `ggml_rope_multi` (len-4 pos).**
   - Graph diff `02` (commit `83babcae7`, 16:39) used `ggml_rope_ext` with length-1 `inp_pos_zero`, rationalized as "at pos 0 RoPE is identity, sections don't matter."
   - Load diff `01` (commit `53075d24c`, 17:16, **later**) Fix 5 switched to `ggml_rope_multi` with `rope_sections` and **length-4** `inp_pos_zero` to match the main MRoPE path and pass an internal assertion.
   - **Resolved:** length-4 `ggml_rope_multi` is the current/correct behavior. Both preserved in `tensor-layout-hf-to-gguf.md` discovery #2.

4. **RoPE dimension numbers: `rope_dimensions = [12,12,12,12]` vs rotary dims `= 64`. — RESOLVED (iter 2).**
   - `qwen-mtp-tensors/README.md` states MRoPE `rope_dimensions = [12,12,12,12]` (sum 48).
   - `mtp_head.py` computes rotary dims as `head_dim(256)·partial_rotary(0.25) = 64`.
   - **Resolved (iter 2):** `n_rot = 64` is **authoritative** — the converter writes
     `add_rope_dimension_count(int(head_dim·partial_rotary)) = int(256·0.25) = 64` (diff `01:103`),
     and the loader reads it into `hparams.n_rot` (`llama-model.cpp:619`, asserted against
     `n_embd_head_k`). `n_rot=64` is the total rotary-dim count passed to `ggml_rope_multi`/`_ext`.
     The `rope_sections[4]` array (the MRoPE per-axis split) is read separately from
     `LLM_KV_ROPE_DIMENSION_SECTIONS` (`llama-model.cpp:1034`). The README's `[12,12,12,12]` is an
     **unverified placeholder**: a valid GGML `sections[]` for this model must sum to `n_rot/2 = 32`
     (not 48), so `[12,12,12,12]` cannot be the real section array. The four section integers are not
     present in any in-corpus config/GGUF, so they remain unpinned — but nothing in the load/graph
     path depends on the specific split at pos 0 (RoPE is identity at pos 0), only on length 4.
     Downgraded from contradiction to "one source's `[12,12,12,12]` unverified; `n_rot=64`
     authoritative." Captured in `qwen35-hybrid-architecture.md` (MRoPE section + core-dims table).

6. **DeltaNet recurrent-state shape: `[48,128,128]` vs `(14,256,256)`. — ADJUDICATED (iter 2, cross-domain; Agent 2A owns).**
   - `modal-mtp/README.md`: state `[48, 128, 128]` per layer, ~77 MB/request snapshot (150 MB at FP32).
   - `recurrent-rollback/README.md` + `docs/TECHNIQUE.md`: state `(n_heads=14, d_k=256, d_v=256)`,
     1.75 MB/layer × 48 = 84 MB.
   - **Authoritative answer (settled from source code):** per-layer GatedDeltaNet recurrent state is
     `ssm_state = (num_v_heads=48, head_v_dim=128, head_k_dim=128)` = 786,432 elements, **plus** a
     `conv_state = (conv_kernel-1=3, conv_channels=10240)` = 30,720 elements. Pinned by three
     independent implementations: MLX `fused_gdn.py:249/:903` (`mx.zeros((B,num_v_heads,head_v_dim,head_k_dim))`);
     llama.cpp `qwen35.cpp:315`/`qwen3next.cpp:390` (`ggml_reshape_4d(state, head_v_dim, head_v_dim,
     num_v_heads, n_seqs)` — uses `head_v_dim` twice because `head_k_dim==head_v_dim==128`); vLLM
     `selective_state_snapshot.py:124` comment `(num_slots, num_v_heads, head_v_dim, head_k_dim)`.
     Element count anchored by `llama_hparams::n_embd_s() = ssm_d_state·ssm_d_inner = 128·6144 =
     786,432` (`llama-hparams.cpp:169`), exactly `head_k_dim·head_v_dim·num_v_heads = 128·128·48`.
     GGUF hparam mapping (converter): `ssm_d_state=linear_key_head_dim=128`,
     `ssm_d_inner=linear_value_head_dim·linear_num_value_heads=128·48=6144`,
     `ssm_dt_rank=linear_num_value_heads=48`, `ssm_n_group=linear_num_key_heads=16`,
     `ssm_d_conv=linear_conv_kernel_dim=4`.
   - **Reconciliation of the two figures:**
     - modal `[48,128,128]` / ~77 MB is **CORRECT**: `(num_v_heads, head_v_dim, head_k_dim)`;
       BF16 ssm = 1.573 MB/layer × 48 = 75.5 MB (≈77); + conv ≈ 78 MB; FP32 ssm = 151 MB (matches
       modal's "150 MB at float32 vs 77 MB at BF16").
     - recurrent-rollback `(14,256,256)` / 84 MB has the **WRONG SHAPE** (wrong head count 14 vs 48
       and wrong per-head dims 256 vs 128). Its *total* (84 MB) lands near the true ~78 MB only
       because `14·256·256 = 917,504` is within ~17% of the true `786,432` — coincidence. The
       per-head `(d_k × d_v)` *structure* in TECHNIQUE.md §1 is right; only the three numbers are
       wrong. The rollback **mechanism** (snapshot/restore, sub-ms copy at 546 GB/s) is unaffected.
   - **Internal vLLM placeholder:** `deltanet_adjuster.py` defaults `linear_num_value_heads=32`
     (→ would give 50.3 MB) — this is a **placeholder default, not the real value**. The
     authoritative `selective_state_snapshot.py` (48), modal `[48,…]`, the ~77 MB figure, and the
     converter `ssm_dt_rank` mapping all confirm **48**.
   - **Known vs assumed:** the three logical dims (128/128/48), `num_k_heads=16`, `conv_kernel=4`,
     and derived `n_embd_s=786,432` are pinned in source. No raw Qwen3.5-27B `config.json` is present
     in-corpus to read `linear_*` directly, but four independent implementations agree, so treated
     as established. Captured in `qwen35-hybrid-architecture.md` → "DeltaNet recurrent state".

7. **MTP-head body GGUF naming: `blk.64.nextn.attn_q` (README) vs `blk.64.attn_q` (diff/loader). — RESOLVED (iter 2).**
   - `qwen-mtp-tensors/README.md` (HF→GGUF table) shows the MTP transformer body under a `nextn.`
     prefix (`blk.64.nextn.attn_norm`, `.nextn.attn_q`, `.nextn.ffn_norm`, …).
   - The actual converter remap + loader (diffs `01`/`02`) put the MTP **body** in the *plain*
     per-layer slots (`blk.64.attn_q`, `blk.64.ffn_norm`, `blk.64.attn_norm`, …) and only the four
     MTP-specific tensors carry the `nextn.` prefix (`nextn.eh_proj/enorm/hnorm/shared_head_norm`,
     + optional `nextn.embed_tokens/shared_head_head`). **Resolved:** the README's `nextn.`-prefixed
     body names are an early/idealized convention; the real on-disk names (used in
     `tensor-layout-hf-to-gguf.md`) are the plain attention/FFN slots. The converter literally does
     `name.replace("mtp.layers.0", f"model.layers.{n_base}")` for the body (no `nextn` inserted).

5. **`mtp.shared_head.head` presence.** The tensor-mapping README lists `model.mtp.shared_head.head.weight` as an HF tensor (then "uses main output.weight"), and the converter remaps `mtp.norm → shared_head.norm`. The extraction scripts and `mtp_head.py` only enumerate `mtp.norm` (15 tensors) with no `shared_head.head`. **Resolved:** for Qwen3.5-27B (`mtp_use_dedicated_embeddings: false`) there is no usable dedicated lm_head — the 15-tensor count is correct, and any `shared_head` namespacing collapses to the main `output.weight`. Noted in `mtp-head-anatomy.md`.

## Gaps / referenced-but-missing files
- **`qwen-mtp-tensors/docs/tensor-layout.md` does NOT exist.** The repo's README ("What's in
  this repo") references `docs/tensor-layout.md`, but the `docs/` folder contains only
  `index.html` (a rendered HTML page). The actual `tensor-layout.md` markdown content lives in
  `qwen-mtp-research/docs/tensor-layout.md` (and the duplicate `qwen-ops/research/findings/tensor-layout.md`).
  Treated those as the canonical text. No content lost, but the cross-repo reference is broken.
- Diff 02's commit message references the MLX reference file `~/mlx-fork/fused_gdn.py` (`MTPHead`)
  and `~/mlx-fork/mtp_weights.safetensors` — paths outside this corpus (the author's machine).
  The equivalent in-corpus file is `mlx-qwen-mtp/src/mtp_head.py` (used as the reference here).
- `qwen-mtp-research/docs/the-bug.md`, `hybrid-deltanet-notes.md`, `methodology.md` and the
  `scripts/` referenced in `qwen-mtp-research/README.md` are **not present** in the repo (only
  `tensor-layout.md`, `integration-plan.md`, `mlx-reference.md`, `per-position-heads.md`,
  `the-recipe.md` exist under `docs/`). Those are Agent B/E concerns; flagging the missing files
  for the cross-domain ledger.
- **`qwen-mtp-tensors/docs/index.html`** is a rendered GitHub-Pages landing page. Its
  architecture-relevant content (layer table, MTP head intro, HF→GGUF map) is fully captured in
  the arch docs. Its *unique* content is **Agent B (LLAMACPP) territory**: the 11+5 patch series
  catalog, "patch 11 rollback bookkeeping" one-line fix, baseline numbers (plain decode 17.90
  tok/s; K=1 MTP spec 7.64 tok/s = 0.43×), and the apply commands (`git am ... patches/*.patch`,
  upstream commit `9c7911f4f`, target `llama-mtp-speculative`). Flagged for the cross-domain ledger
  — no architecture fact lost.

## Losslessness verification (iter 2)

Every architecture-relevant fact in the listed sources, with where it lives in `MTP/` and status.
`✓` = captured; `dup` = byte/semantic duplicate already collapsed; `xdom` = belongs to another
agent's domain (only the arch sliver taken here).

### `qwen-mtp-tensors/` (every file)

| Source file | Architectural fact(s) | Captured where | Status |
|---|---|---|---|
| `README.md` | layer table (48/16/1, 3:1); HF→GGUF map; eh_proj concat; the bug + 5 fixes; 3 discoveries; `mtp_use_dedicated_embeddings:false`; `[12,12,12,12]`; `attn_post_norm`↔`ffn_norm`; `nextn.`-prefixed body names | `tensor-layout-hf-to-gguf.md`, `mtp-head-anatomy.md`, `qwen35-hybrid-architecture.md` | ✓ (body-name + rope discrepancies resolved in ledger #4,#7) |
| `diffs/01-qwen35-tensor-load.diff` | converter `__init__` block_count bump + tensor_map rebuild; `add_ssm_*`/`add_nextn_predict_layers`/`add_rope_dimension_count`; `mtp.*` remap; `.A_log → -exp`, `.dt_bias`; constants.py FFN_NORM+6 NEXTN; arch.cpp ATTN_POST_NORM→FFN_NORM + reclassify LAYER_OUTPUT→LAYER_REPEATING; model.cpp recurrent mask `(i<n_base)&&((i+1)%4!=0)`, type switch `n_layer-nextn==64→27B`, nextn alloc `{2*n_embd,n_embd}` etc., FFN_NORM post-attn-norm load | `tensor-layout-hf-to-gguf.md` (5 fixes table + 3 discoveries + DeltaNet tensor map + A_log transform) | ✓ + copied to `tensor-diffs/01` |
| `diffs/02-qwen35-graph-tensors.diff` | `build_mtp_head` graph; `LLM_GRAPH_TYPE_MTP` early-return; `ggml_concat(e_norm,h_norm,0)` embed-first; gated Q `*2` split; `ggml_rope_ext`+len-1 (earlier) vs `ggml_rope_multi`+len-4 (resolved); `il_mtp=n_layer-nextn`; `t_mtp_prev_hidden_in`/`t_mtp_pos_zero_in`; `n_embd_head_v==n_embd_head_k` assert; shared-head fallback | `mtp-head-anatomy.md` (forward + graph notes), `tensor-layout-hf-to-gguf.md` (discovery #2) | ✓ + copied to `tensor-diffs/02` |
| `docs/index.html` | (rendered landing page) layer table + MTP intro = arch; patch series / baselines / apply cmds = Agent B | arch part: arch docs; rest: flagged | xdom (Agent B) — see Gaps |
| `LICENSE` | MIT | — | n/a (license only) |

### llama.cpp fork — architecture-relevant symbols

| Source (file:lines) | Architectural fact | Captured where | Status |
|---|---|---|---|
| `llama-arch.h:44-45` | `LLM_ARCH_QWEN35`, `LLM_ARCH_QWEN35MOE` enums | `tensor-layout-hf-to-gguf.md`, `qwen35-hybrid-architecture.md` | ✓ |
| `llama-arch.h:190,207` | `LLM_KV_NEXTN_PREDICT_LAYERS`, `LLM_KV_FULL_ATTENTION_INTERVAL` | `tensor-layout-hf-to-gguf.md`, `qwen35-hybrid-architecture.md` | ✓ |
| `llama-arch.h:531-536` | 6 `LLM_TENSOR_NEXTN_*` enums | `mtp-head-anatomy.md`, `tensor-layout-hf-to-gguf.md` | ✓ |
| `llama-arch.cpp:186,203` | KV templates `%s.nextn_predict_layers`, `%s.full_attention_interval` | `tensor-layout-hf-to-gguf.md` | ✓ |
| `llama-arch.cpp:427-432` | `blk.%d.nextn.{eh_proj,embed_tokens,enorm,hnorm,shared_head_head,shared_head_norm}` templates | `tensor-layout-hf-to-gguf.md` | ✓ |
| `llama-arch.cpp:1010-1041` | QWEN35 tensor list incl. `SSM_{A_NOSCAN,CONV1D,DT,BETA,ALPHA,NORM,OUT}` + 6 NEXTN | `tensor-layout-hf-to-gguf.md` (DeltaNet per-layer tensors table — added iter 2) | ✓ |
| `llama-arch.cpp` LLM_TENSOR_INFOS | NEXTN reclass LAYER_OUTPUT→LAYER_REPEATING (from diff 01) | `tensor-layout-hf-to-gguf.md` (Fix 3) | ✓ |
| `llama-hparams.h:48-49` | `n_embd_head_k`/`n_embd_head_v` (=256, attention head_dim; ≠ DeltaNet 128) | `qwen35-hybrid-architecture.md` (head_dim 256) | ✓ |
| `llama-hparams.h:85` | `nextn_predict_layers` field (default 0) | `tensor-layout-hf-to-gguf.md` (Fix 5), `qwen35-hybrid-architecture.md` | ✓ |
| `llama-hparams.h:120` | `std::array<int,4> rope_sections` | `qwen35-hybrid-architecture.md` (MRoPE) | ✓ |
| `llama-hparams.h:133-137` | `ssm_d_conv/d_inner/d_state/dt_rank/n_group` fields | `qwen35-hybrid-architecture.md` (DeltaNet dims table) | ✓ |
| `llama-hparams.h:143,278` | `recurrent_layer_arr`, `is_recurrent(il)` | `qwen35-hybrid-architecture.md` (layer table note), `tensor-layout-hf-to-gguf.md` | ✓ |
| `llama-hparams.cpp:131-169` | `n_embd_r()` (conv state size) + `n_embd_s()=ssm_d_state·ssm_d_inner` (=786,432) | `qwen35-hybrid-architecture.md` ("DeltaNet recurrent state") | ✓ (added iter 2) |
| `llama-model.cpp:209-230` | read `nextn_predict_layers`; recurrent mask `(i<n_base)&&…`; type switch `…==64→27B` | `tensor-layout-hf-to-gguf.md` (Fix 3/4), `qwen35-hybrid-architecture.md` | ✓ |
| `llama-model.cpp:619,1034` | `n_rot` from `ROPE_DIMENSION_COUNT` (=64); `rope_sections` from `ROPE_DIMENSION_SECTIONS` | `qwen35-hybrid-architecture.md` (MRoPE table, `n_rot=64`) | ✓ (added iter 2) |
| `llama-model.cpp:1231-1235` etc. | SSM hparam load (`ssm_d_conv/inner/state/dt_rank/n_group`) | `qwen35-hybrid-architecture.md` (GGUF-name column) | ✓ |
| `llama-model.cpp:7446-7548` | QWEN35 loader: attn_post_norm←FFN_NORM; nextn alloc block (`eh_proj {2*n_embd,n_embd}`, required vs `TENSOR_NOT_REQUIRED`) | `tensor-layout-hf-to-gguf.md` (Fix 4 + on-disk layout) | ✓ |
| `models/qwen35.cpp:36-55` | `LLM_GRAPH_TYPE_MTP` early return, `il_mtp`, exports | `mtp-head-anatomy.md` (graph notes) | ✓ |
| `models/qwen35.cpp:239-467` | GDN layer: `ssm_d_state`/`ssm_dt_rank`/`ssm_n_group` → head dims; state `reshape_4d(head_v_dim,head_v_dim,num_v_heads,n_seqs)`; conv_channels=10240; `build_qkvz`,`ssm_{beta,alpha,dt,a,conv1d,norm,out}` | `qwen35-hybrid-architecture.md` ("DeltaNet recurrent state" + dims), `tensor-layout-hf-to-gguf.md` (DeltaNet tensors) | ✓ |
| `models/qwen35.cpp:556-595` | resolved MTP MRoPE: `inp_pos_zero` len-4, `ggml_rope_multi(n_rot, rope_sections)`, `build_attn` | `tensor-layout-hf-to-gguf.md` (discovery #2), `mtp-head-anatomy.md` | ✓ |
| `models/qwen3next.cpp:209-418` | parallel GDN (same `ssm_*`→dims, same state `reshape_4d`) — confirms shared shape | `qwen35-hybrid-architecture.md` ("DeltaNet recurrent state" cross-check) | ✓ |

### MLX / vLLM / modal / recurrent-rollback — arch slivers

| Source | Architectural fact | Captured where | Status |
|---|---|---|---|
| `MTP/03-MLX/src/fused_gdn.py:249,903` | runtime state alloc `(B,num_v_heads,head_v_dim,head_k_dim)`; conv `(B,conv_kernel-1,conv_dim)` | `qwen35-hybrid-architecture.md` ("DeltaNet recurrent state") | ✓ (authoritative) |
| `modal-mtp/README.md` | `[48,128,128]`/layer, 77 MB BF16 / 150 MB FP32; 3:1 pattern | `qwen35-hybrid-architecture.md` | ✓ (reconciled, ledger #6) |
| `qwen-ops/vllm/optimizations/selective_state_snapshot.py` | authoritative ssm/conv shapes; `linear_num_value_heads=48`, head_dim 128, conv_kernel 4 | `qwen35-hybrid-architecture.md` | ✓ |
| `qwen-ops/vllm/optimizations/deltanet_adjuster.py` | `linear_num_key_heads=16`; placeholder `num_value_heads=32` | `qwen35-hybrid-architecture.md` (placeholder note) + ledger #6 | ✓ |
| `recurrent-rollback/README.md` + `docs/TECHNIQUE.md` | rollback mechanism; cited `(14,256,256)`/84 MB (WRONG shape) | `qwen35-hybrid-architecture.md` (corrected) + ledger #6 | ✓ (corrected) |

**Result: no architecture fact found uncaptured.** Two source-content errors were *corrected in
place* (recurrent-rollback state shape; README `[12,12,12,12]`/`nextn.`-body-names), with the
journey preserved in the ledger. New material added iter 2: DeltaNet recurrent-state section +
GDN dims table + DeltaNet per-layer tensor map + `A_log` transform + `n_rot=64` resolution +
`n_embd_s`/`n_embd_r` formulas.
