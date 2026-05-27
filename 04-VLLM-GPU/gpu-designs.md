# GPU / vLLM Design Docs (Distilled)

Navigable distillation of the 9 GPU/vLLM design documents from
`qwen-ops/research/designs/`. Each entry: the idea, the status, the key finding.
(`modal-mtp-design.md` is byte-identical to `modal-mtp/docs/design.md` and is
collapsed into [`modal-self-speculative.md`](modal-self-speculative.md);
`CASCADE-MTP-TRAINING.md` and `per-position-heads.md` belong to Agent E.)

| # | Doc | One-line verdict |
|---|---|---|
| 1 | GH200 compute/bandwidth | Workload is **overhead-bound**, not compute- or bandwidth-bound; spec decode *hurts* single-request on GH200. **NB: this doc's arch table (hidden 3584, ~16B) is NOT the 27B model — see callout.** |
| 2 | CUDA-graph weight swap | In-place `copy_()` weight swap **works** under CUDA-graph replay (pointers, not values). |
| 3 | Two-graph CUDA dispatch | Capture partial+full graphs; branch on CPU. Break-even at 12.5% early-exit hit rate. |
| 4 | propose_tree CUDA-graph compat | `propose_tree()` **is** graph-compatible; the config path (enforce_eager gate) blocks it. |
| 5 | MTP + TREE_ATTN config path | `speculative_token_tree` **survives** the MTP flow; the "Unsupported method" error is a different path. |
| 6 | PLV full-attn boundaries | Early exit needs **87–94% of layers**; aggressive layer-skip verification ruled out. |
| 7 | DeltaNet weight transplant | Drop dedicated draft weights into the 48 DeltaNet slots; arch unchanged. Result -6% (noise). |
| 8 | EAGLE PR descriptions | 3 upstream-ready bug fixes for MRoPE + tree attention in `propose_tree`. |
| 9 | modal-mtp design | (collapsed → `modal-self-speculative.md`) |

---

## 1. GH200 Compute vs Bandwidth Analysis

**Idea:** Roofline-analyze Qwen3.5-27B W4A16 decode on GH200 to find the
bottleneck. **Status:** measured, 2026-04-11.

**Key findings:**
- The "compute-bound" hypothesis is **wrong**. Operational intensity
  `OI = 30.92 GFLOPs / 8.0 GB = 3.9 FLOPs/byte` vs machine balance
  `1070 TFLOPS / 4.0 TB/s = 267.5` → **deeply bandwidth-bound** (would need
  batch≈69 to become compute-bound).
- But measured **94 tok/s** (batch=1 baseline) is only **19% of the
  bandwidth-bound ceiling (~500 tok/s)** and 0.4% of compute-bound (24,236).
- Telemetry: **27% GPU compute util, 13% memory-bandwidth util, 216 W / 700 W,**
  SM clock 1980 MHz (max). The GPU **idles between kernel launches**.
- Real bottleneck = **overhead-bound**: ~640 kernel launches/token (64 layers ×
  ~10 kernels), piecewise CUDA-graph gaps, skinny GEMMs at batch=1, framework
  overhead.
- **Batch scaling is near-linear (baseline):** batch=1/2/4/8 →
  90.6 / 106.0 / 263.4 / **592.8** agg tok/s (8× batch → 6.5× throughput,
  per-request only drops 90.7→82.0) — confirms bandwidth headroom.
- **Spec decode is SLOWER single-request:** baseline 94 → MTP-tree (9 tokens)
  **36 tok/s (-62%)**. Per-token decode 10.6 ms → 28.0 ms; TTFT 37 ms → 72 ms.
  Linear fits: baseline `36.6 ms + 10.619 ms·t`, spec `71.5 ms + 28.021 ms·t`.
  Drafter overhead exceeds savings because single-token decode is already fast on
  4.8 TB/s.
- **Spec decode hurts at every tested batch too:** spec batch=1/2/4/8 agg =
  36.7 / 26.2 / 87.0 / **225.5** tok/s — all below the corresponding baseline
  batch numbers. (So even batching does not make single-model spec decode win
  here; the win comes from concurrency on the *non-spec* path.)
- **Telemetry @ baseline batch=1:** 27% GPU compute util, 13% mem-bandwidth util
  (reading weights at ~0.6 TB/s, not 4.0), 216 W / 700 W (31%), SM 1980 MHz (max),
  mem clock 2619 MHz (max). ~640 kernel launches/token (64 layers × ~10) ≈ 3.2 ms
  of pure launch overhead (~30% of the 10.6 ms/token).

> **⚠️ Contradiction with the rest of the corpus — and it is measuring a
> DIFFERENT MODEL.** This doc reports **baseline 94 tok/s / spec 36 tok/s** while
> the benchmark docs and deploy runbooks report **186 tok/s (spec=7, batch=1)** /
> 1030 (batch=8). The root cause: **the doc's own "Model Architecture" table is
> not Qwen3.5-27B.** It lists **hidden 3584, 28 attn heads (4 KV), head_dim 128,
> MLP intermediate 18944, vocab 151936, ~16B "effective"** and 64 layers. The
> corpus headline Qwen3.5-27B is **hidden 5120, 24 heads (4 KV), head_dim 256,
> intermediate 17408, vocab 248320** (`early_verify_probe.py` hardcodes
> `HIDDEN_DIM=5120, VOCAB_SIZE=248320`; the arch doc gives 5120/256/17408).
>
> **Proof it is a different arch (FLOP math is self-consistent with 3584, not
> 5120):** the doc's per-layer 0.466 GFLOPs (attn 0.059 + MLP 0.407), ×64 =
> 29.83, + LM-head 1.09 = **total 30.92 GFLOPs/token** reproduce *exactly* from
> the 3584-hidden / 18944-intermediate / 151936-vocab profile. Recomputed against
> the real 27B arch (5120 / 17408 / 248320) the total is **46.16 GFLOPs/token**
> (per-layer 0.682 ×64 = 43.62 + LM-head 2.54), ~1.5× larger. Likewise the doc's
> **W4A16 weight transfer = 8.0 GB** implies ~16B params at 0.5 B/param, not the
> 27B model's ~13.5 GB (corpus: 13.7 GB 4-bit weights). So the "Qwen 3.5-27B
> W4A16" title is a **mislabel**: the roofline/telemetry was run against a
> ~16B-class model (3584 hidden — a Qwen3-14B-shaped profile), and the 70B/
> DeepSeek-V3-R1 references in the sibling TWO-GRAPH doc confirm these GH200 design
> docs were written across multiple model sizes.
>
> The doc itself separately flags the throughput gap: it argues the "186 tok/s
> with spec decode" figure was likely **aggregate throughput under concurrent
> load**, or counted speculated (not accepted) tokens, or a different config.
> **Resolution for this corpus:** treat 186/1030 as the production
> aggregate/batched numbers for the real 27B model (what the benchmark + deploy
> docs measured), and 94/36 as a *single-request measurement on a different,
> smaller model arch* that nonetheless establishes the qualitative point —
> **on GH200 the single-request workload is overhead-bound and single-request spec
> decode can hurt.** Both are kept; neither is "rounded away". The 27B↔16B arch
> mismatch is the deepest root of the 186-vs-94 gap, on top of single-vs-batch.

Recommendation in the doc: disable spec decode for single-request on GH200,
optimize for batching, pursue full CUDA graphs, profile with nsys.

## 2. CUDA-Graph Weight Swap

**Idea:** `NativeMultiHeadProposer` swaps MTP-head weights between K siblings and
replays a captured CUDA graph each time — is that correct? **Status:**
feasible, empirically verified.

**Key finding:** **Yes.** CUDA graphs capture kernel launches and memory
**pointers**, not **values**. In-place writes to the same pointer are visible on
replay. Verified with raw `torch.cuda.CUDAGraph` and `torch.compile`
(reduce-overhead/inductor). vLLM's `CUDAGraphWrapper` only validates *input*
(activation) addresses, never weight addresses — weights are assumed at static
pointers.

**Hard constraint — pointer stability:**
- Safe: `w.copy_(new)`, `w.data.copy_(new)`, `w[:] = new`.
- Breaks the graph (reallocates): `w = new`, `w.data = new`, `del w; w = ...`.

Cost: one device-to-device `copy_` per head per propose (~0.1–0.3 ms for a
~600 MB head). No re-capture, no torch.compile invalidation.

## 3. Two-Graph CUDA Dispatch for Partial-Layer Verification

**Idea:** CUDA graphs can't contain conditionals, but PLV needs "if early-exit
agrees, skip layers N+1..63". Capture **two** graphs (`graph_partial` N layers,
`graph_full` 64) and put the branch on the **CPU** between two graph replays.
**Status:** design proposal (GH200).

**Key findings:**
- Flow: replay `graph_partial` → D2H copy argmax → CPU compare vs draft tokens →
  if all match accept (skip full), else replay `graph_full`.
- D2H sync on GH200 (NVLink-C2C 900 GB/s, 40-byte payload): **~5–10 µs**;
  discrete PCIe 5.0: ~15–25 µs. Negligible vs a layer (~500–800 µs on 70B).
- Memory: shared graph pool means `graph_partial` mostly reuses `graph_full`'s
  pool → realistic overhead **~200–500 MB**, not 2×.
- **Break-even early-exit hit rate = 12.5%** (`P·4.01 + (1−P)·36.01 = 32`).
  Empirical PLV hit rates 40–70% → strongly favorable *in theory*.
- KV-cache hazard (HIGH): if `graph_full` runs after `graph_partial`, it must not
  recompute the KV already written for layers 0..N-1 — simplest is to re-run all
  layers (wastes ~12.5%).
- Follows the existing `modal_mtp` precedent (separate graph set,
  `draft_mode=True` BatchDescriptor). Integration points mapped to specific
  vLLM line numbers (`cudagraph_dispatcher.py`, `cuda_graph.py`,
  `gpu_model_runner.py`). Alternatives compared: eager mode (simpler, loses
  replay speedup), soft probe (no savings), CUDA 12.4 conditional nodes (no
  PyTorch support today).

> This is the design; the **empirical** PLV result (doc #6) shows hit rates are
> *not* favorable on this model — early exit needs 87–94% of layers — so the
> two-graph mechanism is sound but PLV itself doesn't pay off here.

## 4. propose_tree() CUDA-Graph Compatibility Analysis

**Idea:** Is the 22-vs-186 tok/s gap caused by `propose_tree()` being
graph-incompatible? **Status:** analyzed (eagle.py lines 1033–1221).

**Key finding:** **propose_tree() IS CUDA-graph compatible.** The
`for level in range(tree_depth-1)` loop is fixed at init; `torch.where`,
`topk`, `cat`, `argmax`, `masked_fill_` are graph-safe; the `if` guards are
Python-level on init-time constants. `build_for_drafting()` runs *outside* the
graph boundary (pure metadata). `dispatch()` pads variable `num_tokens` to
capture sizes. `TreeAttentionImpl.forward()` uses a Triton kernel with a static
bias tensor. **What blocks it is the config path** — when tree attention is
selected, config validation forces `enforce_eager=True`, so
`eagle_cudagraph_mode=NONE`. **Fix is in config validation, not propose_tree.**

## 5. MTP + TREE_ATTN Config Path

**Idea:** Does `speculative_token_tree` survive `SpeculativeConfig.__post_init__`
for `method="mtp"`? **Status:** analyzed (speculative.py).

**Key finding:** **The hypothesis is wrong — the tree IS preserved.** For
self-speculative MTP (`method="mtp"`, `model=None`): the config sets
`model=target`, converts `qwen3_5 → qwen3_5_mtp` via `hf_config_override`,
auto-detects MTP, and at the tree step **preserves and sorts** a user tree
(breadth-first) instead of overwriting it. `use_eagle()` returns True → creates
`EagleProposer` → `TreeAttentionBackend` reads the tree and builds the bias.

The **"Unsupported speculative method: 'mtp'"** error fires on a *different*
path: passing `method="mtp"` with a **standalone** draft model (model ≠ target).
Then `_skip_mtp_conversion=True` blocks the `qwen3_5 → qwen3_5_mtp` conversion,
`"qwen3_5"` isn't in `MTPModelTypes`, and it falls through to the error. Fix is
in auto-detection / using the right model+method combo, not in tree handling.

Difference table: `mtp` respects+sorts user trees (EagleProposer); `modal_mtp`
always auto-generates a linear chain and **ignores** user trees
(ModalMTPProposer). Both read the tree via TREE_ATTN.

## 6. PLV at Full-Attention Layer Boundaries

**Idea:** Since every 4th layer is full-attention (the global-context
checkpoints), exiting *there* should give better next-token agreement than
exiting at DeltaNet layers. **Status:** measured (bf16 CPU, two sweeps).

**Key findings — the hypothesis fails:**
- For layers 4–32, p_agree ≈ **10% (random, 1/vocab-ish)** at **both**
  full-attention and DeltaNet exit points. Full-attn boundaries give **no
  advantage** in the first half.
- Transition is **gradual and late**: first signal above noise at layer 40
  (20% p_agree); usable only at the very end:
  - **Layer 56** (87.5% depth): 30% top-1, 57% top-5.
  - **Layer 60** (93.8%): 53% top-1, 83% top-5.
  - Layer 48 (75%): 13% / 17% — no better than random.
- `next_layer_input_layernorm` before final norm **does not help** (sometimes
  hurts) — intermediate reps are in a different subspace, not just mis-normalized.
- KL divergence decreases monotonically 6.7 (L2) → 3.0 (L60) even while argmax is
  random — the model builds its prediction incrementally across all layers.
- A 52M-param regression adapter (h_L→h_63) lifts p_agree (1.6%→25% @L3,
  15.8%→36.8% @L47) but **plateaus far below usable**.

**Conclusion:** "run half the layers to verify" is **ruled out** — the last 16
layers (48–64) do critical work. Next ideas floated: layer-skip (not early-exit)
verification; use the DeltaNet-only draft model as a fast verifier; use layer-56
top-5 as a cheap reject pre-filter (~43% false-positive).

## 7. DeltaNet Weight Transplant

**Idea:** modal_mtp drafts with the main model's DeltaNet subnetwork, which was
never optimized standalone. Transplant a **dedicated 48-layer DeltaNet draft
model's** weights into the main model's 48 DeltaNet slots — better drafts, zero
arch change. **Status:** designed + scripted (`deltanet_transplant.py`).

**Key findings:**
- **Layer mapping:** main DeltaNet indices are non-contiguous (gap every 4th for
  the attention layer). Formula: **`main_idx = draft_idx + (draft_idx // 3)`**
  (draft 0→main 0, draft 3→main 4, … draft 47→main 62).
- **All DeltaNet tensor shapes match** between draft and main (e.g.
  `in_proj_qkv [10240,5120]`, `out_proj [5120,6144]`, `conv1d [10240,1,4]`,
  `A_log [48]`, hidden 5120, intermediate 17408).
- **norm.weight gap:** 12 draft layers (every 4th: 3,7,11,…,47) lack
  `linear_attn.norm.weight`; the script skips those, leaving the main model's
  norm in place — 36 clean transplants, 12 with minor distribution mismatch.
- Transplanted: `linear_attn.*`, MLP, layernorms. **Not** transplanted:
  embed_tokens, lm_head, final norm, full-attention layers, MTP head, vision.
  MTP head optionally copied.
- Reversible (reload main), no shape changes, no config changes. During draft
  mode the attention layers are identity, so the draft weights run in exactly the
  environment they were trained for.
- **Result: 174 tok/s (-6%, within noise)** — not a win in practice.

## 8. EAGLE PR Descriptions (upstream bug fixes)

**Idea:** Three upstream-ready vLLM PRs fixing `propose_tree`
(`v1/spec_decode/eagle.py`, lines ~1033–1221) for **MRoPE + tree-attention**
models — repro target **Qwen3.5-MoE-A3B** (any `uses_mrope=True` model with
EAGLE/MTP spec decode + tree attention; `self.positions` is `(3, N)` not `(N,)`).
**Status:** written, not found as existing issues/PRs (7 `gh search` queries, all
0 results, as of 2026-04-11). These ship in `eagle.patch`.

- **PR 1 — position accessors:** `propose_tree` accesses `self.positions`
  directly in 3 places; for MRoPE `self.positions` is `(3, N)`. Use
  `_get_positions()`/`_set_positions()`. Error before fix:
  `RuntimeError: shape mismatch: value tensor of shape [N] cannot be broadcast
  to indexing result of shape [3, N]`.
- **PR 2 — `model_returns_tuple()` unpacking:** `last_hidden, hidden =
  self.model(...)` hardcodes a 2-tuple; some draft heads return a single tensor.
  Error: `ValueError: not enough values to unpack (expected 2, got 1)`. Guard
  with `model_returns_tuple()`.
- **PR 3 — 1D position arithmetic:** scalar arithmetic on `(3, N)` positions
  (`positions.view(batch_size,-1)`, `positions + level`) breaks. Extract
  `positions_1d = positions[0]` for MRoPE, expand back to `(3,N)` on write-back.
  Error: `RuntimeError: shape '[-1, 1]' is invalid for input of size 6`.

Suggested commits:
```
fix(spec_decode): use position accessors in propose_tree for MRoPE compat
fix(spec_decode): use model_returns_tuple() in propose_tree forward call
fix(spec_decode): extract 1D positions for tree draft arithmetic on MRoPE models
```

## Related

- [`modal-self-speculative.md`](modal-self-speculative.md) — design #9 (modal-mtp) collapsed here; the draft-mode mechanism these designs build around.
- [`vllm-patches-and-strategies.md`](vllm-patches-and-strategies.md) — the patches/strategy modules that implement these designs (eagle PRs, weight-swap, transplant, PLV).
- [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) — the real 27B dimensions (5120/256/17408/248320) that expose the design-#1 roofline mislabel.
- [`README.md`](README.md) — the production results table that contextualizes "none beat baseline".
- [`../99-LEDGER/open-questions.md`](../99-LEDGER/open-questions.md) — the GH200 roofline-arch mismatch (16B-class vs 27B) as an open ledger item.
- [`../00-OVERVIEW/glossary.md`](../00-OVERVIEW/glossary.md) — PLV, CUDA graph, roofline, MRoPE.

---
*Provenance: `qwen-ops/research/designs/{GH200-COMPUTE-BANDWIDTH-ANALYSIS,
CUDAGRAPH-WEIGHT-SWAP,TWO-GRAPH-CUDA-DISPATCH,PROPOSE-TREE-CUDAGRAPH-ANALYSIS,
MTP-TREE-CONFIG-PATH,PLV-FULL-ATTN-BOUNDARIES,DELTANET-WEIGHT-TRANSPLANT,
EAGLE-PR-DESCRIPTIONS}.md`. `modal-mtp-design.md` collapsed into
`modal-self-speculative.md`.*
