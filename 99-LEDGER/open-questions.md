# Open Questions & Cross-Domain Contradiction Registry

**Owner: Agent 3C (99-LEDGER). Authoritative cross-domain record.** This is where every
contradiction, evolving thread, and unresolved question that spanned more than one source
(or that changed over time) is reconciled to a single answer — with the canonical doc that
now holds it. Per-domain inventories raised these; this file adjudicates and indexes them.

Each entry: the contradiction → who disagreed → the resolution (or why still open) → the
canonical doc. Status is **RESOLVED** or **OPEN**. Numbers are preserved verbatim; nothing
is rounded or flattened.

Summary: **13 RESOLVED, 3 OPEN** (16 tracked items).

---

## RESOLVED

### R1 — DeltaNet recurrent-state shape: `[48,128,128]`+conv ✓ vs `(14,256,256)` ✗

| | |
|---|---|
| **Contradiction** | Per-layer GatedDeltaNet recurrent state shape & size. |
| **Sources that disagreed** | `modal-mtp/README.md`: `[48,128,128]`/layer, ~77 MB BF16 (150 MB FP32). `recurrent-rollback/README.md` + `docs/TECHNIQUE.md`: `(n_heads=14, d_k=256, d_v=256)`, 1.75 MB/layer × 48 = 84 MB. |
| **Resolution** | **modal `[48,128,128]` is CORRECT.** Authoritative ssm state = `(num_v_heads=48, head_v_dim=128, head_k_dim=128)` = **786,432 elements**, **plus** conv state `(conv_kernel-1=3, conv_channels=10240)` = 30,720 elements. Pinned by 4 independent impls: MLX `fused_gdn.py:249/:903`, llama.cpp `qwen35.cpp:315`/`qwen3next.cpp:390` (`reshape_4d`, uses `head_v_dim` twice since `head_k_dim==head_v_dim==128`), vLLM `selective_state_snapshot.py:124`, and `n_embd_s()=ssm_d_state·ssm_d_inner=128·6144=786,432` (`llama-hparams.cpp:169`). BF16 ssm = 1.573 MB/layer × 48 = 75.5 MB (≈77); + conv ≈ 78 MB; FP32 = 151 MB (matches modal's 150). The rollback repo README/TECHNIQUE figure `(14,256,256)` has the **WRONG shape** (head count 14 vs 48; per-head dims 256 vs 128); its 84 MB total lands near the true ~78 MB only because `14·256·256=917,504` is within ~17% of `786,432` — coincidence. The rollback **mechanism** (snapshot/restore, sub-ms copy) is unaffected. vLLM `deltanet_adjuster.py` default `linear_num_value_heads=32` is a **placeholder**, not the real 48. |
| **Canonical doc** | `01-ARCHITECTURE/qwen35-hybrid-architecture.md` → "DeltaNet recurrent state" (shape corrected in place; journey preserved in inventory-01 thread #6). |

### R2 — Precision-divergence saga (FP32 100% → "precision problem" → SUPERSEDED → CPU-vs-GPU kernel numerics)

| | |
|---|---|
| **Contradiction** | `modal-mtp/README.md` contradicts *itself* across sequential sections on why FP16/GPU draft accuracy collapses. |
| **Sources that disagreed** | Same doc, three stages: (1) FP32/CPU 100% (5/5) → "architecture is sound"; (2) FP16/GPU ~15% (12/12) → "FP16 Accuracy Update" concludes *"it's a precision problem"* + frames `--mamba-ssm-cache-dtype float32` as a *precision* fix; (3) "Critical Finding: CPU vs GPU Divergence" — CPU FP16 = 100% (1146/1146 tokens, per `validate_extended_output.log`) while GPU FP16 = ~15% ⇒ same precision, different device ⇒ NOT precision. |
| **Resolution** | **It is CUDA fused-kernel numerical behavior** (`causal_conv1d`, `chunk_gated_delta_rule`), not precision. Architecture proven sound at all precisions on CPU. Test matrix: CPU-FP32 100% (5/5), CPU-FP16 100% (12/12), GPU-FP16 ~15%, GPU-BF16 ~50% (4/8). The stage-2 "precision problem" framing is preserved but explicitly labeled **SUPERSEDED**. The FP32-state fix is kept but re-labeled: may still help (FP32 damps kernel noise) but the stage-2 *reason* was the wrong diagnosis; remains **untested in vLLM**. |
| **Canonical doc** | `04-VLLM-GPU/precision-divergence.md` (resolved picture up top, 4-row matrix, SUPERSEDED dead-end preserved). |

### R3 — "MLX uses trained per-position heads" → single head + chained recurrence + gating + 0.8B companion

| | |
|---|---|
| **Contradiction** | What produced the MLX 1.68×/1.73× speedup? |
| **Sources that disagreed** | Early framing (`qwen-mtp-research/docs/per-position-heads.md` and elsewhere): trained per-position MTP heads (DeepSeek-V3 style). Resolved truth (`mlx-reference.md`, confirmed vs `mtp_weights_vanilla.safetensors`): the shipped checkpoint has **exactly ONE** MTP block (`mtp.layers.0.*`; no `layers.1+`). |
| **Resolution** | The speedup is a **$0-training runtime strategy** from `stacked_v2.py`: (1) chained recurrent application of the single head (feed its own output hidden as next `prev_hidden`), (2) a ~0.8B companion draft model for position 3, (3) confidence gating (`max(softmax) ≥ threshold`). Same recipe (components 1+3) gave llama.cpp **1.99× over K=1 vanilla** (`MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`). Per-position heads remain a valid *alternative* training-based path, gated by Phase-0. |
| **Canonical doc** | `05-THEORY-AND-DESIGNS/per-position-heads-design.md` §0 (correction stated *before* the design). Cross-confirmed in `03-MLX/the-journey.md` (Phase 7) + `02-LLAMACPP` inventory thread #1. |

### R4 — GH200 186 vs 94 tok/s

| | |
|---|---|
| **Contradiction** | GH200 single-request throughput: 186 tok/s (benchmarks + deploy docs) vs 94 tok/s baseline / 36 tok/s spec (`GH200-COMPUTE-BANDWIDTH-ANALYSIS.md`, which explicitly disputes 186). |
| **Sources that disagreed** | `GH200-COMPUTE-BANDWIDTH-ANALYSIS.md` (94/36, single-request) vs `vllm-patches-benchmarks.md` + deploy docs (186 single / 1030 batch=8). |
| **Resolution** | **Two causes.** (a) **The GH200 doc is mislabeled** — its FLOP table reproduces *exactly* from a ~16B-class arch (hidden 3584, 28 heads, head_dim 128, intermediate 18944, vocab 151936, 64 layers → 30.92 GFLOPs/tok; W4A16 transfer 8.0 GB ≈ 16B params), **not** the real Qwen3.5-27B (hidden 5120, 24 heads, head_dim 256, intermediate 17408, vocab 248320 → 46.16 GFLOPs/tok ≈1.5×, ~13.5 GB). 248320 is the real vocab (live code `early_verify_probe.py VOCAB_SIZE=248320`); the 3584/18944 arch appears in no other source. (b) **Single-request vs batch** — 186/1030 are production aggregate/batched on the real 27B; 94/36 are single-request on the smaller arch. Both kept verbatim, neither rounded. Single-request spec decode *hurts* (spec b1/2/4/8 = 36.7/26.2/87.0/225.5 vs baseline 90.6/106.0/263.4/592.8) — GH200 single-request is overhead-bound. |
| **Canonical doc** | `04-VLLM-GPU/gpu-designs.md` #1 (expanded contradiction callout with FLOP proof). |

### R5 — rms_norm + matmul fusion: 2.0 ms micro-win vs +0.76 ms slower on full model

| | |
|---|---|
| **Contradiction** | Does fusing rms_norm into the matmul save time? |
| **Sources that disagreed** | `DISPATCH_BARRIER_PROFILING.md` + `mlx-qwen-mtp/README.md`: 2.0 ms measured saving (256-matmul microbenchmark via `mx.fast.metal_kernel`), 10.4 ms (30%) projection with full metallib. `RESEARCH_FRONTIERS.md`: "NOT pursuing" — real model showed **+0.76 ms slower** because `mx.compile` already handles dispatch. |
| **Resolution** | Real win **only** with full metallib integration replacing the stock kernel. The `mx.fast.metal_kernel` prototype wins the microbenchmark but loses to `mx.compile`'s scheduling on the full model. **Not in the shipped path.** |
| **Canonical doc** | `03-MLX/kernel-fusion.md`. |

### R6 — Entropy-coding: 3.27× claim → 1.08× / 7% debunk

| | |
|---|---|
| **Contradiction** | Can entropy-coding the 4-bit weights cut bandwidth ~3×? |
| **Sources that disagreed** | `RESEARCH_FRONTIERS.md` first frames it as the transformative 3.3× change ("the big prize, 80–90 tok/s"; ECQ measured 3.27× → 140 tok/s on M3 Pro). Same doc, dated correction: measured on Qwen3.5-27B-4bit → **3.72 bits actual entropy** (not 1.1), **1.08× compression, 7% savings**. |
| **Resolution** | The 1.1-bit figure was a *different* quantization format with learned codebooks. rANS on standard 4-bit Qwen3.5-27B is not worth a custom kernel. Both the claim and the debunk are preserved; the combined-projection table is flagged as depending on the debunked number. The 3.27× ECQ number is a real measured M3-Pro result (different format) and is kept verbatim. |
| **Canonical doc** | `03-MLX/bandwidth-and-dispatch.md` (entropy section) + `05-THEORY-AND-DESIGNS/research-frontiers.md` §B.1 + §C caveat. |

### R7 — ANE MTP-head latency: estimate → confirmed 2.24 ms

| | |
|---|---|
| **Contradiction** | Can the MTP head run on the Apple Neural Engine, and at what latency? |
| **Sources that disagreed** | `TIMELINE.md`: ANE = "dead end" (coremltools broken on Py3.14, DeltaNet ops unsupported). `RESEARCH_FRONTIERS.md`: early estimate 0.037 ms compute / 10–22 ms weight load; then **CONFIRMED 2.24 ms** median latency (424.7M params / 849.5 MB, exact MLX output match; 1.46 ms CoreML proof of embed+norm+lm_head). |
| **Resolution** | The *draft model* on ANE is a dead end (DeltaNet ops unsupported). The *MTP head* (pure attn+MLP) **does** convert and runs at **2.24 ms** on ANE with zero GPU-bandwidth interference. Both states recorded (dead-end for the draft model, confirmed for the head). |
| **Canonical doc** | `03-MLX/the-journey.md` (Phase 4) + `03-MLX/bandwidth-and-dispatch.md` (frontiers); summarized in `05-THEORY-AND-DESIGNS/research-frontiers.md` §B.2. |

### R8 — Pre-fix vs post-fix llama.cpp variant numbers

| | |
|---|---|
| **Contradiction** | Which optimization-variant "wins" are real? |
| **Sources that disagreed** | `qwen-mtp-optimizations/README.md` reports wins (1.16×, 1.72×, 2.5×, drift-refresh "10× accept jump", predictive-hidden "+75pt accept") — most measured on **degraded text** *before* the cache-bookkeeping bug (infra patch 11) was fixed. |
| **Resolution** | Only **infra patch 01 adaptive chain (1.99×, post-fix)**, **08 ensemble fast-path (broken, post-fix)**, and **09 stacked-noise (0.58–0.69×, post-fix)** are trustworthy post-fix numbers. All pre-fix figures are marked "(pre-fix)" and preserved as an evolving thread, not flattened. (Index note: inventory-02 labels them by the optimization-patch numbering 01/08/09.) |
| **Canonical doc** | `02-LLAMACPP/optimization-variants.md` (honesty table) + `02-LLAMACPP/the-bug.md`. |

### R9 — RoPE dimensions: `rope_dimensions=[12,12,12,12]` vs `n_rot=64`

| | |
|---|---|
| **Contradiction** | The MTP-head MRoPE rotary configuration. |
| **Sources that disagreed** | `qwen-mtp-tensors/README.md`: MRoPE `rope_dimensions = [12,12,12,12]` (sum 48). `mtp_head.py`: rotary dims = `head_dim(256)·partial_rotary(0.25) = 64`. |
| **Resolution** | **`n_rot = 64` is authoritative.** Converter writes `add_rope_dimension_count(int(256·0.25))=64` (diff `01:103`); loader reads `hparams.n_rot` (`llama-model.cpp:619`, asserted vs `n_embd_head_k`). The `rope_sections[4]` MRoPE per-axis split is read separately from `LLM_KV_ROPE_DIMENSION_SECTIONS` (`llama-model.cpp:1034`). The README's `[12,12,12,12]` is an **unverified placeholder** — a valid GGML `sections[]` must sum to `n_rot/2 = 32`, not 48, so `[12,12,12,12]` cannot be the real array. Nothing in the load/graph path depends on the specific split at pos 0 (RoPE is identity there), only on length 4. (The 4 section integers themselves remain unpinned — see **O2**.) |
| **Canonical doc** | `01-ARCHITECTURE/qwen35-hybrid-architecture.md` (MRoPE section + core-dims table). |

### R10 — Split-recurrence rollback net-savings: ~6 ms vs 5.3 ms vs ~7.85 ms

| | |
|---|---|
| **Contradiction** | How much wall time does split-recurrence rollback save per step? |
| **Sources that disagreed** | `recurrent-rollback/README.md`: ~6 ms net / +2 tok/s (overhead ~1 ms / redo ~7 ms). `docs/TECHNIQUE.md` detailed model: 5.3 ms net (overhead 3.4 ms over 48 layers / redo 8.7 ms), and elsewhere ~7.85 ms (0.85 ms overhead / 8.7 ms redo). |
| **Resolution** | **Not a true contradiction — different per-layer dispatch accounting.** All land in **~5–8 ms net saved**. Each figure is carried with its own accounting basis. |
| **Canonical doc** | `05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md` §4.2. |

### R11 — MTP prediction offset: "+1" vs "+2"

| | |
|---|---|
| **Contradiction** | Which future token does the MTP head predict? |
| **Sources that disagreed** | `qwen-mtp-tensors/README.md`: predicts "the +1 token from the layer-63 hidden state". `mlx-qwen-mtp/README.md`: predicts "token t+2 from hidden at t and embedding of token t+1". |
| **Resolution** | **Same operation, two reference points.** Given `h_t` + the embedding of the next committed token (`t+1`), it outputs logits for the token after — `t+2` w.r.t. the hidden, `+1` w.r.t. the embedding input. Not a contradiction. |
| **Canonical doc** | `01-ARCHITECTURE/mtp-head-anatomy.md`. |

### R12 — MTP-head body GGUF naming: `blk.64.nextn.attn_q` vs `blk.64.attn_q`

| | |
|---|---|
| **Contradiction** | Do the MTP transformer-body tensors carry a `nextn.` prefix? |
| **Sources that disagreed** | `qwen-mtp-tensors/README.md` HF→GGUF table shows body under `nextn.` (`blk.64.nextn.attn_norm/.attn_q/.ffn_norm`). Converter remap + loader (diffs `01`/`02`) put the body in *plain* per-layer slots. |
| **Resolution** | The README's `nextn.`-prefixed body names are an **early/idealized convention**; the real on-disk names are the plain attention/FFN slots (`blk.64.attn_q`, …). Only the four MTP-specific tensors carry `nextn.` (`nextn.eh_proj/enorm/hnorm/shared_head_norm`, + optional `embed_tokens/shared_head_head`). The converter literally does `name.replace("mtp.layers.0", f"model.layers.{n_base}")` for the body (no `nextn` inserted). |
| **Canonical doc** | `01-ARCHITECTURE/tensor-layout-hf-to-gguf.md` (uses the real plain names). |

### R13 — Attention head shape: `24/4/256` vs `40/8/128`

| | |
|---|---|
| **Contradiction** | The real Qwen3.5-27B MTP-head attention dims. |
| **Sources that disagreed** | Real model: `num_heads=24, num_kv_heads=4, head_dim=256` (working `mtp_head.py`, MLX README explicit shapes, three vLLM configs reading real `text_config`). Design placeholders: `tensor-layout.md` uses `40 heads · head_dim 128` (square-Q `attn_q [5120,5120]`, `attn_q_norm [128]`); `per-position-heads.md` writes "n_head=40, n_head_kv=8, head_dim=128". |
| **Resolution** | **`24/4/256` is authoritative** (real, measured). The `40/8/128` figures are design placeholders for a *not-yet-trained* N=4 head sketch, documented as such — not measured values. |
| **Canonical doc** | `01-ARCHITECTURE/tensor-layout-hf-to-gguf.md` + `qwen35-hybrid-architecture.md`. |

---

## OPEN

### O1 — Model naming: "Qwen3.5-27B" (whole corpus) vs "Qwen3.6-27B" (Go CLI)

| | |
|---|---|
| **Question** | Is the model Qwen3.5-27B or Qwen3.6-27B? |
| **Sources that disagree** | Every domain doc + `_meta/CONVENTIONS.md`: **Qwen3.5-27B**. The `qwen-ops` Go CLI download/serve defaults: **Qwen3.6-27B** (`unsloth/Qwen3.6-27B-GGUF`). |
| **Why still open** | Cannot verify which is correct from in-corpus data — no checkpoint or upstream model card is present. The patches/tensors/arch the CLI drives are the documented Qwen3.5 / Qwen3-Next MTP work, so this reads as **source naming drift in the later tool**, not a different model. |
| **Corpus stance** | Canon = **Qwen3.5-27B**. The CLI's `Qwen3.6-27B` default is recorded, not silently normalized. |
| **Where flagged** | `06-TOOLING/README.md` (thread #1) + inventory-06. |

### O2 — Exact 4 `rope_sections` integers (and absence of a raw `config.json` in-corpus)

| | |
|---|---|
| **Question** | What are the four `rope_sections[4]` MRoPE per-axis split integers, and the exact `linear_*` config values? |
| **Sources** | `n_rot=64` and `sections[]` length-4 are pinned (R9); the converter reads `rope_sections` from `LLM_KV_ROPE_DIMENSION_SECTIONS`, but the **four integers are not present in any in-corpus config/GGUF**. The `linear_*` DeltaNet values (128/128/48/16, conv_kernel 4) are corroborated by 4 independent implementations (R1) but **not read from a checkpoint**. |
| **Why still open** | No raw Qwen3.5-27B `config.json` or GGUF is in the corpus to read these directly. A valid `sections[]` must sum to `n_rot/2 = 32`, ruling out `[12,12,12,12]`, but the actual split is unpinned. |
| **Why it doesn't block** | Nothing in the load/graph path depends on the specific section split at pos 0 (RoPE is identity there), only on length 4. The `linear_*` values agree across 4 impls, so they are treated as established. |
| **Where flagged** | `01-ARCHITECTURE/qwen35-hybrid-architecture.md` (sections marked placeholder) + inventory-01 threads #4, #6. |

### O3 (NOTE) — Minor benchmark rounding variants (both preserved verbatim)

| | |
|---|---|
| **Question** | Reconcile slightly different roundings of the same measured runs. |
| **Variants** | Per-position acceptance: `87/68/54/39/28/21/16%` (`vllm-patches-benchmarks.md`) vs `85/68/54/40/31%` (`07-FRESH-INSTALL.md`). llama.cpp K=2/thresh=0.85: **13.98 tok/s** 5-prompt mean (recipe + optimization README) vs **15.0 tok/s** K-sweep "sweet spot" single-prompt row. |
| **Disposition** | **Not a contradiction — different rounding / measurement basis (aggregate-mean vs single-prompt; two acceptance-table roundings).** Both kept **verbatim**, neither averaged. Recorded as an open NOTE only because the corpus deliberately carries both numbers rather than picking one. |
| **Where preserved** | `04-VLLM-GPU/README.md` (acceptance roundings, with note) + `02-LLAMACPP/the-recipe.md` (13.98 vs 15.0, noted inline). |

---

## Index: other reconciled threads (resolved within a single domain; recorded for completeness)

These were raised in the inventories and resolved cleanly inside one domain's docs; logged
here so the registry is exhaustive.

| Thread | Disposition | Canonical doc |
|---|---|---|
| MTP-head RoPE op: `ggml_rope_ext` (len-1 pos, earlier) vs `ggml_rope_multi` (len-4, later) | RESOLVED — len-4 `ggml_rope_multi` is current/correct; both preserved | `01-ARCHITECTURE/tensor-layout-hf-to-gguf.md` discovery #2 |
| `mtp.shared_head.head` presence | RESOLVED — for Qwen3.5-27B (`mtp_use_dedicated_embeddings:false`) no usable dedicated lm_head; 15-tensor count correct; collapses to main `output.weight` | `01-ARCHITECTURE/mtp-head-anatomy.md` |
| K>1 chaining: infra patch 05 ("architecturally K=1") vs the recipe winning at K=2 | RESOLVED — patch 05 says *naive unconditional* chaining doesn't help; recipe adds confidence gating; both preserved | `02-LLAMACPP/the-recipe.md` + `optimization-variants.md` |
| `MTP_FORCE_AR` vs `MTP_VERIFY_FORCE_AR` | RESOLVED — two distinct flags (disable drafting vs force verify-DeltaNet AR loop); disambiguated | `02-LLAMACPP/infrastructure-patches.md` + `source/README.md` |
| Authorship "Karpathy's original port" (patch 05 msg) vs `quivent` git author (commit 83babcae7) | NOTED, not resolved — likely colloquial narrative attribution, not literal committer | inventory-02 thread #5 |
| MLX best-tok/s: ~45 (mlx-qwen-mtp shipped stack) vs 51.1 (parallel-mtp-voting adaptive chain) | RESOLVED — different stacks; 51.1 is corpus headline; both stated | `03-MLX/README.md` |
| Vanilla adaptive chain 52.0 (MLX 0.30.6, historical) vs 51.1 (revalidated 2026-04-08) | RESOLVED — ~2% MLX version drift, noise-level; 51.1 is headline | `03-MLX/README.md` / `huihui-abliterated.md` |
| `speculative-draft-override.patch` vs `speculative-dual-mode.patch` | RESOLVED — superseding pair (gate-on-`mtp_num_hidden_layers` vs class-flag `_skip_mtp_conversion`); both copied | `04-VLLM-GPU/vllm-patches-and-strategies.md` thread #4 |
| modal_mtp potential (176.6 tok/s) vs measured (3.2 tok/s) | RESOLVED — draft path runs without CUDA graphs; *potential* vs *measured-today* distinguished | `04-VLLM-GPU/modal-self-speculative.md` thread #3 |
| modal patches duplicated across repos (modal `01/02/03` vs qwen-ops `qwen3_next`/`qwen3_5-shadow-state`/`modal_mtp`) | RESOLVED — same hooks, slightly different versions; both copied (modal prefixed `modal-`), relationship noted | `04-VLLM-GPU/vllm-patches-and-strategies.md` thread #5 |
| `01-SERVER-STATUS.md` describes RTX 5090 but lives under `deploy/gh200/` | RESOLVED — preserved as-is with a placement note | `04-VLLM-GPU/deployment.md` |
| `deltanet_adjuster.py` "~453M" (line 13) vs "~440M" (line 195) full MTP step | NOTED — code's own minor inconsistency; doc uses 453M (primary); both preserved | `04-VLLM-GPU/vllm-patches-and-strategies.md` |
| MTP-head swap size: ~600 MB / 0.1–0.3 ms (design doc) vs 849 MB / ~1 ms (code) | NOTED — different size assumptions; both retained | `04-VLLM-GPU/gpu-designs.md` #2 + strategies table |
| Layer-count phrasing "48 DeltaNet + 16 attention = 64" vs "full forward pass (64 layers)" | RESOLVED — consistent; only recurrent (48) layers are split | `05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md` |

---

*Provenance: synthesized by Agent 3C from `inventory-0[1-6]-*.md` "Contradictions & evolving
threads" sections + `coverage-report.md`. Canonical docs verified present this iteration.*
