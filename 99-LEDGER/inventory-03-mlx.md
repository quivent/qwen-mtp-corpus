# Inventory — 03-MLX (Agent C)

Every source file consumed for the MLX / Apple Silicon track → MTP destination → status.

Status legend: **distilled** (narrative absorbed), **copied** (artifact copied verbatim),
**merged** (combined with others into one doc), **duplicate-of-X** (byte-identical to
another source; collapsed to one canonical home), **deferred** (belongs to another agent).

---

## Docs / narrative sources

| Source file | MTP destination | Status |
|---|---|---|
| `mlx-qwen-mtp/README.md` | `03-MLX/README.md`, `mtp-self-speculative-mlx.md`, `kernel-fusion.md` | distilled |
| `qwen-inference-lab/README.md` | `03-MLX/README.md`, `the-journey.md` | distilled |
| `qwen-inference-lab/docs/TIMELINE.md` | `the-journey.md` (primary), `kernel-fusion.md`, `mtp-self-speculative-mlx.md` | distilled (canonical) |
| `qwen-inference-lab/docs/BANDWIDTH_ANALYSIS.md` | `bandwidth-and-dispatch.md` | merged (canonical) |
| `qwen-inference-lab/docs/DISPATCH_BARRIER_PROFILING.md` | `bandwidth-and-dispatch.md` | merged (canonical) |
| `qwen-inference-lab/docs/HUIHUI_ABLITERATED.md` | `huihui-abliterated.md` | distilled (canonical) |
| `qwen-inference-lab/docs/RESEARCH_FRONTIERS.md` | `bandwidth-and-dispatch.md` (frontiers section), `the-journey.md`, `kernel-fusion.md` | distilled — **cross-cutting** (see notes) |
| `qwen-mtp-research/docs/mlx-reference.md` | `the-journey.md` (Phase 7 / evolving thread), `README.md` | distilled (canonical) |

## Code / artifact sources

| Source file | MTP destination | Status |
|---|---|---|
| `mlx-qwen-mtp/src/mtp_head.py` | `03-MLX/src/mtp_head.py` | copied (canonical) |
| `mlx-qwen-mtp/src/generate.py` | `03-MLX/src/generate.py` | copied (canonical) |
| `mlx-qwen-mtp/src/fused_kernels_t2.py` | `03-MLX/src/fused_kernels_t2.py` | copied (canonical) |
| `mlx-qwen-mtp/src/extract_weights.py` | `03-MLX/src/extract_weights.py` | copied (canonical) |
| `mlx-qwen-mtp/src/__init__.py` | `03-MLX/src/__init__.py` | copied |
| `mlx-qwen-mtp/src/__pycache__/*.pyc` | — | skipped (build artifacts) |
| `mlx-qwen-mtp/{LICENSE,pyproject.toml}` | — | not copied (Apache-2.0 noted in README) |
| `qwen-inference-lab/kernels/fused_gdn.py` | `03-MLX/src/fused_gdn.py` | copied (V2–V7 exploration; winning kernels also in generate.py) |
| `qwen-inference-lab/benchmarks/bench_v7.py` | `03-MLX/src/bench_v7.py` | copied |
| `qwen-inference-lab/benchmarks/extract_mtp_huihui.py` | `03-MLX/src/extract_mtp_huihui.py` | copied |
| `qwen-inference-lab/logs/adaptive_mtp_vanilla.log` | `03-MLX/src/logs/adaptive_mtp_vanilla.log` | copied (51.1 tok/s run) |
| `qwen-inference-lab/logs/adaptive_mtp_huihui.log` | `03-MLX/src/logs/adaptive_mtp_huihui.log` | copied (49.5 tok/s run) |
| `qwen-inference-lab/{LICENSE,pyproject.toml}` | — | not copied |

## Duplicates collapsed (byte-identical — verified via `diff`)

| Duplicate path | Canonical home | Status |
|---|---|---|
| `qwen-ops/mlx/mtp_head.py` | `mlx-qwen-mtp/src/mtp_head.py` | duplicate-of (byte-identical) |
| `qwen-ops/mlx/generate.py` | `mlx-qwen-mtp/src/generate.py` | duplicate-of (byte-identical) |
| `qwen-ops/mlx/fused_kernels_t2.py` | `mlx-qwen-mtp/src/fused_kernels_t2.py` | duplicate-of (byte-identical) |
| `qwen-ops/mlx/extract_weights.py` | `mlx-qwen-mtp/src/extract_weights.py` | duplicate-of (byte-identical) |
| `qwen-ops/mlx/__init__.py` | `mlx-qwen-mtp/src/__init__.py` | duplicate-of (byte-identical) |
| `qwen-ops/mlx/pyproject.toml` | `mlx-qwen-mtp/pyproject.toml` | duplicate-of |
| `qwen-ops/research/findings/TIMELINE.md` | `qwen-inference-lab/docs/TIMELINE.md` | duplicate-of (byte-identical) |
| `qwen-ops/research/findings/BANDWIDTH_ANALYSIS.md` | `qwen-inference-lab/docs/BANDWIDTH_ANALYSIS.md` | duplicate-of (byte-identical) |
| `qwen-ops/research/findings/DISPATCH_BARRIER_PROFILING.md` | `qwen-inference-lab/docs/DISPATCH_BARRIER_PROFILING.md` | duplicate-of (byte-identical) |
| `qwen-ops/research/findings/HUIHUI_ABLITERATED.md` | `qwen-inference-lab/docs/HUIHUI_ABLITERATED.md` | duplicate-of (byte-identical) |
| `qwen-ops/research/findings/RESEARCH_FRONTIERS.md` | `qwen-inference-lab/docs/RESEARCH_FRONTIERS.md` | duplicate-of (byte-identical) |
| `qwen-ops/research/findings/mlx-reference.md` | `qwen-mtp-research/docs/mlx-reference.md` | duplicate-of (byte-identical) |

**Net: 12 byte-identical duplicate files collapsed** (5 code + 1 pyproject + 6 docs). All
confirmed identical with `diff -q`.

## Deferred to other agents

| Source file | Owner | Why |
|---|---|---|
| `qwen-inference-lab/logs/llama_cpp_server.log` | Agent B (llama.cpp) | llama.cpp server output (`Qwen3.5-27B-Q4_K_M.gguf`, `qwen35` arch); referenced by the MLX README as "MTP+draft performance" but is a llama.cpp artifact. |
| `qwen-inference-lab/logs/llama_cpp_server_nomtp.log` | Agent B (llama.cpp) | llama.cpp no-MTP baseline server log. |
| `qwen-ops/research/findings/the-recipe.md` | Agent B (llama.cpp) | the 1.99× llama.cpp recipe. |
| `qwen-ops/research/findings/tensor-layout.md` | Agent A (Architecture) | HF→GGUF tensor mapping. |
| `qwen-ops/research/findings/modal-mtp-precision-divergence.md` | Agent D (vLLM/GPU) | Modal FP16 precision saga. |
| `qwen-ops/research/designs/per-position-heads.md`, `qwen-mtp-research/docs/per-position-heads.md` | Agent E (Theory) | per-position-heads design (MLX cross-links to it re: chained single-head vs per-position). |
| `qwen-ops/research/designs/{CASCADE-MTP-TRAINING,DELTANET-WEIGHT-TRANSPLANT,EAGLE-PR-DESCRIPTIONS,MTP-TREE-CONFIG-PATH,PLV-FULL-ATTN-BOUNDARIES,integration-plan,modal-mtp-design}.md` etc. | Agents A/D/E | designs outside MLX scope. |
| `qwen-ops/research/designs/{GH200-COMPUTE-BANDWIDTH-ANALYSIS,CUDAGRAPH-WEIGHT-SWAP,PROPOSE-TREE-CUDAGRAPH-ANALYSIS,TWO-GRAPH-CUDA-DISPATCH}.md` | Agent D (vLLM/GPU) | CUDA/GH200 specific. |
| `recurrent-rollback/docs/TECHNIQUE.md`, `recurrent-rollback/src/*` | Agent E (Theory) | general split-recurrence technique; MLX docs cross-link to `../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`. |

---

## Contradictions & evolving threads

1. **"MLX uses trained per-position heads" → CORRECTED to "single head, chained
   recurrence."** Early framing attributed the MLX 1.68× to multiple trained MTP heads.
   `mlx-reference.md` resolves it: `mtp_weights_vanilla.safetensors` has exactly **one**
   block (`mtp.layers.0.*`, no `layers.1`). The win is *algorithmic* — chained recurrent
   application of the single head + a stacked 0.8B draft + confidence gating, at **$0
   training cost**. Presented resolved-up-top in `the-journey.md` (Phase 7) and
   `mtp-self-speculative-mlx.md`, with the journey preserved. (The per-position-heads
   *design* itself is Theory's, Agent E.)

2. **rms_norm+matmul fusion: "saves 10.4 ms / 2.0 ms" vs "+0.76 ms slower on real model."**
   `DISPATCH_BARRIER_PROFILING.md` and `mlx-qwen-mtp/README.md` report a 2.0 ms measured
   saving (256-matmul microbenchmark via `mx.fast.metal_kernel`) and an upstream 10.4 ms
   (30%) projection with full metallib integration. `RESEARCH_FRONTIERS.md` lists it under
   "NOT pursuing," noting the real model showed **+0.76 ms slower** because `mx.compile`
   already handles the dispatch pipeline. Resolution (in `kernel-fusion.md`): the fusion
   is a real win **only** with full metallib integration replacing the stock kernel; the
   `mx.fast.metal_kernel` prototype wins a microbenchmark but loses to `mx.compile`'s
   scheduling on the full model. **Not in the shipped path.**

3. **Best-tok/s figure: ~45 (mlx-qwen-mtp README) vs 51.1 (qwen-inference-lab README).**
   Not a contradiction — different stacks. `mlx-qwen-mtp` projects ~45 tok/s (~1.52×) for
   *its own* shipped stack (split-recurrence + partial rms_norm fusion). The **51.1**
   headline is the **adaptive MTP confidence chain + batch verify** from the sibling
   `parallel-mtp-voting` project (only its logs are in-corpus). Both stated explicitly in
   `README.md`; 51.1 is the corpus headline (matches CONVENTIONS).

4. **Vanilla adaptive chain: 52.0 (historical, MLX 0.30.6) vs 51.1 (revalidated
   2026-04-08, current MLX).** ~2% MLX version drift, noise-level. The 52.0 anchor is
   "verified" per `HUIHUI_ABLITERATED.md`; 51.1 is the current measured value and the one
   used as the headline.

5. **Entropy-coded weights: "the big prize, 80–90 tok/s" → DEBUNKED.** `RESEARCH_FRONTIERS.md`
   first frames entropy coding as the transformative 3.3× change, then (same doc, dated
   correction) measures **3.72 bits actual entropy (not 1.1), 1.08× compression, 7%
   savings** on Qwen3.5-27B-4bit. Recorded resolved in `bandwidth-and-dispatch.md`.

6. **CoreML/ANE: "dead end" (TIMELINE) → "CONFIRMED 2.24 ms" (RESEARCH_FRONTIERS).**
   The *draft model* on ANE is a dead end (coremltools broken on Py3.14, DeltaNet ops
   unsupported). The *MTP head* (pure attn+MLP) **does** convert and runs at 2.24 ms on
   ANE with zero GPU-bandwidth interference. Both states recorded in `the-journey.md`
   (Phase 4) and `bandwidth-and-dispatch.md` (frontiers).

7. **`RESEARCH_FRONTIERS.md` is cross-cutting (iter-2 explicitly flagged for iter-3
   dedup).** It contains M4-Max-specific findings (ANE 2.24 ms + the 0.037 ms/11 TOPS/
   10–22 ms feasibility figures, CPU SME, die topology, SLC, the 20% bandwidth-gap
   physics, entropy-debunk on *this* model, megakernel, M5 outlook) that **stay here**,
   **and** platform-general speculative-decode *math* that overlaps Agent E's Theory
   domain:
   - parallel-MTP probabilities (`P(both correct)=0.79²` etc.) + the 22% break-even,
   - the O(1)/token ceiling reframing,
   - batched-inference weight reuse (batch=4 → 6.3 ms/tok),
   - the BPW→GB→ms→tok/s bandwidth-floor math behind the combined-projection table.

   In iteration 2 the frontiers section of `bandwidth-and-dispatch.md` was re-annotated
   inline marking each subsection **MLX-hardware-specific (keep here)** vs **general theory
   (see `../05-THEORY-AND-DESIGNS/research-frontiers.md`)**. Nothing was deleted — the
   general math is retained here too (with M4-Max numbers plugged in) so the MLX doc stays
   self-contained.

   **Confirmed against Agent E's actual doc:** `05-THEORY-AND-DESIGNS/research-frontiers.md`
   already exists and is structured for exactly this split — its **Section A**
   ("this domain") owns the general theory (A.1 Parallel MTP n=2/n=3 with the same
   `P(both correct)=0.79²` math and 22% break-even, §0 the O(1)-per-token ceiling and
   213 tok/s @N=8, A.x batched reuse), and its **Section B** is a *lossless summary* of the
   hardware frontiers that explicitly defers detail to `../03-MLX/`. The two docs are
   already cross-consistent (same numbers: 52.6 / 57.4 / 82.3 / 87.4 tok/s combined
   projection; 2.24 ms ANE; 3.72-bit entropy debunk). **Iter-3 dedup action:** the general
   math currently duplicated in `bandwidth-and-dispatch.md` (parallel-MTP probabilities,
   break-even, O(1) ceiling, batched reuse, BPW→ms→tok/s floor math) can be reduced to a
   one-line pointer to Theory Section A, leaving the M4-Max numeric instantiation here;
   conversely Theory Section B should keep deferring hardware detail here. No content
   conflict between the two — only intentional overlap to collapse.

## Gaps / not-in-corpus

- The `adaptive_mtp.py` / `stacked_v2.py` driver that produces 51.1 tok/s lives in the
  sibling `parallel-mtp-voting` project, **not in this corpus**. Only its logs were
  captured. The technique is described from `mlx-reference.md` + the logs; the source is
  unavailable.
- `~/optimizations/qwen-mtp-inference/`, `~/mlx-fork/`, the ANE `.mlpackage`, and the
  `quivent/mlx-fused-qmv` metallib work are referenced but external to the corpus.

---

## Losslessness verification (iter 2)

Iteration-2 pass: enumerated EVERY file in `mlx-qwen-mtp/`, `qwen-inference-lab/`, and the
MLX part of `qwen-ops/mlx/`; re-ran `diff -q` on all duplicates and all copied `src/`
artifacts (all still byte-identical); and verified that every decimal number in
`TIMELINE.md`, `BANDWIDTH_ANALYSIS.md`, `DISPATCH_BARRIER_PROFILING.md`,
`RESEARCH_FRONTIERS.md`, `HUIHUI_ABLITERATED.md`, `mlx-reference.md`, and both source
READMEs appears verbatim somewhere in `MTP/03-MLX/` (automated grep sweep → **zero
missing**).

| Source file | Captured where (MTP/03-MLX) | Status (iter 2) |
|---|---|---|
| `mlx-qwen-mtp/README.md` | README.md, mtp-self-speculative-mlx.md, kernel-fusion.md | distilled — verified complete (incl. ~45/~1.52×, ~75 tok/s projection) |
| `mlx-qwen-mtp/src/mtp_head.py` | src/mtp_head.py | copied — `diff -q` identical ✓ |
| `mlx-qwen-mtp/src/generate.py` | src/generate.py | copied — identical ✓; prose re-verified against code |
| `mlx-qwen-mtp/src/fused_kernels_t2.py` | src/fused_kernels_t2.py | copied — identical ✓; kernel names re-verified |
| `mlx-qwen-mtp/src/extract_weights.py` | src/extract_weights.py | copied — identical ✓ |
| `mlx-qwen-mtp/src/__init__.py` | src/__init__.py | copied — identical ✓ |
| `mlx-qwen-mtp/{LICENSE,pyproject.toml}` | — | not copied (Apache-2.0 noted in README); intentional |
| `qwen-inference-lab/README.md` | README.md, the-journey.md | distilled — verified complete |
| `qwen-inference-lab/docs/TIMELINE.md` | the-journey.md (canonical), kernel-fusion.md | distilled — all numbers present ✓ |
| `qwen-inference-lab/docs/BANDWIDTH_ANALYSIS.md` | bandwidth-and-dispatch.md | merged — all numbers present ✓ |
| `qwen-inference-lab/docs/DISPATCH_BARRIER_PROFILING.md` | bandwidth-and-dispatch.md, kernel-fusion.md | merged — full per-op tables present ✓ |
| `qwen-inference-lab/docs/RESEARCH_FRONTIERS.md` | bandwidth-and-dispatch.md (frontiers) | distilled — **iter-2 fixed gaps** (see below); cross-cutting flagged |
| `qwen-inference-lab/docs/HUIHUI_ABLITERATED.md` | huihui-abliterated.md (canonical) | distilled — all numbers present ✓ |
| `qwen-inference-lab/kernels/fused_gdn.py` | src/fused_gdn.py | copied — identical ✓ |
| `qwen-inference-lab/benchmarks/bench_v7.py` | src/bench_v7.py | copied — identical ✓ |
| `qwen-inference-lab/benchmarks/extract_mtp_huihui.py` | src/extract_mtp_huihui.py | copied — identical ✓ |
| `qwen-inference-lab/logs/adaptive_mtp_vanilla.log` | src/logs/adaptive_mtp_vanilla.log | copied — identical ✓ (51.1 tok/s) |
| `qwen-inference-lab/logs/adaptive_mtp_huihui.log` | src/logs/adaptive_mtp_huihui.log | copied — identical ✓ (49.5 tok/s) |
| `qwen-inference-lab/logs/llama_cpp_server{,_nomtp}.log` | — | deferred to Agent B (llama.cpp), intentional |
| `qwen-inference-lab/{LICENSE,pyproject.toml}` | — | not copied; intentional |
| `qwen-mtp-research/docs/mlx-reference.md` | the-journey.md (Phase 7), README.md | distilled — all numbers present ✓ |
| `qwen-ops/mlx/{mtp_head,generate,fused_kernels_t2,extract_weights,__init__}.py` | (canonical = mlx-qwen-mtp/src) | duplicate-of — re-`diff`'d byte-identical ✓ |
| `qwen-ops/mlx/pyproject.toml` | (canonical = mlx-qwen-mtp/pyproject.toml) | duplicate-of — byte-identical ✓ |

**Gaps found and fixed in iteration 2** (all in `bandwidth-and-dispatch.md`, all from
`RESEARCH_FRONTIERS.md`):

1. **ECQ measured benchmark** — `3.27× throughput (42.9 → 140 tok/s) on M3 Pro`. A real
   measured Apple-Silicon number; was previously summarized only as the "~3.3×" claim.
   Now stated verbatim in the entropy-coding section (with the debunk that it doesn't
   transfer to standard 4-bit Qwen3.5-27B).
2. **The combined-stack projection table** — the full 6-row table (current 42.7 → n=2
   52.6 → mixed-3/4 **57.4** → entropy **82.3** → entropy+n=3 **87.4** → entropy+mixed
   **~95** tok/s, with effective BPW / GB / BW-floor columns). Previously only the 80–90
   tok/s headline was kept; the intermediate rows were missing.
3. **Frontier-3 ANE feasibility numbers** — MTP compute **0.037 ms** at ~11 TOPS; ANE MTP
   weight load **10–22 ms**; enabling CoreML proof of embed+norm+lm_head at **1.46 ms**.
   Previously only the confirmed 2.24 ms was kept.
4. **`P(both wrong) = 0.04` (4%, NOT 20%)** — the fourth line of the parallel-MTP
   probability table; previously dropped.
5. **Megakernel detail** — `~500+ kernel dispatches per token`, argument-buffer overhead,
   and the `2.4 ms gap (33.9 − 31.4 ms)` attribution. Previously only the 10–15% estimate.
6. **Entropy-coding references** — drxddy/ecq, MLX #3043, EntroLLM (arXiv:2505.02380),
   DFloat11 (arXiv:2504.11651), Float8@2bits. Previously partially dropped.
7. **RESEARCH_FRONTIERS priority order (1–5)** — added as a compact list.
8. Die-topology detail (wire distance 3–20 mm, ~0.2 ns propagation; SLC distributed near
   controllers; depth-first scheduling reasoning) — folded into the die-topology bullet.

**Code↔prose verification (iter 2, Task 2).** Read `src/generate.py` and
`src/fused_kernels_t2.py` and confirmed every prose claim:
- Shipped Metal kernels: Python wrappers `fused_conv1d_silu` (internal Metal name
  `fused_conv1d_silu_v2`), `fused_gdn_step` (internal `fused_gdn_step_v2`, masked variant
  `fused_gdn_step_v2_mask`), `fused_gdn_call_v2`, `patch_model`, `mtp_generate` — all
  present and named as `kernel-fusion.md`/`src/README.md` state.
- T=2 verify kernels: `fused_conv1d_silu_t2` (outputs `conv_out_0, conv_out_1, mid_state,
  final_state`), `fused_gdn_step_with_intermediate` (snapshots `state_mid` after t=0),
  `fused_rms_norm_qmv` — all present, matching the prose.
- `fused_gdn_step` dispatch dims `grid=(32, Dv, B*Hv)`, `threadgroup=(32, 4, 1)` match
  kernel-fusion.md exactly.
- `patch_model` concatenates the 4 input-projection quantized weights/scales/biases along
  `axis=0`, precomputes `_A_exp = exp(A_log)`, pre-flattens conv weights to `[conv_dim, 4]`
  — matches the prose.
- Split-recurrence rollback in `mtp_generate`/`fwd_t2_rollback`: `delta_stash` saves
  `(conv_mid, rnn_mid)` zero-copy refs; on reject `rollback()` restores DeltaNet
  `c[0],c[1]=delta_stash[di]` and sets attention `c.offset = kv_offsets[ki] + 1` (keep
  token 0); accept path keeps `[draft, bonus]`; optimistic next-draft + single
  `mx.eval(verify, bonus, next_draft)` async pipeline — matches mtp-self-speculative-mlx.md
  step-for-step. **No mismatch found; no prose correction needed.**

**Net iter-2 result:** losslessness CONFIRMED for the MLX domain. 12 byte-identical
duplicates remain correctly collapsed; 10 src artifacts remain byte-identical copies;
8 distinct numeric/reference gaps from `RESEARCH_FRONTIERS.md` were filled. Code matches
prose with zero corrections required.
