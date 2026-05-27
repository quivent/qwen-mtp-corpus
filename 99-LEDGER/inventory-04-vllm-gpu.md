# Inventory — 04 vLLM / GPU track (Agent D)

Every source file consumed → MTP destination → status. Domain: vLLM + GH200 /
RTX 5090, modal self-speculative, precision-divergence, AWQ/GPTQ W4A16, GPU
design docs.

Status legend: **distilled** (prose absorbed into a doc), **copied** (verbatim
artifact), **merged** (folded into a multi-source doc), **duplicate-of** (byte-
identical, collapsed), **deferred** (other agent's domain).

## modal-mtp/

| Source | MTP destination | Status |
|---|---|---|
| `modal-mtp/README.md` | `modal-self-speculative.md`, `precision-divergence.md`, `README.md` (results) | distilled |
| `modal-mtp/docs/design.md` | `modal-self-speculative.md` | distilled |
| `modal-mtp/debug_indexerror.md` | `modal-self-speculative.md` (IndexError section) | distilled |
| `modal-mtp/modal_mtp.py` | `optimizations/modal_mtp_proposer_reference.py`; cited in `modal-self-speculative.md` | copied + distilled |
| `modal-mtp/diagnose_divergence.py` | `optimizations/diagnose_divergence.py`; cited in `precision-divergence.md` | copied + distilled |
| `modal-mtp/validate_draft_accuracy.py` | `optimizations/validate_draft_accuracy.py` | copied + distilled |
| `modal-mtp/validate_extended.py` | `optimizations/validate_extended.py` | copied + distilled |
| `modal-mtp/validate_extended_output.log` | `optimizations/validate_extended_output.log`; decisive evidence in `precision-divergence.md` | copied + distilled |
| `modal-mtp/test_partial_skip.py` | `optimizations/test_partial_skip.py`; cited in `modal-self-speculative.md` | copied + distilled |
| `modal-mtp/patches/01-skip-attention.patch` | `patches/modal-01-skip-attention.patch` | copied |
| `modal-mtp/patches/02-draft-mode-and-state.patch` | `patches/modal-02-draft-mode-and-state.patch` | copied |
| `modal-mtp/patches/03-model-runner-draft-loop.patch` | `patches/modal-03-model-runner-draft-loop.patch` | copied |

## qwen-ops/vllm/

| Source | MTP destination | Status |
|---|---|---|
| `vllm/patches/apply.sh` | `patches/apply.sh`; `vllm-patches-and-strategies.md` | copied + distilled |
| `vllm/patches/eagle.patch` | `patches/eagle.patch`; tables in `vllm-patches-and-strategies.md`, `gpu-designs.md` (EAGLE PRs) | copied + distilled |
| `vllm/patches/qwen3_next.patch` | `patches/qwen3_next.patch` | copied + distilled |
| `vllm/patches/qwen3_5-shadow-state.patch` | `patches/qwen3_5-shadow-state.patch` | copied + distilled |
| `vllm/patches/gdn-shadow-state.patch` | `patches/gdn-shadow-state.patch` | copied + distilled |
| `vllm/patches/gdn-inhibition-cycle.patch` | `patches/gdn-inhibition-cycle.patch` | copied + distilled |
| `vllm/patches/speculative-dual-mode.patch` | `patches/speculative-dual-mode.patch` | copied + distilled |
| `vllm/patches/speculative-draft-override.patch` | `patches/speculative-draft-override.patch` (noted as superseded by dual-mode) | copied + distilled |
| `vllm/patches/speculative-mtp-tree-compat.patch` | `patches/speculative-mtp-tree-compat.patch` | copied + distilled |
| `vllm/patches/recurrent-rollback.patch` | `patches/recurrent-rollback.patch`; `deployment.md`, `vllm-patches-and-strategies.md` | copied + distilled |
| `vllm/patches/modal_mtp.patch` | `patches/modal_mtp.patch` | copied + distilled |
| `vllm/patches/int8-embedding.patch` | `patches/int8-embedding.patch`; `deployment.md` | copied + distilled |
| `vllm/patches/gh200-strip-torch-dep.patch` | `patches/gh200-strip-torch-dep.patch` | copied + distilled |
| `vllm/patches/llmcompressor-conv1d.patch` | `patches/llmcompressor-conv1d.patch`; `quantization.md` | copied + distilled |
| `vllm/optimizations/adaptive_mtp.py` | `optimizations/adaptive_mtp.py`; table | copied + distilled |
| `vllm/optimizations/modal_mtp.py` | `optimizations/modal_mtp.py`; `modal-self-speculative.md` | copied + distilled |
| `vllm/optimizations/partial_layer_verify.py` | `optimizations/partial_layer_verify.py`; table | copied + distilled |
| `vllm/optimizations/plv_bench.py` | `optimizations/plv_bench.py`; table | copied + distilled |
| `vllm/optimizations/plv_layer60_bench.py` | `optimizations/plv_layer60_bench.py`; table | copied + distilled |
| `vllm/optimizations/early_verify_probe.py` | `optimizations/early_verify_probe.py`; table | copied + distilled |
| `vllm/optimizations/deltanet_transplant.py` | `optimizations/deltanet_transplant.py`; `gpu-designs.md` | copied + distilled |
| `vllm/optimizations/deltanet_transplant_w4a16.py` | `optimizations/deltanet_transplant_w4a16.py`; table | copied + distilled |
| `vllm/optimizations/cascade_mtp_corrective.py` | `optimizations/cascade_mtp_corrective.py`; table | copied + distilled |
| `vllm/optimizations/deltanet_adjuster.py` | `optimizations/deltanet_adjuster.py`; table | copied + distilled |
| `vllm/optimizations/enhanced_mtp_proposer.py` | `optimizations/enhanced_mtp_proposer.py`; table | copied + distilled |
| `vllm/optimizations/native_multi_head.py` | `optimizations/native_multi_head.py`; table, `gpu-designs.md` | copied + distilled |
| `vllm/optimizations/sibling_sequential.py` | `optimizations/sibling_sequential.py`; table | copied + distilled |
| `vllm/optimizations/selective_state_snapshot.py` | `optimizations/selective_state_snapshot.py`; `modal-self-speculative.md` | copied + distilled |
| `vllm/microgreens/__init__.py` | `optimizations/microgreens/__init__.py` | copied |
| `vllm/microgreens/mtp_clone.py` | `optimizations/microgreens/mtp_clone.py`; `quantization.md` (MTP keys) | copied + distilled |
| `vllm/microgreens/mtp_diversity_train.py` | `optimizations/microgreens/mtp_diversity_train.py`; table | copied + distilled |
| `vllm/microgreens/sibling_mtp_proposer.py` | `optimizations/microgreens/sibling_mtp_proposer.py`; table | copied + distilled |
| `vllm/scripts/vllm-tree-spec.sh` | `optimizations/scripts/vllm-tree-spec.sh`; table | copied + distilled |
| `vllm/scripts/bench-tok-s.py` | `optimizations/scripts/bench-tok-s.py`; table | copied + distilled |
| `vllm/scripts/quantize_deltanet.py` | `optimizations/scripts/quantize_deltanet.py`; `quantization.md` | copied + distilled |

## qwen-ops/quantization/autoawq-qwen35/

| Source | MTP destination | Status |
|---|---|---|
| `qwen3_5.py` | `quantization/qwen3_5.py`; `quantization.md` | copied + distilled |
| `inject_mtp_weights.py` | `quantization/inject_mtp_weights.py`; `quantization.md` | copied + distilled |
| `__init__.py.patch` | `quantization/__init__.py.patch`; `quantization.md` | copied + distilled |
| `auto.py.patch` | `quantization/auto.py.patch`; `quantization.md` | copied + distilled |
| `base.py.patch` | `quantization/base.py.patch`; `quantization.md` | copied + distilled |
| `quantizer.py.patch` | `quantization/quantizer.py.patch`; `quantization.md` | copied + distilled |

## qwen-ops/deploy/

| Source | MTP destination | Status |
|---|---|---|
| `deploy/gh200/deploy.sh` | `deploy/gh200/deploy.sh`; `deployment.md` | copied + distilled |
| `deploy/gh200/01-SERVER-STATUS.md` | `deploy/gh200/01-SERVER-STATUS.md`; `deployment.md` (RTX 5090 section) | copied + distilled |
| `deploy/gh200/07-FRESH-INSTALL.md` | `deploy/gh200/07-FRESH-INSTALL.md`; `deployment.md`, `quantization.md` | copied + distilled |
| `deploy/gh200/08-GH200-AGENT-INSTALL.md` | `deploy/gh200/08-GH200-AGENT-INSTALL.md`; `deployment.md` | copied + distilled |
| `deploy/rtx5090/05-NIXOS-GUIDE.md` | `deploy/rtx5090/05-NIXOS-GUIDE.md`; `deployment.md` | copied + distilled |
| `deploy/rtx5090/nixos-captain-configuration.nix` | `deploy/rtx5090/nixos-captain-configuration.nix`; `deployment.md` | copied + distilled |
| `deploy/rtx5090/vllm-serve.sh` | `deploy/rtx5090/vllm-serve.sh`; `deployment.md` | copied + distilled |
| `deploy/rtx5090/vllm-watchdog.sh` | `deploy/rtx5090/vllm-watchdog.sh`; `deployment.md` | copied + distilled |

## qwen-ops/research/benchmarks/

| Source | MTP destination | Status |
|---|---|---|
| `vllm-patches-benchmarks.md` | `README.md` (results), `vllm-patches-and-strategies.md` | distilled |
| `autoawq-benchmarks.md` | `quantization.md`, `README.md` (RTX 5090 results) | distilled |
| `inference-lab-benchmarks.md` | — | deferred (Agent C, M4 Max) |

## qwen-ops/research/findings/

| Source | MTP destination | Status |
|---|---|---|
| `modal-mtp-precision-divergence.md` | `precision-divergence.md` | distilled |
| `BANDWIDTH_ANALYSIS.md`, `DISPATCH_BARRIER_PROFILING.md`, `TIMELINE.md`, `HUIHUI_ABLITERATED.md`, `RESEARCH_FRONTIERS.md`, `mlx-reference.md`, `tensor-layout.md`, `the-recipe.md` | — | deferred (Agents A/B/C/E; mostly M4 Max / arch / dup) |

## qwen-ops/research/designs/

| Source | MTP destination | Status |
|---|---|---|
| `GH200-COMPUTE-BANDWIDTH-ANALYSIS.md` | `gpu-designs.md` #1 | distilled |
| `CUDAGRAPH-WEIGHT-SWAP.md` | `gpu-designs.md` #2 | distilled |
| `TWO-GRAPH-CUDA-DISPATCH.md` | `gpu-designs.md` #3 | distilled |
| `PROPOSE-TREE-CUDAGRAPH-ANALYSIS.md` | `gpu-designs.md` #4 | distilled |
| `MTP-TREE-CONFIG-PATH.md` | `gpu-designs.md` #5 | distilled |
| `PLV-FULL-ATTN-BOUNDARIES.md` | `gpu-designs.md` #6 | distilled |
| `DELTANET-WEIGHT-TRANSPLANT.md` | `gpu-designs.md` #7 | distilled |
| `EAGLE-PR-DESCRIPTIONS.md` | `gpu-designs.md` #8 | distilled |
| `modal-mtp-design.md` | `modal-self-speculative.md` | **duplicate-of** `modal-mtp/docs/design.md` (byte-identical, collapsed) |
| `CASCADE-MTP-TRAINING.md` | — | deferred (Agent E) |
| `per-position-heads.md` | — | deferred (Agent E) |
| `integration-plan.md` | — | deferred (cross-domain plan; Agent A/B) |

---

## Contradictions & evolving threads

### #1 — The precision-divergence saga (the big one)

**The single most important contradiction in this corpus.** The modal-mtp README
contradicts *itself* across sequential sections, and the understanding evolved
through three stages:

1. **FP32/CPU 100% (5/5)** → "architecture is sound."
2. **FP16/GPU ~15% (12/12)** → README's "FP16 Accuracy Update" concludes *"the
   attention layers are doing real work at FP16… it's a precision problem"*.
   It also adds an "FP32 State Accumulation Path" framing the
   `--mamba-ssm-cache-dtype float32` fix as a *precision* fix.
3. **CPU FP16 = 100% (12/12)** (proven by `validate_extended_output.log`:
   1146/1146 tokens) while GPU FP16 = ~15% → README's later "Critical Finding:
   CPU vs GPU Divergence" reverses stage 2: *same precision, different device* ⇒
   **NOT precision — it's CUDA fused-kernel numerical behavior**
   (`causal_conv1d`, `chunk_gated_delta_rule`).

**How I resolved the presentation:** `precision-divergence.md` leads with the
RESOLVED current picture (kernel-level, fixable/tolerable, architecture proven
sound at all precisions on CPU), with the full 4-row test matrix up top, then
explicitly labels the stage-2 "precision problem" framing as **SUPERSEDED** and
preserves it as a documented dead-end. The companion findings doc
(`modal-mtp-precision-divergence.md`) already presents the resolved framing and
was used to confirm the resolution. The FP32-state fix is kept but re-labeled:
it may still help (FP32 damps kernel noise) but the *reason* given in stage 2 was
the wrong diagnosis, and it remains **untested in vLLM**.

### #2 — GH200 throughput: 186 tok/s vs 94 tok/s

`GH200-COMPUTE-BANDWIDTH-ANALYSIS.md` reports **baseline 94 tok/s / spec-decode
36 tok/s** single-request and explicitly disputes the **186 tok/s** figure used
everywhere else (benchmarks + deploy docs). It also uses a *different model
architecture profile* (hidden 3584, 28 heads, head_dim 128, MLP intermediate
18944, vocab 151936, 64 layers, ~16B "effective") than the corpus headline
Qwen3.5-27B (hidden 5120, 24 heads, head_dim 256, intermediate 17408, vocab
248320). The doc's own hypothesis: 186 was aggregate/concurrent throughput or
counted speculated (not accepted) tokens or a different config.

**Iter-2 finding — it is measuring a DIFFERENT MODEL, proven by FLOP math.** The
doc's own FLOP table (per-layer 0.466 GFLOPs = attn 0.059 + MLP 0.407, ×64 =
29.83, + LM-head 1.09 = **30.92 GFLOPs/token**) reproduces *exactly* from the
3584-hidden / 18944-intermediate / 151936-vocab profile. Recomputing against the
real 27B arch (5120 / 17408 / 248320) gives **46.16 GFLOPs/token** (≈1.5×). The
doc's W4A16 weight transfer of **8.0 GB** = ~16B params @ 0.5 B/param, not the
27B model's ~13.5 GB (corpus says 13.7 GB 4-bit). 248320 is the real vocab (live
code: `early_verify_probe.py VOCAB_SIZE=248320`, `sibling_mtp_proposer.py`,
`mtp_diversity_train.py`); the 3584/18944 arch appears in **no other source
file**. The "Qwen 3.5-27B W4A16" title is therefore a **mislabel** — the roofline
was run against a ~16B-class (Qwen3-14B-shaped) model. The sibling
TWO-GRAPH-CUDA-DISPATCH doc citing "70B model" and "DeepSeek-V3/R1" confirms these
GH200 design docs span multiple model sizes.

**Resolution:** both numbers kept, neither rounded away. `gpu-designs.md` #1 now
carries an expanded contradiction callout with the FLOP proof: 186/1030 =
production aggregate/batched on the real 27B (benchmark + deploy), 94/36 =
single-request on a *different, smaller arch* that establishes the qualitative
point (GH200 single-request is **overhead-bound**, and single-request spec decode
can *hurt* — spec is below baseline at every tested batch: spec b1/2/4/8 =
36.7/26.2/87.0/225.5 vs baseline 90.6/106.0/263.4/592.8). The 27B↔16B arch gap is
the deepest root of 186-vs-94, on top of single-vs-batch. Consistent with the
per-position acceptance decay (87→16%) and "no optimization beat baseline".

### #3 — modal_mtp potential vs measured speed

The modal-mtp design predicts ~25–60% compute savings and 176.6 tok/s
(W4A16+MTP5), but the actual vLLM benchmark measured **3.2 tok/s** because the
draft path runs without CUDA graphs. Resolved by clearly distinguishing
*potential* (once draft-mode graphs are captured — a "to build" item) from
*measured-today* in `modal-self-speculative.md` and the README results table.

### #4 — `speculative-draft-override.patch` vs `speculative-dual-mode.patch`

Two patches touch `config/speculative.py` for the same concern (don't hijack a
standalone draft model into the MTP conversion). The draft-override is the
simpler/earlier form (gate on `mtp_num_hidden_layers is not None`); dual-mode is
the class-flag (`_skip_mtp_conversion`) form. Noted in
`vllm-patches-and-strategies.md` as the superseding pair; both copied.

### #5 — modal patches duplicated across repos

`modal-mtp/patches/0{1,2,3}-*.patch` overlap with qwen-ops's `qwen3_next.patch`,
`qwen3_5-shadow-state.patch`, `modal_mtp.patch` (same hooks, slightly different
versions — qwen-ops adds persistent snapshot buffers and the BatchDescriptor
draft_mode). Both copied (modal ones prefixed `modal-`) and the relationship is
noted in the patches table so no detail is lost.

### Minor: per-position acceptance rounding

`vllm-patches-benchmarks.md` gives 87/68/54/39/28/21/16%; `07-FRESH-INSTALL.md`
gives 85/68/54/40/31%. Both preserved verbatim in `README.md` with a note that
they round slightly differently.

### Minor: `01-SERVER-STATUS.md` placement

Describes the RTX 5090 "captain" box but lives under `deploy/gh200/` in the
source tree. Preserved as-is with a note in `deployment.md` and the deploy README.

---

## Losslessness verification (iter 2)

Re-enumerated every source file in the domain and verified each is present in
`MTP/04` (byte-identical copy where applicable, confirmed with `diff -q`) and/or
faithfully distilled. **Result: lossless. No missing files, no missing numbers.**
Only enhancements were made (GH200 arch finding, captured code constants); no
content was wrong enough to require correction.

### Copied artifacts — byte-identical (verified `diff -q`, iter 2)

| Source group | Files | MTP/04 dest | Check |
|---|---|---|---|
| `modal-mtp/` py + log | diagnose_divergence, validate_draft_accuracy, validate_extended, validate_extended_output.log, test_partial_skip (5) | `optimizations/` | identical |
| `modal-mtp/modal_mtp.py` | 1 | `optimizations/modal_mtp_proposer_reference.py` | identical (also ≡ `qwen-ops/vllm/optimizations/modal_mtp.py`) |
| `modal-mtp/patches/0{1,2,3}` | 3 | `patches/modal-0{1,2,3}-*.patch` | identical |
| `qwen-ops/vllm/patches/` | 14 (incl apply.sh) | `patches/` | all identical |
| `qwen-ops/vllm/optimizations/` | 14 | `optimizations/` | all identical |
| `qwen-ops/vllm/microgreens/` | 4 | `optimizations/microgreens/` | all identical |
| `qwen-ops/vllm/scripts/` | 3 | `optimizations/scripts/` | all identical |
| `qwen-ops/quantization/autoawq-qwen35/` | 6 | `quantization/` | all identical |
| `qwen-ops/deploy/{gh200,rtx5090}/` | 8 | `deploy/` | all identical |

### Distilled docs — re-verified against sources (iter 2)

| Source | MTP/04 doc | Verification |
|---|---|---|
| `modal-mtp/README.md` + `docs/design.md` (≡ `designs/modal-mtp-design.md`, `diff -q` confirms byte-identical) + `debug_indexerror.md` | `modal-self-speculative.md` | 3:1 pattern, attn indices, 77 MB/req snapshot, 125/193/176.6/1800+ tok/s, 3.2 tok/s measured, test_partial_skip 6 configs + ALL_ATTN_INDICES, IndexError root cause + Fix Option 1 — all match sources exactly |
| `modal-mtp/README.md` (4 evolving sections) + `findings/modal-mtp-precision-divergence.md` + `validate_extended_output.log` | `precision-divergence.md` | Test matrix CPU-FP32 100% (5/5), CPU-FP16 100% (12/12=1146/1146 — log confirms, incl haiku=46 tok), GPU-FP16 ~15%, GPU-BF16 ~50% (4/8); stage-2 "precision" framing labeled SUPERSEDED; 150 MB / +73 MB FP32-state; kernel names `causal_conv1d`/`chunk_gated_delta_rule` — all sourced & correct |
| `benchmarks/vllm-patches-benchmarks.md` | `README.md`, `vllm-patches-and-strategies.md` | 186/1030/27/3.2/5/139/186/174/47 + per-pos 87/68/54/39/28/21/16 — exact match |
| `benchmarks/autoawq-benchmarks.md` | `quantization.md`, `README.md` | GPTQ 151/51%/347/19.5 GB vs AWQ 77/31%/313/18.6 GB — exact match |
| `designs/` 8 GPU docs | `gpu-designs.md` | re-read all incl 3 long ones (TWO-GRAPH 451L, EAGLE-PR 320L, GH200 255L); all key numbers/decisions present; GH200 arch finding deepened (see thread #2) |
| 16 patches (13 vLLM + 3 modal) + ~15 strategy .py | `vllm-patches-and-strategies.md` | every module opened; doc entries accurate; captured constants added (PLV env defaults, probe layers, sibling-head env) |

### Iter-2 enhancements (not corrections — originals were accurate)

- `gpu-designs.md` #1: added FLOP-math proof that the GH200 doc measures a
  ~16B-class model (3584 hidden), not the 27B; added spec-batch table
  (36.7/26.2/87.0/225.5) and telemetry/launch-overhead numbers.
- `gpu-designs.md` #8: added repro model (Qwen3.5-MoE-A3B), eagle.py line range,
  "7 gh queries, 0 results".
- `vllm-patches-and-strategies.md`: surfaced code constants for
  `partial_layer_verify` (PLV_EXIT_LAYER=32, FULL_VERIFY_INTERVAL=8,
  AGREEMENT_THRESHOLD=0.90), `early_verify_probe` (HIDDEN_DIM=5120,
  VOCAB_SIZE=248320, PROBE_LAYERS=[3,7,11,15,31,47]), `native_multi_head`
  (env vars + 849 MB/~1 ms swap).

### Minor source-vs-code inconsistencies noted (left as-is, both preserved)

- `deltanet_adjuster.py` docstring: "~453M" (line 13/196) vs "~440M" (line 195)
  for a full MTP step — code's own minor inconsistency; doc uses 453M (primary).
- MTP-head swap size: `gpu-designs.md` #2 keeps the design doc's "~600 MB /
  0.1–0.3 ms"; the strategies table keeps the code's "849 MB / ~1 ms". Different
  size assumptions; both retained.
- per-position acceptance rounds two ways (87/68/54/39/28/21/16 vs
  85/68/54/40/31) — already noted above, both kept verbatim.
