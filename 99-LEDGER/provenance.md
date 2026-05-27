# Provenance — Merged Source → Destination Map

**Owner: Agent 3C (99-LEDGER).** The lossless audit trail: every source repo → its files →
where each landed in `MTP/`. Merges the six per-domain inventories (`inventory-0[1-6]-*.md`)
into one navigable map. Status vocabulary: **distilled** (prose absorbed into a doc),
**copied** (verbatim artifact), **duplicate-of** (byte-identical, collapsed to one canonical
home), **referenced** (left in place, cited by path), **not copied** (boilerplate/binary,
enumerated in `coverage-report.md` §5).

> **`MTP/` is self-contained.** The 383 MB `llama-mtp/` upstream llama.cpp fork tree is
> **referenced by path**, not copied — only its **~9 MTP-specific files** were copied into
> `02-LLAMACPP/source/` (see §0). The MTP *hooks* in the larger upstream files
> (`llama-arch.*`, `llama-model.cpp`, `llama-hparams.*`, `llama-graph.h`, `llama-context.*`,
> `llama-memory-snapshot.*`) were captured via the patch series + distilled into 01/02 docs,
> not copied as whole upstream files.

Coverage headline (see `coverage-report.md`): **211 source files in scope, 211 losslessly
represented (100%), 0 missing.** 11 source repos covered below (10 conventions repos +
the `llama-mtp/` fork exception).

---

## 0. `llama-mtp/` — the llama.cpp fork (383 MB, referenced; ~9 files copied)

Excluded from the 211 count (mostly-upstream). Only the MTP-specific source was copied.

| Source (`llama-mtp/`) | Destination | Status | Owner |
|---|---|---|---|
| `src/models/qwen35.cpp` | `02-LLAMACPP/source/qwen35.cpp` | copied (+ distilled into 01/02 arch+anatomy docs) | B (+A reads) |
| `src/models/qwen3next.cpp` | `02-LLAMACPP/source/qwen3next.cpp` | copied (+ distilled, GDN cross-check) | B (+A reads) |
| `common/speculative-mtp.cpp` | `02-LLAMACPP/source/speculative-mtp.cpp` | copied | B |
| `common/speculative-mtp.h` | `02-LLAMACPP/source/speculative-mtp.h` | copied | B |
| `examples/mtp-speculative/mtp-speculative.cpp` | `02-LLAMACPP/source/mtp-speculative.cpp` | copied | B |
| `examples/mtp-speculative/CMakeLists.txt` | `02-LLAMACPP/source/mtp-speculative.CMakeLists.txt` (renamed) | copied | B |
| `src/llama-arch.{h,cpp}` | `01-ARCHITECTURE/*` + `02-LLAMACPP/source/README.md` | distilled / referenced (enums, KV templates, tensor lists) | A/B |
| `src/llama-hparams.{h,cpp}` | `01-ARCHITECTURE/qwen35-hybrid-architecture.md`, `tensor-layout-hf-to-gguf.md` | distilled (`n_embd_s/r`, `rope_sections`, ssm dims) | A |
| `src/llama-model.cpp` | `01-ARCHITECTURE/*` | distilled (loader fixes, type detection, recurrent mask, `n_rot`) | A |
| `src/llama-graph.h`, `llama-context.*`, `llama-memory-snapshot.*` | — | referenced by path (hooks via patches) | B |
| rest of tree (383 MB upstream) | — | referenced (not copied, per conventions) | — |

---

## 1. `qwen-mtp-tensors/` (3 in-scope) — Owner A (01-ARCHITECTURE)

| Source | Destination | Status |
|---|---|---|
| `README.md` | `01-ARCHITECTURE/tensor-layout-hf-to-gguf.md`, `qwen35-hybrid-architecture.md`, `mtp-head-anatomy.md` | distilled (canonical "tensor archaeology") |
| `diffs/01-qwen35-tensor-load.diff` | `01-ARCHITECTURE/tensor-diffs/01-qwen35-tensor-load.diff` | copied + distilled |
| `diffs/02-qwen35-graph-tensors.diff` | `01-ARCHITECTURE/tensor-diffs/02-qwen35-graph-tensors.diff` | copied + distilled |
| `docs/index.html` | — | not copied (rendered HTML; markdown is source of truth) |
| `LICENSE` | — | not copied (license only) |
| `docs/tensor-layout.md` | — | **never existed** in source (README references it; only `index.html` present) — dangling ref, content lives in `qwen-mtp-research/docs/tensor-layout.md` |

## 2. `qwen-mtp-llamacpp/` (18 in-scope) — Owner B (02-LLAMACPP)

| Source | Destination | Status |
|---|---|---|
| `README.md` | `02-LLAMACPP/README.md`, `infrastructure-patches.md` | distilled (all numbers preserved) |
| `patches/00-base.txt` | `02-LLAMACPP/patches/infrastructure/00-base.txt` | copied (byte-identical) |
| `patches/0000`..`0004`, `01`..`11` (15 `.patch`) | `02-LLAMACPP/patches/infrastructure/` | copied + commit msgs distilled into `infrastructure-patches.md` |
| `docs/index.html` | — | duplicate-of `README.md` (HTML render) — not copied |
| `LICENSE` | — | not copied |

> Canonical copy source for the patch files = `qwen-ops/llamacpp/infrastructure/`, verified
> byte-identical to `qwen-mtp-llamacpp/patches/` via `diff -q`.

## 3. `qwen-mtp-optimizations/` (10 in-scope) — Owner B (02-LLAMACPP)

| Source | Destination | Status |
|---|---|---|
| `README.md` | `02-LLAMACPP/optimization-variants.md`, `the-recipe.md` | distilled |
| `patches/01`..`09` (9 `.patch`) | `02-LLAMACPP/patches/optimizations/` | copied + commit msgs distilled |
| `docs/index.html` | — | duplicate-of `README.md` — not copied |
| `LICENSE` | — | not copied |

> 9 variants: 01 adaptive chain (WINNER, 1.99× post-fix), 02 debug-verify, 03 drift-refresh,
> 04 predictive-hidden, 05 perturbed-ensemble, 06–07 branching tree, 08 ensemble fast-path
> (BROKEN), 09 stacked-hidden-noise (NEGATIVE).

## 4. `qwen-mtp-research/` (11 in-scope) — Owners A / B / E

| Source | Destination | Status |
|---|---|---|
| `README.md` | `02-LLAMACPP/{README,the-bug,the-recipe,integration-plan}.md`, `01-ARCHITECTURE/*`, `05-THEORY/per-position-heads-design.md` | distilled (multi-owner) |
| `docs/the-recipe.md` | `02-LLAMACPP/the-recipe.md` (**canonical**) | distilled near-verbatim (all numbers) |
| `docs/integration-plan.md` | `02-LLAMACPP/integration-plan.md` | distilled |
| `docs/per-position-heads.md` | `05-THEORY-AND-DESIGNS/per-position-heads-design.md` (canonical) + headlines in `02-LLAMACPP/integration-plan.md` + arch slivers in `01-ARCHITECTURE/` | distilled (multi-owner) |
| `docs/tensor-layout.md` | `01-ARCHITECTURE/tensor-layout-hf-to-gguf.md` | distilled (N=4 generalization, storage est., enorm/hnorm naming) |
| `docs/mlx-reference.md` | `03-MLX/the-journey.md` (Phase 7), `03-MLX/README.md`, `05-THEORY/per-position-heads-design.md` §0 | distilled (multi-owner) |
| `patches/graph-stub.patch` | `02-LLAMACPP/patches/per-position-heads-stubs/graph-stub.patch` | **copied (iter 2)** — was referenced-only |
| `patches/inference-stub.patch` | `02-LLAMACPP/patches/per-position-heads-stubs/inference-stub.patch` | **copied (iter 2)** |
| `patches/loader-stub.patch` | `02-LLAMACPP/patches/per-position-heads-stubs/loader-stub.patch` | **copied (iter 2)** |
| `scripts/build_training_data.py` | `05-THEORY-AND-DESIGNS/training/build_training_data.py` | copied (duplicate-of `qwen-ops/training/`) + distilled |
| `scripts/train_per_position_heads.py` | `05-THEORY-AND-DESIGNS/training/train_per_position_heads.py` | copied + distilled |
| `LICENSE` | — | not copied |
| `docs/the-bug.md`, `docs/hybrid-deltanet-notes.md`, `docs/methodology.md`, `scripts/bench-honest.sh`, `scripts/compare-decode.py` | (content via README → `02-LLAMACPP/the-bug.md`, `README.md`) | **never existed** in source — dangling refs (see §11); content survives via README distillation |

## 5. `mlx-qwen-mtp/` (7 in-scope) — Owner C (03-MLX)

| Source | Destination | Status |
|---|---|---|
| `README.md` | `03-MLX/{README,mtp-self-speculative-mlx,kernel-fusion}.md` + arch slivers → `01-ARCHITECTURE/*` | distilled (multi-owner) |
| `src/mtp_head.py` | `03-MLX/src/mtp_head.py` | copied (canonical) + distilled into `01-ARCHITECTURE/mtp-head-anatomy.md` |
| `src/generate.py` | `03-MLX/src/generate.py` | copied |
| `src/fused_kernels_t2.py` | `03-MLX/src/fused_kernels_t2.py` | copied |
| `src/extract_weights.py` | `03-MLX/src/extract_weights.py` + `01-ARCHITECTURE/extract/extract_weights.py` | copied (canonical; ≡ `qwen-ops/mlx/extract_weights.py`) |
| `src/__init__.py` | `03-MLX/src/__init__.py` | copied |
| `src/__pycache__/*.pyc` | — | not copied (bytecode) |
| `pyproject.toml`, `LICENSE` | — | not copied (Apache-2.0 noted in `03-MLX/README.md`) |

## 6. `qwen-inference-lab/` (14 in-scope) — Owner C (03-MLX); 2 logs → B

| Source | Destination | Status |
|---|---|---|
| `README.md` | `03-MLX/README.md`, `the-journey.md` | distilled |
| `docs/TIMELINE.md` | `03-MLX/the-journey.md` (canonical), `kernel-fusion.md`, `mtp-self-speculative-mlx.md` | distilled |
| `docs/BANDWIDTH_ANALYSIS.md` | `03-MLX/bandwidth-and-dispatch.md` | merged (canonical) |
| `docs/DISPATCH_BARRIER_PROFILING.md` | `03-MLX/bandwidth-and-dispatch.md`, `kernel-fusion.md` | merged (canonical) |
| `docs/HUIHUI_ABLITERATED.md` | `03-MLX/huihui-abliterated.md` | distilled (canonical) |
| `docs/RESEARCH_FRONTIERS.md` | `03-MLX/bandwidth-and-dispatch.md` (frontiers) + `05-THEORY/research-frontiers.md` | distilled — **cross-cutting** (MLX-HW here, general theory in 05) |
| `kernels/fused_gdn.py` | `03-MLX/src/fused_gdn.py` | copied (also authoritative for DeltaNet state shape → 01) |
| `benchmarks/bench_v7.py` | `03-MLX/src/bench_v7.py` | copied |
| `benchmarks/extract_mtp_huihui.py` | `03-MLX/src/extract_mtp_huihui.py` + `01-ARCHITECTURE/extract/extract_mtp_huihui.py` | copied |
| `logs/adaptive_mtp_vanilla.log` | `03-MLX/src/logs/adaptive_mtp_vanilla.log` | copied (51.1 tok/s run) |
| `logs/adaptive_mtp_huihui.log` | `03-MLX/src/logs/adaptive_mtp_huihui.log` | copied (49.5 tok/s run) |
| `logs/llama_cpp_server.log` | `02-LLAMACPP/logs/llama_cpp_server.log` + `benchmarks.md` | copied (iter 2, md5-verified) — **Owner B** (llama.cpp artifact) |
| `logs/llama_cpp_server_nomtp.log` | `02-LLAMACPP/logs/llama_cpp_server_nomtp.log` + `benchmarks.md` | copied (iter 2) — **Owner B** |
| `pyproject.toml`, `LICENSE` | — | not copied |

## 7. `recurrent-rollback/` (9 in-scope) — Owner E (05-THEORY-AND-DESIGNS)

| Source | Destination | Status |
|---|---|---|
| `README.md` | `05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md` (+ theory doc §3) | distilled |
| `docs/TECHNIQUE.md` (499 L) | `recurrent-rollback-technique.md`, `speculative-decoding-on-hybrid-models.md` §2/§8, `research-frontiers.md` §0/§A.3–A.6 | distilled (lossless) |
| `src/__init__.py` | `05-THEORY-AND-DESIGNS/recurrent-rollback-src/src/__init__.py` | copied |
| `src/split_recurrence.py` | `recurrent-rollback-src/src/split_recurrence.py` | copied |
| `src/delta_net_rollback.py` | `recurrent-rollback-src/src/delta_net_rollback.py` | copied |
| `examples/__init__.py` (0 B) | `recurrent-rollback-src/examples/__init__.py` | copied |
| `examples/mtp_speculative_decode.py` | `recurrent-rollback-src/examples/mtp_speculative_decode.py` | copied |
| `examples/simple_example.py` | `recurrent-rollback-src/examples/simple_example.py` | copied |
| `pyproject.toml`, `LICENSE` | — | not copied |

> Note: the rollback repo's DeltaNet state shape `(14,256,256)` is **wrong**; corrected to
> `[48,128,128]`+conv in `01-ARCHITECTURE/` (open-questions R1). The mechanism is unaffected.

## 8. `modal-mtp/` (12 in-scope) — Owner D (04-VLLM-GPU)

| Source | Destination | Status |
|---|---|---|
| `README.md` | `04-VLLM-GPU/{modal-self-speculative,precision-divergence,README}.md` | distilled |
| `docs/design.md` | `04-VLLM-GPU/modal-self-speculative.md` | distilled (≡ `qwen-ops/research/designs/modal-mtp-design.md`, collapsed) |
| `debug_indexerror.md` | `04-VLLM-GPU/modal-self-speculative.md` (IndexError section) | distilled |
| `modal_mtp.py` | `04-VLLM-GPU/optimizations/modal_mtp_proposer_reference.py` | copied + distilled |
| `diagnose_divergence.py` | `04-VLLM-GPU/optimizations/diagnose_divergence.py` | copied + distilled |
| `validate_draft_accuracy.py` | `04-VLLM-GPU/optimizations/validate_draft_accuracy.py` | copied + distilled |
| `validate_extended.py` | `04-VLLM-GPU/optimizations/validate_extended.py` | copied + distilled |
| `validate_extended_output.log` | `04-VLLM-GPU/optimizations/validate_extended_output.log` | copied (decisive 1146/1146 evidence) |
| `test_partial_skip.py` | `04-VLLM-GPU/optimizations/test_partial_skip.py` | copied + distilled |
| `patches/01-skip-attention.patch` | `04-VLLM-GPU/patches/modal-01-skip-attention.patch` | copied |
| `patches/02-draft-mode-and-state.patch` | `04-VLLM-GPU/patches/modal-02-draft-mode-and-state.patch` | copied |
| `patches/03-model-runner-draft-loop.patch` | `04-VLLM-GPU/patches/modal-03-model-runner-draft-loop.patch` | copied |

## 9. `qwen-ops/` (127 in-scope) — the earlier partial consolidation (mostly duplicates)

`qwen-ops/` mirrors the nine satellite repos; almost all content is a duplicate of a satellite
file already chosen as canonical. **Unique content = the Go CLI only.** Grouped by subtree:

### 9a. `cmd/` + `go.{mod,sum}` (8) — Owner 2F (06-TOOLING) — THE ONLY UNIQUE qwen-ops CONTENT

| Source | Destination | Status |
|---|---|---|
| `cmd/{main,serve,mtp,patches,status,download}.go` | `06-TOOLING/qwen-ops-cli/cmd/*.go` | copied (verbatim) + distilled into `06-TOOLING/README.md` |
| `go.mod` | `06-TOOLING/qwen-ops-cli/go.mod` | copied (needed to build) |
| `go.sum` | `06-TOOLING/qwen-ops-cli/go.sum` | copied (build checksums) |

### 9b. `vllm/` (35) — Owner D (04-VLLM-GPU) — **canonical copy** (Agent D used qwen-ops as source)

| Source group | Files | Destination | Status |
|---|---|---|---|
| `vllm/patches/*` (incl `apply.sh`) | 14 | `04-VLLM-GPU/patches/` | copied + distilled (eagle, qwen3_next, shadow-state, gdn-*, speculative-*, recurrent-rollback, modal_mtp, int8-embedding, gh200-strip-torch-dep, llmcompressor-conv1d) |
| `vllm/optimizations/*.py` | 14 | `04-VLLM-GPU/optimizations/` | copied + distilled (adaptive_mtp, modal_mtp, partial_layer_verify, plv_bench, plv_layer60_bench, early_verify_probe, deltanet_transplant{,_w4a16}, cascade_mtp_corrective, deltanet_adjuster, enhanced_mtp_proposer, native_multi_head, sibling_sequential, selective_state_snapshot) |
| `vllm/microgreens/*` | 4 | `04-VLLM-GPU/optimizations/microgreens/` | copied + distilled (mtp_clone, mtp_diversity_train, sibling_mtp_proposer, __init__) |
| `vllm/scripts/*` | 3 | `04-VLLM-GPU/optimizations/scripts/` | copied + distilled (vllm-tree-spec.sh, bench-tok-s.py, quantize_deltanet.py) |

### 9c. `quantization/autoawq-qwen35/` (6) — Owner D — **canonical copy**

| Source | Destination | Status |
|---|---|---|
| `qwen3_5.py`, `inject_mtp_weights.py`, `__init__.py.patch`, `auto.py.patch`, `base.py.patch`, `quantizer.py.patch` | `04-VLLM-GPU/quantization/` | copied + distilled into `quantization.md` |

### 9d. `deploy/` (8) — Owner D — **canonical copy**

| Source | Destination | Status |
|---|---|---|
| `deploy/gh200/{deploy.sh,01-SERVER-STATUS.md,07-FRESH-INSTALL.md,08-GH200-AGENT-INSTALL.md}` | `04-VLLM-GPU/deploy/gh200/` | copied + distilled into `deployment.md` |
| `deploy/rtx5090/{05-NIXOS-GUIDE.md,nixos-captain-configuration.nix,vllm-serve.sh,vllm-watchdog.sh}` | `04-VLLM-GPU/deploy/rtx5090/` | copied + distilled |

### 9e. `training/` (2) — Owner E — **canonical copy**

| Source | Destination | Status |
|---|---|---|
| `training/build_training_data.py` | `05-THEORY-AND-DESIGNS/training/build_training_data.py` | copied (≡ `qwen-mtp-research/scripts/`) + distilled |
| `training/train_per_position_heads.py` | `05-THEORY-AND-DESIGNS/training/train_per_position_heads.py` | copied + distilled |

### 9f. `research/designs/` (13) — Owners D / E — canonical + collapses

| Source | Destination | Status |
|---|---|---|
| `GH200-COMPUTE-BANDWIDTH-ANALYSIS.md` | `04-VLLM-GPU/gpu-designs.md` #1 | distilled (mislabeled-model finding, open-questions R4) |
| `CUDAGRAPH-WEIGHT-SWAP.md` | `gpu-designs.md` #2 | distilled |
| `TWO-GRAPH-CUDA-DISPATCH.md` | `gpu-designs.md` #3 | distilled |
| `PROPOSE-TREE-CUDAGRAPH-ANALYSIS.md` | `gpu-designs.md` #4 | distilled |
| `MTP-TREE-CONFIG-PATH.md` | `gpu-designs.md` #5 | distilled |
| `PLV-FULL-ATTN-BOUNDARIES.md` | `gpu-designs.md` #6 | distilled |
| `DELTANET-WEIGHT-TRANSPLANT.md` | `gpu-designs.md` #7 | distilled |
| `EAGLE-PR-DESCRIPTIONS.md` | `gpu-designs.md` #8 | distilled |
| `CASCADE-MTP-TRAINING.md` | `05-THEORY-AND-DESIGNS/cascade-mtp-training.md` | distilled |
| `per-position-heads.md` | `05-THEORY-AND-DESIGNS/per-position-heads-design.md` | duplicate-of `qwen-mtp-research/docs/per-position-heads.md` (collapsed) |
| `modal-mtp-design.md` | `04-VLLM-GPU/modal-self-speculative.md` | duplicate-of `modal-mtp/docs/design.md` (collapsed) |
| `integration-plan.md` | `02-LLAMACPP/integration-plan.md` | duplicate-of `qwen-mtp-research/docs/integration-plan.md` |

### 9g. `research/findings/` (9) — mixed canonical / duplicate

| Source | Destination | Status |
|---|---|---|
| `modal-mtp-precision-divergence.md` | `04-VLLM-GPU/precision-divergence.md` | distilled (canonical) |
| `tensor-layout.md` | `01-ARCHITECTURE/tensor-layout-hf-to-gguf.md` | duplicate-of `qwen-mtp-research/docs/tensor-layout.md` |
| `mlx-reference.md` | `03-MLX/` / `05-THEORY/per-position-heads-design.md` §0 | duplicate-of `qwen-mtp-research/docs/mlx-reference.md` |
| `TIMELINE.md`, `BANDWIDTH_ANALYSIS.md`, `DISPATCH_BARRIER_PROFILING.md`, `HUIHUI_ABLITERATED.md`, `RESEARCH_FRONTIERS.md` | (canonical = `qwen-inference-lab/docs/`) | duplicate-of (byte-identical, collapsed) — `RESEARCH_FRONTIERS.md` also = `05-THEORY/research-frontiers.md` source |

### 9h. `research/benchmarks/` (3) — Owners D / C

| Source | Destination | Status |
|---|---|---|
| `vllm-patches-benchmarks.md` | `04-VLLM-GPU/README.md`, `vllm-patches-and-strategies.md` | distilled |
| `autoawq-benchmarks.md` | `04-VLLM-GPU/quantization.md`, `README.md` | distilled |
| `inference-lab-benchmarks.md` | `03-MLX/` | distilled (deferred to Agent C, M4 Max) |

### 9i. `llamacpp/` (29) — Owners B / A — byte-identical duplicates

| Source group | Destination | Status |
|---|---|---|
| `llamacpp/infrastructure/*` | (canonical = `02-LLAMACPP/patches/infrastructure/`) | duplicate-of `qwen-mtp-llamacpp/patches/` |
| `llamacpp/optimizations/*` | (canonical = `02-LLAMACPP/patches/optimizations/`) | duplicate-of `qwen-mtp-optimizations/patches/` |
| `llamacpp/tensor-mapping/{README.md, 01-*.diff, 02-*.diff}` | (canonical = `01-ARCHITECTURE/`) | duplicate-of `qwen-mtp-tensors/` |

### 9j. `mlx/` (6) — Owner C — byte-identical duplicates

| Source | Destination | Status |
|---|---|---|
| `mlx/{mtp_head,generate,fused_kernels_t2,extract_weights,__init__}.py` | (canonical = `mlx-qwen-mtp/src/` → `03-MLX/src/`) | duplicate-of (byte-identical) |
| `mlx/pyproject.toml` | — | duplicate-of `mlx-qwen-mtp/pyproject.toml` (not copied) |

### 9k. `validation/` (7) — Owners A / C / D — byte-identical duplicates (closed iter 2)

| Source | Byte-identical to | Canonical home |
|---|---|---|
| `validation/fused_gdn.py` | `qwen-inference-lab/kernels/fused_gdn.py` | `03-MLX/src/fused_gdn.py` |
| `validation/bench_v7.py` | `qwen-inference-lab/benchmarks/bench_v7.py` | `03-MLX/src/bench_v7.py` |
| `validation/extract_mtp_huihui.py` | `qwen-inference-lab/benchmarks/extract_mtp_huihui.py` | `01-ARCHITECTURE/extract/` & `03-MLX/src/` |
| `validation/diagnose_divergence.py` | `modal-mtp/diagnose_divergence.py` | `04-VLLM-GPU/optimizations/diagnose_divergence.py` |
| `validation/validate_draft_accuracy.py` | `modal-mtp/validate_draft_accuracy.py` | `04-VLLM-GPU/optimizations/validate_draft_accuracy.py` |
| `validation/validate_extended.py` | `modal-mtp/validate_extended.py` | `04-VLLM-GPU/optimizations/validate_extended.py` |
| `validation/test_partial_skip.py` | `modal-mtp/test_partial_skip.py` | `04-VLLM-GPU/optimizations/test_partial_skip.py` |

### 9l. `README.md` + binary (2)

| Source | Destination | Status |
|---|---|---|
| `qwen-ops/README.md` | distilled across 00–05 (results, directory tour, key insight, bug summary, quickstart); cross-ref by 06 | distilled |
| `qwen-ops/qwen-ops` (3.9 MB Mach-O arm64 binary) | — | not copied (build artifact, regenerable from `cmd/*.go`) |

---

## 10. Intentionally not copied (enumerated — not losses)

| File(s) | Reason |
|---|---|
| 4× `pyproject.toml` (mlx-qwen-mtp, qwen-inference-lab, recurrent-rollback, qwen-ops/mlx) | Python packaging metadata |
| `qwen-ops/qwen-ops` | Compiled binary (regenerable) |
| 7× `LICENSE`, 4× `docs/index.html`, 3× `*.pyc` | Filtered from the 211 scope (boilerplate / rendered HTML / bytecode) |
| `llama-mtp/` 383 MB upstream tree | Referenced by path; ~9 MTP-specific files copied (§0) |

> `go.mod`/`go.sum` ARE copied (required to build the only-unique-in-qwen-ops Go CLI) — listed
> under §9a, not here.

## 11. Dangling source references (referenced but never existed — not losses)

Named in source READMEs/docs but absent from every source repo. Listed so a future reader does
not hunt for content that was never written.

| Referenced path | Referenced by | Reality |
|---|---|---|
| `qwen-mtp-research/docs/the-bug.md` | `qwen-mtp-research/README.md` | Never existed. Bug story reconstructed from README + patch-11 commit msg → `02-LLAMACPP/the-bug.md`. |
| `qwen-mtp-research/docs/hybrid-deltanet-notes.md` | README | Never existed. Content via README → `02-LLAMACPP/README.md` + `optimization-variants.md`. |
| `qwen-mtp-research/docs/methodology.md` | README | Never existed. Methodology bullets → `02-LLAMACPP/the-bug.md` lessons. |
| `qwen-mtp-research/scripts/bench-honest.sh` | README | Never existed in source `scripts/`. Referenced only. |
| `qwen-mtp-research/scripts/compare-decode.py` | README | Never existed in source `scripts/`. Referenced only. |
| `qwen-mtp-tensors/docs/tensor-layout.md` | `qwen-mtp-tensors/README.md` | Not present as markdown (`docs/` has only `index.html`). Canonical text in `qwen-mtp-research/docs/tensor-layout.md` → `01-ARCHITECTURE/`. |
| `~/mlx-fork/fused_gdn.py`, `~/mlx-fork/mtp_weights{,_vanilla}.safetensors` | tensor diff-02 commit msg; `mlx-reference.md` | Author's machine, outside corpus. In-corpus equivalent: `mlx-qwen-mtp/src/mtp_head.py` / `03-MLX/src/`. |
| `stacked_v2.py`, `adaptive_mtp.py` (MLX), `collect_hidden_states.py`, `cascade_mtp_corrective.py` | `mlx-reference.md`, inventories 03/05 | Live in sibling `parallel-mtp-voting` / `~/optimizations/` projects, not in corpus. Technique described from docs + the captured `.log` files (51.1 tok/s run). |
| `~/optimizations/qwen-mtp-inference/`, ANE `.mlpackage`, `quivent/mlx-fused-qmv` metallib | inventory 03 | External to corpus; referenced only. |

---

*Provenance: merged by Agent 3C from `inventory-0[1-6]-*.md` + `coverage-report.md`. Covers
all 10 conventions source repos + the `llama-mtp/` fork. All copied-artifact destinations and
canonical homes verified present this iteration.*
