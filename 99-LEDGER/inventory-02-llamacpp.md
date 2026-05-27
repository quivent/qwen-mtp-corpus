# Inventory — 02-LLAMACPP (Agent B)

Provenance ledger for the llama.cpp track. Every source file consumed → MTP destination
→ status.

## Status legend
- **distilled** — content absorbed into a distilled MTP doc
- **copied** — byte-for-byte copied into MTP/
- **merged-into-X** — folded into doc X
- **referenced** — left in place, referenced by path (not copied)
- **deferred** — owned by another agent's domain

---

## Source READMEs / docs consumed

| Source file | MTP destination | Status |
|---|---|---|
| `qwen-mtp-llamacpp/README.md` | `02-LLAMACPP/README.md`, `infrastructure-patches.md` | distilled |
| `qwen-mtp-optimizations/README.md` | `02-LLAMACPP/optimization-variants.md`, `the-recipe.md` | distilled |
| `qwen-mtp-research/README.md` | `02-LLAMACPP/the-bug.md`, `the-recipe.md`, `integration-plan.md`, `README.md` | distilled |
| `qwen-mtp-research/docs/the-recipe.md` | `02-LLAMACPP/the-recipe.md` (canonical home) | distilled (near-verbatim, all numbers preserved) |
| `qwen-mtp-research/docs/integration-plan.md` | `02-LLAMACPP/integration-plan.md` | distilled |
| `qwen-mtp-research/docs/per-position-heads.md` | `02-LLAMACPP/integration-plan.md` (headlines only) | merged-into-integration-plan + **deferred** to `05-THEORY-AND-DESIGNS/` (deep design) & `01-ARCHITECTURE/` (tensor layout) |
| `qwen-mtp-research/docs/tensor-layout.md` | — | deferred to `01-ARCHITECTURE/` (tensor domain) |
| `qwen-mtp-research/docs/mlx-reference.md` | — | deferred to `03-MLX/` |
| `qwen-mtp-research/docs/the-bug.md` | `02-LLAMACPP/the-bug.md` | **MISSING IN SOURCE** — referenced by README but file does not exist; bug story reconstructed from README + patch-11 commit message |
| `qwen-mtp-research/scripts/build_training_data.py` | — | deferred to `01-ARCHITECTURE/` / `05-THEORY-AND-DESIGNS/` (per-position-head training) |
| `qwen-mtp-research/scripts/train_per_position_heads.py` | — | deferred (same) |
| `qwen-mtp-research/patches/{graph,inference,loader}-stub.patch` | — | referenced in `integration-plan.md` Phase 4; deferred to `05-THEORY-AND-DESIGNS/` (design stubs, not the working series) |
| `qwen-mtp-research/docs/hybrid-deltanet-notes.md` | (named in README "what's in this repo") | **NOT PRESENT** in source docs/ — content captured via README into `README.md` "hybrid challenge" + `optimization-variants.md` |
| `qwen-mtp-research/docs/methodology.md` | (named in README) | **NOT PRESENT** in source docs/ — methodology bullets captured into `the-bug.md` lessons |
| `qwen-mtp-research/scripts/bench-honest.sh` | (named in README) | **NOT PRESENT** in source scripts/ — referenced only |
| `qwen-mtp-research/scripts/compare-decode.py` | (named in README) | **NOT PRESENT** in source scripts/ — referenced only |

## Patch series consumed → copied

Canonical copy source = `qwen-ops/llamacpp/` (verified **byte-identical** to the satellite
repos `qwen-mtp-llamacpp/patches/` and `qwen-mtp-optimizations/patches/` via `diff -q`).

| Source dir | MTP destination | Files | Status |
|---|---|---|---|
| `qwen-ops/llamacpp/infrastructure/*` (= `qwen-mtp-llamacpp/patches/*`) | `02-LLAMACPP/patches/infrastructure/` | 15 `.patch` + `00-base.txt` = 16 | copied (commit messages also distilled into `infrastructure-patches.md`) |
| `qwen-ops/llamacpp/optimizations/*` (= `qwen-mtp-optimizations/patches/*`) | `02-LLAMACPP/patches/optimizations/` | 9 `.patch` | copied (commit messages also distilled into `optimization-variants.md`) |
| `qwen-ops/llamacpp/tensor-mapping/*` | — | (01-qwen35-tensor-load.diff, 02-qwen35-graph-tensors.diff, README.md) | deferred to `01-ARCHITECTURE/` (tensor domain) |

### Infrastructure patches (16, all copied)
`00-base.txt`, `0000`..`0004` (base MTP layer), `01`..`11` (qwen35 port + the rollback
fix). Apply against upstream parent commit `9c7911f4f`.

### Optimization patches (9, all copied)
`01` adaptive chain (WINNER), `02` debug-verify, `03` drift-refresh, `04`
predictive-hidden, `05` perturbed-ensemble, `06`–`07` branching tree, `08` ensemble
fast-path (BROKEN), `09` stacked-hidden-noise (NEGATIVE).

## Fork MTP source consumed → copied

| Source file (in `llama-mtp/`) | MTP destination | Status |
|---|---|---|
| `src/models/qwen35.cpp` | `02-LLAMACPP/source/qwen35.cpp` | copied |
| `src/models/qwen3next.cpp` | `02-LLAMACPP/source/qwen3next.cpp` | copied |
| `common/speculative-mtp.cpp` | `02-LLAMACPP/source/speculative-mtp.cpp` | copied |
| `common/speculative-mtp.h` | `02-LLAMACPP/source/speculative-mtp.h` | copied |
| `examples/mtp-speculative/mtp-speculative.cpp` | `02-LLAMACPP/source/mtp-speculative.cpp` | copied |
| `examples/mtp-speculative/CMakeLists.txt` | `02-LLAMACPP/source/mtp-speculative.CMakeLists.txt` | copied (renamed to avoid collision) |
| `src/llama-arch.*`, `llama-model.cpp`, `llama-hparams.h`, `llama-graph.h`, `llama-context.*`, `llama-memory-snapshot.*` | — | referenced by path in `source/README.md` (MTP hooks captured via patch series, not copied as whole files) |
| rest of `llama-mtp/` (383 MB upstream) | — | referenced (NOT copied, per conventions) |

---

## Files produced (this domain)

| File | Role |
|---|---|
| `02-LLAMACPP/README.md` | Track narrative: model-won't-load → correct → 1.99×; honest perf; hybrid challenge |
| `02-LLAMACPP/infrastructure-patches.md` | 16 infra patches, ordered + explained; patch 11 highlighted |
| `02-LLAMACPP/optimization-variants.md` | 9 variants; pre/post-fix honesty; status table |
| `02-LLAMACPP/the-bug.md` | Cache-bookkeeping bug story; 6 agents; 3 lessons |
| `02-LLAMACPP/the-recipe.md` | **Canonical** 1.99× recipe; env vars; 5-prompt table; cost breakdown; levers |
| `02-LLAMACPP/integration-plan.md` | 6-phase per-position-heads plan w/ spend gates |
| `02-LLAMACPP/source/README.md` | Source file map + env-var list |
| `99-LEDGER/inventory-02-llamacpp.md` | this file |

---

## Contradictions & evolving threads

1. **"MLX uses per-position trained heads" → CORRECTED to "single head + chained
   recurrence".** The early belief was that MLX's 1.68×/1.73× came from DeepSeek-V3-style
   per-position trained heads. `qwen-mtp-research/README.md` and `docs/the-recipe.md`
   **correct this**: the MLX checkpoint contains exactly **one** MTP block (same single
   head as llama.cpp); the speedup is a *runtime* strategy (chained recurrent application
   + 0.8B companion + confidence gate, **zero training cost**). Per-position heads remain
   a *separate*, training-based **alternative** path (the integration plan), not what MLX
   does. **Flagged here; deep resolution owned by `05-THEORY-AND-DESIGNS/` (Theory agent).**

2. **Pre-fix vs post-fix numbers.** Most variant "wins" (1.16×, 1.72×, 2.5×, drift-refresh
   "10× accept jump", predictive-hidden "+75pt accept") were measured on **degraded text**
   before the cache-bookkeeping bug (patch 11) was fixed. The README/optimization-variants
   honesty table marks these "(pre-fix)". Only **01 adaptive chain (1.99×, post-fix)**,
   **08 ensemble fast-path (broken, post-fix)**, and **09 stacked-noise (0.58–0.69×,
   post-fix)** have trustworthy post-fix numbers. Preserved as evolving thread, not flattened.

3. **K>1 chaining: patch 05 vs the recipe.** Infra patch 05's commit message concludes
   Qwen3.5's single head is "architecturally limited to K=1" (K=2/K=3 accept identical to
   pre-fix → chaining is out-of-distribution). The later **recipe** (`the-recipe.md`)
   nonetheless wins at **K=2** — because confidence *gating* trims the wasted chained passes
   and the bug fix made the measurement honest. Not a strict contradiction: patch 05 says
   *naive* unconditional chaining doesn't help; the recipe adds the gate. Both preserved.

4. **13.98 vs 15.0 tok/s for K=2/thresh=0.85.** The 5-prompt **mean** is 13.98 tok/s
   (recipe + optimization README run tables); the K-sweep "sweet spot" row lists **15.0**
   tok/s (single-prompt). Both reported verbatim; noted inline in `the-recipe.md`.

5. **Authorship attribution: "Karpathy's original port".** Infra patch 05's commit message
   attributes the original K-step `prev_hidden` shortcut to "Karpathy's original port
   (commit 83babcae7)" — but patch 02 (commit `83babcae7`) is authored by `quivent` in the
   git metadata. Likely a colloquial/narrative attribution in the message, not a literal
   committer. Noted, not resolved.

6. **`MTP_FORCE_AR` vs `MTP_VERIFY_FORCE_AR`.** Two distinct flags: `MTP_FORCE_AR`
   (patch 08) disables drafting entirely → plain greedy AR decoder (byte-identical to
   llama-simple). `MTP_VERIFY_FORCE_AR` (patch 09) keeps drafting but forces the *verify*
   DeltaNet through an in-graph AR loop. Easy to conflate; disambiguated in
   `infrastructure-patches.md` and `source/README.md`.

7. **`docs/the-bug.md` referenced but absent.** `qwen-mtp-research/README.md` links to
   `docs/the-bug.md` and lists `docs/hybrid-deltanet-notes.md`, `docs/methodology.md`,
   `scripts/bench-honest.sh`, `scripts/compare-decode.py` in its "what's in this repo"
   section — **none of these files exist** in the source repo. Their content survives only
   via the README narrative, which has been distilled into `the-bug.md` (bug + 3 lessons),
   `README.md` (hybrid notes), and `the-bug.md`/methodology bullets. **Gap flagged.**

---

## Losslessness verification (iter 2)

Full file-by-file enumeration of every source I own: `qwen-mtp-llamacpp/`,
`qwen-mtp-optimizations/`, the llama.cpp-relevant bits of `qwen-mtp-research/`, plus the
two orphaned `qwen-inference-lab/logs/` server logs. **Status legend** as above.

### `qwen-mtp-llamacpp/`

| Source file | Captured where | Status |
|---|---|---|
| `README.md` | `02-LLAMACPP/README.md`, `infrastructure-patches.md` | distilled (all numbers preserved) |
| `LICENSE` | — | referenced (MIT/Apache boilerplate; not corpus content) |
| `docs/index.html` | — | duplicate-of `README.md` (HTML landing-page render; numbers 17.90/7.64/1.99× all already captured) |
| `patches/00-base.txt` | `02-LLAMACPP/patches/infrastructure/00-base.txt` | copied (byte-identical) |
| `patches/0000`..`0004`, `01`..`11` (15 `.patch`) | `02-LLAMACPP/patches/infrastructure/` | copied + commit msgs distilled into `infrastructure-patches.md` (verified against real diffs iter 2) |

### `qwen-mtp-optimizations/`

| Source file | Captured where | Status |
|---|---|---|
| `README.md` | `02-LLAMACPP/optimization-variants.md`, `the-recipe.md` | distilled |
| `LICENSE` | — | referenced (boilerplate) |
| `docs/index.html` | — | duplicate-of `README.md` (HTML render; "Six variants" is a marketing subset of the 9 — all 9 captured) |
| `patches/01`..`09` (9 `.patch`) | `02-LLAMACPP/patches/optimizations/` | copied + commit msgs distilled into `optimization-variants.md` (verified against real diffs iter 2) |

### `qwen-mtp-research/` (llama.cpp-relevant bits I own)

| Source file | Captured where | Status |
|---|---|---|
| `README.md` | `02-LLAMACPP/README.md`, `the-bug.md`, `the-recipe.md`, `integration-plan.md` | distilled (all numbers preserved) |
| `LICENSE` | — | referenced (boilerplate) |
| `docs/index.html` | — | duplicate-of `README.md` (HTML render; numbers 17.90/13.98/7.02/1.99× captured) |
| `docs/the-recipe.md` | `02-LLAMACPP/the-recipe.md` (canonical) | distilled near-verbatim |
| `docs/integration-plan.md` | `02-LLAMACPP/integration-plan.md` | distilled |
| `docs/per-position-heads.md` | `05-THEORY-AND-DESIGNS/per-position-heads-design.md` + headlines in `integration-plan.md` | deferred to 05-THEORY (deep design) |
| `docs/tensor-layout.md` | `01-ARCHITECTURE/` | deferred to 01-ARCHITECTURE |
| `docs/mlx-reference.md` | `03-MLX/` | deferred to 03-MLX |
| `patches/graph-stub.patch` | `02-LLAMACPP/patches/per-position-heads-stubs/graph-stub.patch` | **copied (iter 2)** — was referenced-only; now byte-identical copy + README |
| `patches/inference-stub.patch` | `02-LLAMACPP/patches/per-position-heads-stubs/inference-stub.patch` | **copied (iter 2)** |
| `patches/loader-stub.patch` | `02-LLAMACPP/patches/per-position-heads-stubs/loader-stub.patch` | **copied (iter 2)** |
| `scripts/build_training_data.py` | `05-THEORY-AND-DESIGNS/training/build_training_data.py` | deferred to 05-THEORY (confirmed present iter 2) |
| `scripts/train_per_position_heads.py` | `05-THEORY-AND-DESIGNS/training/train_per_position_heads.py` | deferred to 05-THEORY (confirmed present iter 2) |
| `docs/the-bug.md`, `docs/hybrid-deltanet-notes.md`, `docs/methodology.md`, `scripts/bench-honest.sh`, `scripts/compare-decode.py` | `the-bug.md`, `README.md` | **DO NOT EXIST in source** — content survives via README distillation only (see thread #7) |

### `qwen-inference-lab/logs/` (orphaned server logs — gap fixed iter 2)

| Source file | Captured where | Status |
|---|---|---|
| `logs/llama_cpp_server.log` | `02-LLAMACPP/logs/llama_cpp_server.log` + numbers in `benchmarks.md` | **copied (iter 2, md5-verified)** — was deferred/never captured |
| `logs/llama_cpp_server_nomtp.log` | `02-LLAMACPP/logs/llama_cpp_server_nomtp.log` + numbers in `benchmarks.md` | **copied (iter 2, md5-verified)** — unadvertised baseline sibling |
| `logs/adaptive_mtp_{vanilla,huihui}.log` | — | deferred to 03-MLX (MLX runs, not llama.cpp) |

### Iter-2 closure summary

- **Orphaned server logs**: both copied + folded into new `benchmarks.md`. Key numbers:
  MTP-head accept ~2% (2/98), 0.8B companion accept ~49% (47/95), MTP+draft decode
  18.5–21.8 tok/s vs no-draft baseline 20.81 tok/s → server-mode spec is
  throughput-neutral (consistent with the CLI K=1=0.43× finding).
- **Stub patches**: the only remaining "referenced-only" llama.cpp artifacts → now copied.
- **Patch docs**: verified against real diffs (0001/03/06/07/09/10/11 infra; 01/05/06/07/08/09
  opt). Enriched patch 10 (`scoped_force_ar` RAII), `source/README.md` (exact stats field
  names + full draft→verify→rollback code-flow section).
- **No unique fact, number, env var, primitive, or artifact in this domain is now lost.**
  Remaining "missing in source" items (thread #7) are genuinely absent upstream, not
  capture gaps.
