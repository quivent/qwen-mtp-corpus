# Inventory — 06-TOOLING (Agent 2F)

Domain: the `qwen-ops` Cobra Go CLI — the operational umbrella tool. Not consumed
by any iteration-1 agent; this inventory closes that gap.

Owner folder: `MTP/06-TOOLING/`.

## Source → destination → status

| Source file (`qwen-ops/`) | MTP destination | Status |
|---|---|---|
| `cmd/main.go` | `06-TOOLING/qwen-ops-cli/cmd/main.go` | copied (verbatim) + distilled into `06-TOOLING/README.md` |
| `cmd/serve.go` | `06-TOOLING/qwen-ops-cli/cmd/serve.go` | copied + distilled (`serve` reference) |
| `cmd/mtp.go` | `06-TOOLING/qwen-ops-cli/cmd/mtp.go` | copied + distilled (`mtp apply` / `mtp bench`) |
| `cmd/patches.go` | `06-TOOLING/qwen-ops-cli/cmd/patches.go` | copied + distilled (`patches list` / `patches apply`) |
| `cmd/status.go` | `06-TOOLING/qwen-ops-cli/cmd/status.go` | copied + distilled (`status`) |
| `cmd/download.go` | `06-TOOLING/qwen-ops-cli/cmd/download.go` | copied + distilled (`download`) |
| `go.mod` | `06-TOOLING/qwen-ops-cli/go.mod` | copied (needed to build; module + cobra dep) |
| `go.sum` | `06-TOOLING/qwen-ops-cli/go.sum` | copied (needed to build; checksums only — boilerplate) |

## Files produced (this domain)

| File | Role |
|---|---|
| `06-TOOLING/README.md` | Distilled CLI reference: narrative on top, per-subcommand flag tables below; path-assumption map; cross-links to 02/04 |
| `06-TOOLING/qwen-ops-cli/README.md` | Origin note + build instructions for the copied Go sources |
| `06-TOOLING/qwen-ops-cli/cmd/*.go` | 6 Go sources (verbatim) |
| `06-TOOLING/qwen-ops-cli/{go.mod,go.sum}` | Go module metadata (verbatim, buildable) |
| `99-LEDGER/inventory-06-tooling.md` | this file |

## NOT copied (intentional)

| Source | Reason |
|---|---|
| `qwen-ops/qwen-ops` | Compiled Mach-O arm64 binary (3.9 MB) — build artifact, regenerable from `cmd/*.go`. |
| `qwen-ops/README.md` | Repo-level README of the *source consolidation* repo. Its content (results table, directory tour, key insight, the-bug summary, quickstart) is already distilled across domains 00–05 (results → `00-OVERVIEW`/`02-LLAMACPP`; insight + bug → `02`/`05`). Used as a cross-reference, not a tooling artifact. |
| All non-`cmd/` subtrees of `qwen-ops/` (`llamacpp/`, `vllm/`, `mlx/`, `deploy/`, `quantization/`, `research/`, `training/`, `validation/`) | Duplicates of the satellite repos; captured (de-duplicated to canonical homes) in domains 01–05. Enumerated in `coverage-report.md`. |

## Contradictions & evolving threads

1. **Model naming: `Qwen3.6-27B` (CLI) vs `Qwen3.5-27B` (rest of corpus).** The
   `qwen-ops` CLI defaults reference `Qwen3.6-27B` / `unsloth/Qwen3.6-27B-GGUF`,
   while every other domain and the conventions target Qwen3.5-27B. Treated as a
   naming drift in the later tool, not a separate model — the patches/tensors/arch
   it drives are the documented Qwen3.5 / Qwen3-Next MTP work. Recorded in
   `06-TOOLING/README.md`, not silently normalized.

2. **vLLM "optimizations" are `.py`, not `.diff`.** `patches.go` scans
   `~/qwen-ops/vllm/optimizations/` for `*.patch`/`*.diff`, but that directory
   holds Python strategy modules. The CLI finds zero diffs there; the optimization
   `.py` files are applied as source by the deploy scripts, not `git apply`.
   Noted in `06-TOOLING/README.md`.

3. **Hard-coded `~/qwen-ops/` patch paths.** The CLI reads patches from
   `~/qwen-ops/...`, not from the corpus. The path-assumption table in
   `06-TOOLING/README.md` maps each expected path to its corpus equivalent so the
   tool can be pointed at the de-duplicated copies.
