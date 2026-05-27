# Corpus Map — where everything came from, and where to find it

> The reader-facing provenance overview. **What** the 10 source repos were, **how** they
> collapsed into the 6 MTP domains, and a **"where do I find X?"** quick-reference. For the
> exhaustive file-by-file audit trail see [`../99-LEDGER/provenance.md`](../99-LEDGER/provenance.md);
> for the losslessness proof (211/211 source files represented) see
> [`../99-LEDGER/coverage-report.md`](../99-LEDGER/coverage-report.md).

---

## 1. The 10 source repos → 6 MTP domains

`MTP/` was synthesized from 10 sibling repositories under `/home/ubuntu/qwen27/`. Each row
is what that repo *contributed* and where it landed.

| Source repo | What it contributed | Landed in |
|---|---|---|
| `llama-mtp/` | The actual llama.cpp **fork** (383 MB, mostly upstream). Only its ~9 MTP-specific source files were copied; the tree is **referenced by path**. | [`02-LLAMACPP/source/`](../02-LLAMACPP/source/) + distilled into 01/02 |
| `qwen-mtp-llamacpp/` | The llama.cpp MTP **infrastructure**: 16 ordered patches (HF→GGUF→loader→graph→spec loop) + the narrative of getting the model to load. | [`02-LLAMACPP/`](../02-LLAMACPP/) (patches + `infrastructure-patches.md`) |
| `qwen-mtp-optimizations/` | The 9 llama.cpp **optimization variants** (adaptive chain = winner; plus the broken/negative ones, preserved). | [`02-LLAMACPP/optimization-variants.md`](../02-LLAMACPP/optimization-variants.md) |
| `qwen-mtp-tensors/` | The HF→GGUF **tensor archaeology**: what the converter strips, the `ssm_*` metadata names, the two load/graph diffs. | [`01-ARCHITECTURE/`](../01-ARCHITECTURE/) (`tensor-layout-hf-to-gguf.md`, `tensor-diffs/`) |
| `qwen-mtp-research/` | The research notes: **the-recipe**, the per-position-heads design, the integration plan, the tensor-layout text, and the (reconstructed) bug story. | spread across 01/02/05 |
| `mlx-qwen-mtp/` | The **MLX** (Apple Silicon) implementation: the MTP head reverse-engineered from weight shapes, fused kernels, the generate loop. | [`03-MLX/`](../03-MLX/) (`src/` + docs) |
| `qwen-inference-lab/` | The M4 Max **optimization journey**: TIMELINE, bandwidth analysis, dispatch-barrier profiling, the Huihui variant, the 51.1 tok/s logs. | [`03-MLX/`](../03-MLX/) (`the-journey.md`, `bandwidth-and-dispatch.md`, `huihui-abliterated.md`) |
| `recurrent-rollback/` | The **split-recurrence rollback** technique, generalized — TECHNIQUE.md + reference Python. | [`05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) + `recurrent-rollback-src/` |
| `modal-mtp/` | The **modal self-speculative** idea (skip the 16 attention layers to draft) + the **precision-divergence saga** + validation evidence. | [`04-VLLM-GPU/`](../04-VLLM-GPU/) (`modal-self-speculative.md`, `precision-divergence.md`) |
| `qwen-ops/` | An **earlier partial consolidation** of all the above (127 files). Mostly duplicates; its only *unique* content is the Go CLI and it served as the **canonical copy source** for the vLLM/quant/deploy artifacts. | unique CLI → [`06-TOOLING/`](../06-TOOLING/); rest → 04 (canonical) or de-duped |

Two things to note from the table: (1) the corpus has a **7th folder, `06-TOOLING`**,
which exists *only* because `qwen-ops`'s Go CLI was its one piece of unique content; and
(2) several repos are **multi-owner** — e.g. `qwen-mtp-research/README.md` distilled into
01, 02, and 05 — because their content genuinely spans domains.

## 2. The major de-dup collapses (at a glance)

The corpus is "not confusing" because duplicated content was collapsed to **one canonical
home**. The big ones:

- **`qwen-ops/` mirrored the satellites.** Of its 127 files, the Go CLI (8) is the only
  unique content. Its `llamacpp/` (29), `mlx/` (6), and `validation/` (7) subtrees are
  **byte-identical duplicates** (verified with `diff -q`) of the satellite repos that
  already owned that content. Its `vllm/` (35), `quantization/` (6), `deploy/` (8), and
  `training/` (2) subtrees were used as the **canonical copy source** for domain 04/05.
- **13 byte-identical docs collapsed.** `qwen-ops/research/{findings,designs}/` re-stated
  docs that lived in the satellites: TIMELINE, BANDWIDTH_ANALYSIS, DISPATCH_BARRIER_PROFILING,
  HUIHUI_ABLITERATED, RESEARCH_FRONTIERS, the-recipe, tensor-layout, per-position-heads,
  integration-plan, mlx-reference, modal-mtp-design — each collapsed to one home.
- **The two `eh_proj` / "the 3:1 pattern IS the draft schedule" re-explanations** were
  collapsed to canonical owners (`01-ARCHITECTURE/mtp-head-anatomy.md` and
  `04-VLLM-GPU/modal-self-speculative.md` respectively), with the journey narratives
  keeping a pointer rather than re-explaining.
- **Net result:** 56 files classified as duplicate-of (collapsed), 67 copied as verbatim
  artifacts, 81 distilled into prose — see [`../99-LEDGER/coverage-report.md`](../99-LEDGER/coverage-report.md) §1.

## 3. Where do I find X?

| I want… | Go to |
|---|---|
| **Where to run / what's portable** — M4 Max = reference, not a target; production = GPU | [`platform-scope.md`](platform-scope.md) |
| **The model architecture** — 64-layer 3:1 hybrid, DeltaNet state, MTP head, dims | [`01-ARCHITECTURE/`](../01-ARCHITECTURE/) |
| **Tensor shapes / HF→GGUF mapping** — what the converter strips, `ssm_*` names | [`01-ARCHITECTURE/tensor-layout-hf-to-gguf.md`](../01-ARCHITECTURE/tensor-layout-hf-to-gguf.md) |
| **The 1.99× recipe** (llama.cpp) — the env vars, the 5-prompt table, the K-sweep | [`02-LLAMACPP/the-recipe.md`](../02-LLAMACPP/the-recipe.md) |
| **The bug** — the cache-bookkeeping one-liner that ate a session | [`02-LLAMACPP/the-bug.md`](../02-LLAMACPP/the-bug.md) |
| **The patches** — 16 infra + 9 optimization (llama.cpp); 16 vLLM (13 vLLM + 3 modal) | [`02-LLAMACPP/patches/`](../02-LLAMACPP/patches/), [`04-VLLM-GPU/patches/`](../04-VLLM-GPU/patches/) |
| **Kernel fusion** (MLX Metal) — what fused, what *lost* to `mx.compile` | [`03-MLX/kernel-fusion.md`](../03-MLX/kernel-fusion.md) |
| **The bandwidth wall** — 13.7 GB / 546 GB/s = 25.1 ms floor, dispatch barriers | [`03-MLX/bandwidth-and-dispatch.md`](../03-MLX/bandwidth-and-dispatch.md) |
| **Deployment** — GH200 + RTX 5090/NixOS runbooks, exact commands | [`04-VLLM-GPU/deployment.md`](../04-VLLM-GPU/deployment.md) |
| **Quantization** — AWQ vs GPTQ W4A16 for the hybrid arch | [`04-VLLM-GPU/quantization.md`](../04-VLLM-GPU/quantization.md) |
| **The precision saga** — CPU vs GPU kernel numerics (not precision) | [`04-VLLM-GPU/precision-divergence.md`](../04-VLLM-GPU/precision-divergence.md) |
| **The theory** — why spec-decode is hard on a recurrent hybrid | [`05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md) |
| **Split-recurrence rollback** — the signature technique, generalized | [`05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) |
| **Per-position heads / training** — the design line (design-only) | [`05-THEORY-AND-DESIGNS/per-position-heads-design.md`](../05-THEORY-AND-DESIGNS/per-position-heads-design.md), [`05-THEORY-AND-DESIGNS/training-mtp-heads.md`](../05-THEORY-AND-DESIGNS/training-mtp-heads.md) |
| **Research frontiers** — the O(1) ceiling, open directions | [`05-THEORY-AND-DESIGNS/research-frontiers.md`](../05-THEORY-AND-DESIGNS/research-frontiers.md) |
| **The CLI** — `qwen-ops` download/patch/serve/bench/status | [`06-TOOLING/README.md`](../06-TOOLING/README.md) |
| **Provenance / audit** — every source file → where it landed | [`../99-LEDGER/provenance.md`](../99-LEDGER/provenance.md) |
| **Losslessness proof** — 211/211 represented, the de-dup math | [`../99-LEDGER/coverage-report.md`](../99-LEDGER/coverage-report.md) |
| **Open questions / contradictions registry** | [`../99-LEDGER/open-questions.md`](../99-LEDGER/open-questions.md) |
| **The numbers** | [`results.md`](results.md) |
| **The vocabulary** | [`glossary.md`](glossary.md) |
| **The story** | [`the-big-picture.md`](the-big-picture.md) |

## 4. Self-containment note

All unique small artifacts (patches, kernels, scripts, logs, the MTP-specific fork source)
are **copied into** their domain folders, so `MTP/` stands alone. The one exception is the
**383 MB `../llama-mtp/` upstream fork tree**, referenced by path (only its ~9 MTP-specific
files were copied — see [`../99-LEDGER/provenance.md`](../99-LEDGER/provenance.md) §0). A
handful of boilerplate files (4 `pyproject.toml`, the compiled `qwen-ops` binary, LICENSE/
HTML/bytecode) were intentionally not copied and are enumerated in
[`../99-LEDGER/coverage-report.md`](../99-LEDGER/coverage-report.md) §5.

---

## Related docs

- [`../README.md`](../README.md) — the corpus front door.
- [`the-big-picture.md`](the-big-picture.md) — the whole arc as one narrative.
- [`../99-LEDGER/provenance.md`](../99-LEDGER/provenance.md) — the exhaustive source → destination map.
- [`../99-LEDGER/coverage-report.md`](../99-LEDGER/coverage-report.md) — the losslessness proof.

*Reader-facing provenance map. The seed table is from
[`../_meta/CONVENTIONS.md`](../_meta/CONVENTIONS.md); the de-dup figures are from the
99-LEDGER coverage report.*
