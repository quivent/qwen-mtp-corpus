# MTP — Multi-Token Prediction speculative decoding for Qwen3.5-27B

> **Build v1.0 · 2026-05-27** · lossless **211/211** source files · **0** broken links · audit **0.98/1.00** ✅
> Built via `/iterate` (5 iterations × 5 parallel Opus agents). Full report: [`99-LEDGER/AUDIT-REPORT.md`](99-LEDGER/AUDIT-REPORT.md).

This is the **distilled, lossless, reorganized corpus** of the Multi-Token Prediction
(MTP) speculative-decoding research for **Qwen3.5-27B** across three inference stacks —
**llama.cpp**, **MLX** (Apple Silicon), and **vLLM** (datacenter GPU). It was synthesized
from **10 source repositories** into one navigable whole: every unique fact, number,
tensor name, env var, dead-end, and code artifact was carried across, de-duplicated to a
single source of truth, and re-layered so a researcher can read the story top-down and an
engineer can rebuild the result bottom-up. Qwen3.5-27B ships a trained MTP head that is a
built-in self-drafter; this corpus is the record of getting that head to actually
accelerate decoding on a hybrid (DeltaNet + attention) model that breaks the usual
speculative-decoding assumptions.

## Headline results (best per platform)

| Platform | Hardware | Best tok/s | Speedup | Method |
|---|---|---:|---|---|
| **llama.cpp** | M4 Max (546 GB/s) | **13.98** | 1.99× over K=1 vanilla (= 0.78× of plain decode 17.90) | Chained recurrent MTP + confidence gating (`MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`) |
| **MLX** | M4 Max (546 GB/s) | **51.1** | 1.73× over stock 29.5 | Adaptive MTP confidence chain + batch verify |
| **vLLM** | GH200 480GB | **186** (batch=1) / **1030** (batch=8) | 5.54× batch | Stock MTP spec=7 |
| **vLLM** | RTX 5090 (32 GB) | **151** (single, 256 tok) | — (51% MTP accept) | GPTQ W4A16 + MTP=5 |

> Each speedup is **relative to that platform's own baseline** — the rows are not
> comparable across platforms. An honest caveat lives in the detail: on this bandwidth-
> bound hybrid model, single-request MTP spec decode often *lost* to plain decode
> (llama.cpp's best is still only 0.78× of plain), and on vLLM none of ~14 experimental
> strategies beat stock MTP. The full cited breakdown — every configuration measured,
> the apparent-contradiction disambiguations, and the hardware reference — is in
> [`00-OVERVIEW/results.md`](00-OVERVIEW/results.md).

## The map — what is where

| Domain | One line | Entry |
|---|---|---|
| **00-OVERVIEW** | Synthesis layer: the story, the numbers, the vocabulary, the provenance map | [`00-OVERVIEW/`](00-OVERVIEW/) |
| **01-ARCHITECTURE** | What Qwen3.5-27B *is* — the 64-layer 3:1 hybrid, the MTP head ("layer 64"), HF→GGUF tensor mapping, weight extraction | [`01-ARCHITECTURE/qwen35-hybrid-architecture.md`](01-ARCHITECTURE/qwen35-hybrid-architecture.md) |
| **02-LLAMACPP** | The C++ port: 16 infra patches, 9 optimization variants, the bug that ate a session, and the 1.99× recipe | [`02-LLAMACPP/README.md`](02-LLAMACPP/README.md) |
| **03-MLX** | Apple Silicon: kernel fusion, split-recurrence rollback, the bandwidth wall, 29.5 → 51.1 tok/s | [`03-MLX/README.md`](03-MLX/README.md) |
| **04-VLLM-GPU** | Datacenter GPU: stock MTP serving, modal self-speculation, the precision saga, AWQ/GPTQ quant, deploy runbooks | [`04-VLLM-GPU/README.md`](04-VLLM-GPU/README.md) |
| **05-THEORY-AND-DESIGNS** | Why spec-decode is hard on a recurrent hybrid, the rollback technique, per-position heads, training, research frontiers | [`05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md) |
| **06-TOOLING** | The `qwen-ops` Go CLI — download → patch → serve → bench → status, one box | [`06-TOOLING/README.md`](06-TOOLING/README.md) |
| **99-LEDGER** | Provenance proof, the losslessness coverage report, and the open-questions registry | [`99-LEDGER/provenance.md`](99-LEDGER/provenance.md) |

## If you read one thing

- **The story** — start at [`00-OVERVIEW/the-big-picture.md`](00-OVERVIEW/the-big-picture.md).
  The whole arc, readable straight through, linking out to the canonical docs for depth.
- **The vocabulary** — [`00-OVERVIEW/glossary.md`](00-OVERVIEW/glossary.md) defines every
  term (MTP, DeltaNet, eh_proj, split-recurrence, confidence gating, the O(1) ceiling…)
  with a pointer to the doc that owns it.
- **The numbers** — [`00-OVERVIEW/results.md`](00-OVERVIEW/results.md) is the single
  authoritative benchmark table, every value cited to its source.
- **Where everything came from** — [`00-OVERVIEW/corpus-map.md`](00-OVERVIEW/corpus-map.md)
  maps the 10 source repos to the 6 domains and answers "where do I find X?".

## How this corpus was built

Built over **5 iterations** by a team of parallel distillation agents (one per domain),
under a shared contract ([`_meta/CONVENTIONS.md`](_meta/CONVENTIONS.md)): iteration 1 laid
the skeleton and first-pass distillation; iterations 2–3 did deep distillation, artifact
consolidation, and ruthless cross-domain de-duplication; iteration 4 added this synthesis
layer (overview + front door + losslessness ledger); iteration 5 is the final audit.

The build is **lossless and verified**: every source file was enumerated, distilled or
copied, and accounted for — **211 of 211 in-scope source files are represented (100%), 0
missing**. The full proof, including the de-dup collapses and the handful of
intentionally-not-copied boilerplate files, is in
[`99-LEDGER/coverage-report.md`](99-LEDGER/coverage-report.md); the exhaustive
source → destination audit trail is in [`99-LEDGER/provenance.md`](99-LEDGER/provenance.md).

## Self-containment

`MTP/` is self-contained: all unique small artifacts (patches, scripts, kernels, logs, the
MTP-specific fork source files) are **copied in** under their domain folders. The single
exception is the **383 MB upstream llama.cpp fork tree** (`../llama-mtp/`), which is
mostly stock llama.cpp and is **referenced by path** rather than copied — only its ~9
MTP-specific source files were copied into [`02-LLAMACPP/source/`](02-LLAMACPP/source/).
See [`99-LEDGER/provenance.md`](99-LEDGER/provenance.md) §0 for that exception.

---
*Front door for the `MTP/` corpus. The synthesis layer (this README + `00-OVERVIEW/`) only
links and summarizes; each domain doc owns its own numbers.*
