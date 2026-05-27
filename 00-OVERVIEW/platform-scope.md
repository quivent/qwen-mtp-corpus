# Platform scope & portability — read before assuming "where to run"

> **TL;DR for agents:** This corpus is about a **portable technique**, not a machine.
> The Apple **M4 Max** appears throughout as a **reference measurement platform** — it is
> where much of the early profiling happened, *not* a required target and *not* "the dev
> box." Do **not** default to an M4 (or any specific machine) when acting on this corpus.
> Production-class work belongs on **datacenter GPU** (the GH200 results are the
> highest-throughput numbers here). Any `~/...` local path or SSH host you see is a
> historical/author-machine artifact, not an instruction.

## What is portable vs. what is hardware-specific

| Portable — applies on any backend | Hardware-specific — a measurement, not a recipe |
|---|---|
| The MTP head exists in the checkpoint and is a built-in self-drafter (`01-ARCHITECTURE/`) | tok/s figures, the 25.1 ms bandwidth floor, dispatch-barrier timings (M4 Max) |
| Self-speculative draft/verify on a hybrid DeltaNet+attention model (`05-THEORY/`) | Metal kernel-fusion wins/losses, `mx.compile` behavior (Apple GPU) |
| Split-recurrence rollback for irreversible recurrent state (`05-THEORY/`) | flashinfer GDN JIT, CUDA-graph capture, W4A16 throughput (NVIDIA) |
| Chained-recurrent MTP + confidence gating (the "recipe", `02-LLAMACPP/`) | which exact GPU/host an endpoint ran on |
| HF→GGUF tensor mapping, weight extraction (`01-ARCHITECTURE/`) | the `~/mlx-fork`, `~/qwen-ops`, `~/models` layouts (one author's machine) |

The findings transfer; the **numbers are anchored to the box they were measured on** and
are labeled as such. Re-measure on your own hardware before trusting a throughput figure.

## The platforms in this corpus, by role

| Platform | Role | Where |
|---|---|---|
| **vLLM on datacenter GPU (GH200 / RTX 5090)** | **Production / highest throughput.** 186→1030 tok/s (batch 1→8) on GH200; a persistent single-box endpoint on RTX 5090. Start here if you want to actually serve the model. | [`04-VLLM-GPU/`](../04-VLLM-GPU/) |
| **llama.cpp** | Portable CPU/GPU inference; where the MTP spec path, the cache-bookkeeping bug, and the 1.99× recipe were worked out. Builds anywhere llama.cpp builds. | [`02-LLAMACPP/`](../02-LLAMACPP/) |
| **MLX (Apple Silicon, M4 Max)** | **Reference platform.** Where the architecture was reverse-engineered and the bandwidth/kernel analysis was done. Apple-Silicon-only by nature; treat its numbers as reference data. | [`03-MLX/`](../03-MLX/) |

> The llama.cpp and MLX measurements happened to run on an M4 Max because that was the
> author's machine at the time. That is a fact about the *measurement*, not a constraint
> on the *work*. llama.cpp runs on Linux/NVIDIA/CPU just as well; the technique is the point.

## A note on machine-specific references

Throughout the corpus you may see local home-directory paths (`~/mlx-fork/...`,
`~/qwen-ops/...`, `~/models/...`) and, in the deploy docs, host/SSH placeholders. These are
**the original author's machine layout**, kept for provenance. They are either (a) labeled
as external/author-machine, (b) overridable CLI defaults (see
[`../06-TOOLING/README.md`](../06-TOOLING/README.md)), or (c) redacted before publishing
(the deploy SSH host/port). None of them are a target to operate on. The full
external-reference list is in [`../99-LEDGER/coverage-report.md`](../99-LEDGER/coverage-report.md).

## Related
- [`results.md`](results.md) — the per-platform numbers, with each platform's role marked
- [`the-big-picture.md`](the-big-picture.md) — the portable arc, start to finish
- [`../README.md`](../README.md) — the corpus front door
