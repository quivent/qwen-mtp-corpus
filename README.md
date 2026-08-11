<div align="center">

```
  ___  __  __ _____ ____  
 / _ \|  \/  |_   _|  _ \ 
| | | | |\/| | | | | |_) |
| |_| | |  | | | | |  __/ 
 \__\_\_|  |_| |_| |_|    
    C O R P U S
```

**Distilled, lossless corpus of MTP speculative decoding.**

*Synthesized from 10 source repositories into one navigable whole for Qwen3.5-27B.*

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

</div>

---

## 📑 Table of Contents
- [🎯 Overview](#-overview)
- [📊 Headline Results](#-headline-results)
- [🗺️ The Map](#️-the-map)
- [📖 Quick Start Guide](#-quick-start-guide)
- [🏗️ How This Corpus Was Built](#️-how-this-corpus-was-built)
- [📦 Self-containment](#-self-containment)

---

## 🎯 Overview

> **Build v1.0 · 2026-05-27** · lossless **211/211** source files · **0** broken links · audit **0.98/1.00** ✅
> Built via `/iterate` (5 iterations × 5 parallel Opus agents). Full report: [`99-LEDGER/AUDIT-REPORT.md`](99-LEDGER/AUDIT-REPORT.md).

This is the **distilled, lossless, reorganized corpus** of the Multi-Token Prediction (MTP) speculative-decoding research for **Qwen3.5-27B**. 

The work spans three inference stacks: **vLLM**, **llama.cpp**, and **MLX**. Every unique fact, number, tensor name, env var, dead-end, and code artifact was carried across, de-duplicated to a single source of truth, and re-layered so a researcher can read the story top-down and an engineer can rebuild the result bottom-up.

> [!WARNING]
> **Platform scope (read before assuming "where to run").** The technique is hardware-agnostic; the *numbers* are anchored to the box each was measured on. The **M4 Max is a reference measurement platform, not a target** — don't default to it. Production work belongs on datacenter GPU. Full portability picture: [`00-OVERVIEW/platform-scope.md`](00-OVERVIEW/platform-scope.md).

---

## 📊 Headline Results

| Platform | Role | Hardware | Best tok/s | Speedup | Method |
|---|---|---|---:|---|---|
| **vLLM** | production | GH200 480GB | **186** (batch=1) / **1030** (batch=8) | 5.54× batch | Stock MTP spec=7 |
| **vLLM** | single-box | RTX 5090 (32 GB) | **151** (single, 256 tok) | — (51% MTP accept) | GPTQ W4A16 + MTP=5 |
| **llama.cpp** | portable | reference: M4 Max (546 GB/s) | **13.98** | 1.99× over K=1 vanilla (= 0.78× of plain decode 17.90) | Chained recurrent MTP + confidence gating (`MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`) |
| **MLX** | reference | M4 Max (546 GB/s) | **51.1** | 1.73× over stock 29.5 | Adaptive MTP confidence chain + batch verify |

> [!NOTE]
> Each speedup is **relative to that platform's own baseline** — the rows are not comparable across platforms. The full cited breakdown is in [`00-OVERVIEW/results.md`](00-OVERVIEW/results.md).

---

## 🗺️ The Map

| Domain | One line | Entry |
|---|---|---|
| **00-OVERVIEW** | Synthesis layer: the story, the numbers, the vocabulary, the provenance map | [`00-OVERVIEW/`](00-OVERVIEW/) |
| **01-ARCHITECTURE** | What Qwen3.5-27B *is* — the 64-layer 3:1 hybrid, the MTP head | [`01-ARCHITECTURE/qwen35-hybrid-architecture.md`](01-ARCHITECTURE/qwen35-hybrid-architecture.md) |
| **02-LLAMACPP** | The C++ port: 16 infra patches, 9 optimization variants | [`02-LLAMACPP/README.md`](02-LLAMACPP/README.md) |
| **03-MLX** | Apple Silicon: kernel fusion, split-recurrence rollback | [`03-MLX/README.md`](03-MLX/README.md) |
| **04-VLLM-GPU** | Datacenter GPU: stock MTP serving, AWQ/GPTQ quant, deploy runbooks | [`04-VLLM-GPU/README.md`](04-VLLM-GPU/README.md) |
| **05-THEORY-AND-DESIGNS** | Why spec-decode is hard on a recurrent hybrid | [`05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md) |
| **06-TOOLING** | The `qwen-ops` Go CLI — download → patch → serve → bench | [`06-TOOLING/README.md`](06-TOOLING/README.md) |
| **99-LEDGER** | Provenance proof, the losslessness coverage report | [`99-LEDGER/provenance.md`](99-LEDGER/provenance.md) |

---

## 📖 Quick Start Guide

- **The story** — start at [`00-OVERVIEW/the-big-picture.md`](00-OVERVIEW/the-big-picture.md).
- **The vocabulary** — [`00-OVERVIEW/glossary.md`](00-OVERVIEW/glossary.md) defines every term.
- **The numbers** — [`00-OVERVIEW/results.md`](00-OVERVIEW/results.md) is the single authoritative benchmark table.
- **Where everything came from** — [`00-OVERVIEW/corpus-map.md`](00-OVERVIEW/corpus-map.md) maps the 10 source repos.
- **Where to run / what's portable** — [`00-OVERVIEW/platform-scope.md`](00-OVERVIEW/platform-scope.md).

---

## 🏗️ How This Corpus Was Built

Built over **5 iterations** by a team of parallel distillation agents under a shared contract:
1. Skeleton and first-pass distillation
2. Deep distillation and artifact consolidation
3. Ruthless cross-domain de-duplication
4. Synthesis layer addition
5. Final audit

The build is **lossless and verified**: 211 of 211 in-scope source files are represented (100%), 0 missing. Full proof in [`99-LEDGER/coverage-report.md`](99-LEDGER/coverage-report.md).

---

## 📦 Self-containment

`MTP/` is self-contained: all unique small artifacts are copied in under their domain folders. The single exception is the **383 MB upstream llama.cpp fork tree**, referenced by path rather than copied.

---
*Front door for the `MTP/` corpus. The synthesis layer only links and summarizes; each domain doc owns its own numbers.*
