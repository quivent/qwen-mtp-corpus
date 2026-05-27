# MTP Corpus — Final Audit Report

**Date:** 2026-05-27
**Build method:** `/iterate` — 5 iterations × 5 parallel Opus agents, slow-iterate / write-frequently / consolidate / synthesize.
**Result:** ✅ **COMPLETE — composite audit score 0.98 / 1.00** (threshold 0.90).

---

## 1. What was built

`MTP/` is the fully reorganized, distilled, **lossless, non-confusing** corpus synthesized
from **10 source repositories** of research on **Multi-Token Prediction (MTP) speculative
decoding for Qwen3.5-27B** across three inference stacks (llama.cpp, MLX, vLLM) plus
quantization, deployment, training, and the operational CLI.

**Final size:** 2.7 MB · 36 directories · 66 markdown docs · 121 copied artifacts · 187 files total.

```
MTP/
├── README.md                  front door
├── 00-OVERVIEW/               the-big-picture · results · glossary · corpus-map
├── 01-ARCHITECTURE/           hybrid arch · MTP head · HF→GGUF tensors · weight extraction
├── 02-LLAMACPP/               infra patches · variants · the-bug · the-recipe · source/ · patches/ · logs/
├── 03-MLX/                    kernel fusion · bandwidth/dispatch · the-journey · self-spec · huihui · src/
├── 04-VLLM-GPU/               modal self-spec · precision-divergence · quant · deploy · gpu-designs · patches/
├── 05-THEORY-AND-DESIGNS/     spec-decode-on-hybrid · recurrent-rollback · per-position heads · training · frontiers
├── 06-TOOLING/                qwen-ops Go CLI
├── 99-LEDGER/                 coverage-report · provenance · open-questions · inventories · this report
└── _meta/                     CONVENTIONS · consistency-findings (build provenance)
```

---

## 2. Iteration history & convergence

| Iter | Focus | Agents | Key output |
|---|---|---|---|
| 1 | Skeleton + first-pass distillation + inventory | 5 (per domain) | 49 docs, 108 artifacts, 5 inventories, contradictions flagged |
| 2 | Deep distillation + gap-fill + losslessness check | 5 | Go CLI captured, server logs folded in, **DeltaNet state shape adjudicated**, GH200 mislabel proven, coverage report (211 files) |
| 3 | Consolidation + dedup + contradiction resolution | 5 | frontiers dedup, 49-term glossary, unified `results.md`, `open-questions.md` (13 resolved/3 open), `provenance.md`, coverage → 100% |
| 4 | Synthesis (front door + narrative + nav) | 5 | `README.md`, `the-big-picture.md`, `corpus-map.md`, 2 folder READMEs, broken-link fix, 3:1 dedup, ~40 "Related" footers |
| 5 | Audit gateway + targeted fixes + this report | 5 audit + orchestrator | 5-dimension independent audit, 6 surgical fixes, final report |

**Convergence pattern:** monotonic. Iteration 1 already produced deep drafts (Opus agents);
iterations 2–4 were additive completeness/clarity work with no regressions; iteration 5
found only minor editorial issues (all fixed). No retry loops were required.

---

## 3. Audit scores (5 independent read-only Opus auditors)

| Dimension | Score | Evidence |
|---|---|---|
| **Numerical accuracy** | **1.00** | 38 high-stakes claims sampled across all 6 domains; each traced to the *original source repo* (not another MTP doc); **38/38 exact match**, 0 mismatches. All derived speedups arithmetically re-checked (51.1/29.5=1.73×, 1030/186=5.54×, 13.98/7.02=1.99×, 13.98/17.90=0.78×). |
| **Losslessness** | **0.99** | Independently recounted **211/211** source files (per-repo breakdown reproduced exactly). 35+ copied artifacts `diff`-verified byte-identical; 9 sampled duplicate-collapses confirmed truly identical; **0 silent losses**; all 6 dangling source references confirmed genuinely absent (never existed). |
| **Clarity / non-confusion** | **0.96** | Fresh-reader walkthrough: all **8 navigation questions answered in ≤2 hops**, no dead-ends. All 4 spot-checked resolved contradictions present the answer first with superseded framing explicitly labeled. |
| **Artifact integrity** | **0.99** | **124/125** copied artifacts byte-identical to source (the 1 delta is the intentional "Gated Delta Net"→"Gated DeltaNet" terminology fix in a deploy doc); all patch series complete; all 6 fork source files byte-identical; **no 383M tree leakage** (2.7 MB total, no model/binary files). |
| **Cross-doc consistency** | **0.96** | 6/7 checks pass with resolved facts obeyed everywhere (no bare superseded claims survive). 1 flag (vLLM patch count "14" vs actual 16) — **fixed** this iteration. |
| **Composite** | **0.98** | Mean of the five. |

---

## 4. Issues found in audit → resolution

All fixed in iteration 5:

1. **vLLM patch count "14" → 16** (13 vLLM + 3 modal) in `04-VLLM-GPU/README.md`,
   `00-OVERVIEW/corpus-map.md`, `99-LEDGER/inventory-04-vllm-gpu.md`. (The "14 strategy
   modules" figure on README.md:95 is *correct* and was left as-is — modal validation py
   is listed separately.)
2. **Stale forward-reference** in `00-OVERVIEW/results.md` ("open-questions.md does not yet
   exist") → updated; the registry now exists and is linked.
3. **Off-by-one** in `99-LEDGER/coverage-report.md` (`research/designs/` 13 → 12).
4. **Future-tense punch-list** `_meta/consistency-findings.md` → STATUS banner added marking
   it addressed in iteration 4.

Whole-corpus link check after fixes: **541 internal links, 0 broken** (the only flagged
items are this report's own forward-reference, now created, and a literal `[text](target)`
inside the methodology prose).

---

## 5. Headline research findings preserved (the corpus's substance)

- **The model is its own drafter.** Qwen3.5-27B ships an MTP head (layer 64) that every
  framework strips on load; reverse-engineered → self-speculative decoding, no separate draft model.
- **Best measured results:** llama.cpp **13.98 tok/s** (1.99× over K=1 vanilla, still 0.78× of
  plain decode); MLX **51.1 tok/s** (1.73× over stock 29.5); vLLM GH200 **186→1030 tok/s**
  (batch=1→8, 5.54×); RTX 5090 **151 tok/s** (GPTQ W4A16 + MTP=5). See `00-OVERVIEW/results.md`.
- **The hard problem:** DeltaNet recurrent state is irreversible → standard spec-decode rollback
  breaks. Solved by **split-recurrence rollback** (zero-copy immutable-array state refs) +
  **chained-recurrent MTP with confidence gating**.
- **The adjudicated DeltaNet state shape:** `(num_v_heads=48, head_v_dim=128, head_k_dim=128)`
  = 786,432 elems/layer + `conv_state (3, 10240)`, ~77–78 MB/req BF16 (corrected a wrong
  source figure of (14,256,256)).
- **Three myths corrected:** (a) FP16 GPU divergence is **CUDA-kernel numerics, not precision**
  (CPU FP16 = 100%); (b) MLX's 1.73× is a **single chained head + confidence gating**, NOT
  trained per-position heads; (c) the GH200 "94 tok/s" doc is **mislabeled** — FLOP math proves
  it measured a ~16B model, not the 27B.
- **The bug that ate a session:** a one-line cache-bookkeeping double-write (`id_last = corr`)
  made every spec measurement look like noise. Lesson: validate output text, not just throughput.

The full resolved/open contradiction registry is in [`open-questions.md`](open-questions.md);
the lossless source→destination audit trail is in [`provenance.md`](provenance.md); the
file-by-file representation proof is in [`coverage-report.md`](coverage-report.md).

---

## 6. Verdict

✅ **COMPLETE.** The corpus meets its mandate: **lossless** (211/211 source files represented,
0 silent losses, artifacts byte-identical), **not confusing** (single source of truth per topic,
13 contradictions resolved, ≤2-hop navigation, layered for both researcher and reproducing
engineer), and **self-contained** (2.7 MB, artifacts copied in, the 383M upstream tree referenced
by path). Composite audit **0.98 ≥ 0.90 threshold**.
