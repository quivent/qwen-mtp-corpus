# MTP Corpus — Consistency Findings (Agent 3E, iteration 3)

> **STATUS: ADDRESSED in iteration 4.** This is a frozen iteration-3 punch-list,
> kept for build provenance. The items below (front-door `README.md`, the three
> `00-OVERVIEW/` synthesis files, the two folder READMEs, the broken anchor, the
> 3:1 dedup, terminology, and the "Related" footers) have all been completed. Read
> in past tense. Audit confirmation is in [`../99-LEDGER/AUDIT-REPORT.md`](../99-LEDGER/AUDIT-REPORT.md).

Read-only editorial/structural audit punch-list for the iteration-4 polish pass.
Scope swept: all domain docs in `00-OVERVIEW` (treated as in-progress), `01`–`06`.
**Excluded:** `99-LEDGER/*` (being rebuilt). Research contradictions are 99-LEDGER's
job and are **not** listed here — this is editorial/structural only.

Method: every `[text](target)` link extracted and target-existence checked by script
(176 internal links + the two new 00-OVERVIEW files); terminology axes swept with grep;
duplication concepts mapped doc-by-doc; folder entry-points and nav footers enumerated.

Priorities: **P1** = correctness / dangling link · **P2** = reader confusion · **P3** = polish.

## Counts per category

| Category | P1 | P2 | P3 | Total |
|---|---|---|---|---|
| 1. Dangling / broken cross-links | 1 | 0 | 0 | 1 |
| 2. Terminology / spelling drift | 0 | 1 | 4 | 5 |
| 3. Duplicated explanations | 0 | 2 | 2 | 4 |
| 4. Navigation gaps | 0 | 4 | 1 | 5 |
| 5. Layering | 0 | 0 | 2 | 2 |
| **Total** | **1** | **7** | **9** | **17** |

Headline: the corpus is in **good shape**. Cross-links are almost entirely valid (1 bad
anchor out of 176). Terminology is remarkably consistent (canonical forms dominate by
>95%). The real iteration-4 work is **navigation scaffolding** (the missing 00-OVERVIEW
synthesis files + top-level README + two folder READMEs) and a couple of full-duplication
collapses.

---

## Top 5 most important fixes

1. **(P1)** Fix the one broken anchor: `05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md:112` links to `recurrent-rollback-technique.md#applicability` but the heading is `## 5. Applicability` → GitHub anchor is `#5-applicability`. Change link target to `#5-applicability` (or renumber the heading to bare `## Applicability`).
2. **(P2)** Create the three missing synthesis/entry files (iteration-4 work): top-level `MTP/README.md`, `00-OVERVIEW/the-big-picture.md`, `00-OVERVIEW/corpus-map.md`. See §4 for exact required contents. These are the single biggest navigability gap — a reader has no front door.
3. **(P2)** Add folder-entry READMEs for the two domains that lack them: `01-ARCHITECTURE/README.md` and `05-THEORY-AND-DESIGNS/README.md` (02/03/04/06 already have one). Without them these two folders have no orientation page.
4. **(P2)** Collapse the fully-duplicated "the 3:1 pattern IS the draft/verify schedule" explanation (incl. the ASCII layer diagram + modal draft-mode description) that appears in full in BOTH `05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md §6` and `04-VLLM-GPU/modal-self-speculative.md`. Pick `modal-self-speculative.md` as canonical for the *modal mechanism*; shorten the theory doc §6 to the structural insight + a link.
5. **(P3)** Add a consistent "Related docs" footer to the ~40 docs that lack one (only 4 docs have an explicit see-also footer today). The glossary already models the canonical-home pointer style; mirror it as a short footer everywhere.

---

## 1. Dangling / broken cross-links  (P1×1)

Script-validated all 176 internal `](...)` links + the 2 new 00-OVERVIEW files. **Only one is broken:**

| File:line | Bad target | Issue | Fix |
|---|---|---|---|
| `05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md:112` | `recurrent-rollback-technique.md#applicability` | Target file exists, but the heading is `## 5. Applicability` (line 210) → real anchor is `#5-applicability`. `#applicability` does not resolve. | Either change the link to `#5-applicability`, or drop the number from the heading (`## Applicability`) — but note other docs may rely on the numbered §5 reference, so prefer fixing the link. **P1** |

All other relative file/dir links (including all `../NN-DOMAIN/` cross-domain links and all `patches/`, `src/`, `logs/`, `training/` directory links) resolve correctly. No domain doc forward-references the not-yet-built 00-OVERVIEW files, so no dangling links will appear when those are added.

## 2. Terminology / spelling drift  (P2×1, P3×4)

Canonical forms (per CONVENTIONS hardware strings + dominant usage) win overwhelmingly:
`Qwen3.5-27B` (138 vs scattered), `tok/s` (277 vs 0 true deviations), `DeltaNet` (223),
`split-recurrence` (38), `M4 Max` (36), `eh_proj` (27), `llama.cpp` (129 — all `llamacpp`
hits are paths/identifiers), `RTX 5090`/`GH200 480GB`/`Q4_K_M` (lowercase variants are all
filenames/env-vars/CLI args, legitimate). The few genuine prose deviations:

| File:line | Drift | Canonical | Pri |
|---|---|---|---|
| `04-VLLM-GPU/deploy/gh200/08-GH200-AGENT-INSTALL.md:272` | "Gated Delta Net" (spaced) | "Gated DeltaNet (GDN)" | **P2** — it's the only spaced "Delta Net" in the corpus and sits in an install doc a reader may copy from. |
| `01-ARCHITECTURE/qwen35-hybrid-architecture.md` / `04-VLLM-GPU/gpu-designs.md:74` | "Qwen 3.5-27B" (spaced) | `Qwen3.5-27B` | **P3** — both occurrences are inside quotes *of an external mislabel* ("the \"Qwen 3.5-27B W4A16\" title is a mislabel"), so arguably intentional. Verify, leave if quoting source verbatim. |
| `05-THEORY-AND-DESIGNS/per-position-heads-design.md:20`, `03-MLX/src/README.md:12` | "eh-projection" / "eh-proj" (hyphen) | `eh_proj` (underscore) | **P3** — minor; "eh-projection" reads as the spelled-out word, low risk. |
| `05-THEORY-AND-DESIGNS/research-frontiers.md:118`, `03-MLX/bandwidth-and-dispatch.md` (×6, e.g. :205,:211,:301,:316) | "M4-Max" (hyphen) | "M4 Max" | **P3** — nearly all are compound adjectives ("M4-Max-specific", "M4-Max instantiation") where hyphenation is grammatically defensible; normalize only the noun uses. |
| `03-MLX/the-journey.md:219`, `03-MLX/bandwidth-and-dispatch.md:176,216`, `03-MLX/kernel-fusion.md:148`, `03-MLX/mtp-self-speculative-mlx.md:174`, `05-THEORY-AND-DESIGNS/research-frontiers.md:62`, `training-mtp-heads.md:58` | "tokens/step" / "tokens/s" | These are the **unit `tokens/step`**, NOT a `tok/s` drift — legitimate distinct metric. **No change.** | n/a (false positive, recorded so iter-4 doesn't "fix" it) |

## 3. Duplicated explanations across docs  (P2×2, P3×2)

The glossary (`00-OVERVIEW/glossary.md`) already establishes the canonical-home convention
and most concepts have a clear owner. Genuine *full re-explanations* (vs brief mention +
link, which is fine) needing collapse:

| Concept | Canonical home | Docs re-explaining in full → fix | Pri |
|---|---|---|---|
| **The 3:1 pattern as draft/verify schedule + modal draft-mode** (incl. the ASCII 64-layer diagram) | `04-VLLM-GPU/modal-self-speculative.md` (the modal mechanism) | `05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md §6` (lines 207–242) duplicates the entire ASCII diagram AND re-describes draft-mode/verify-mode + the 100%-of-50-tokens result. Shorten §6 to the structural insight ("the architecture *is* a draft/verify rhythm") and link to the modal doc for the mechanism. | **P2** |
| **The hybrid 48/16 3:1 architecture basics** | `01-ARCHITECTURE/qwen35-hybrid-architecture.md` (owns the layer table + `full_attention_interval`) | `04-VLLM-GPU/modal-self-speculative.md:8–30` and `05-THEORY/speculative-decoding-on-hybrid-models.md` both re-state "64 layers, 48 DeltaNet, 16 attention, 3:1" from scratch. A one-line recap + link to the architecture doc is enough; keep only the modal-specific framing. | **P2** |
| **eh_proj concat-order ("embed-first, not hidden-first → garbage")** | `01-ARCHITECTURE/mtp-head-anatomy.md §The eh_proj concat insight` (line 54) | `03-MLX/the-journey.md:167–169` re-tells it as "the critical bug" (this is a *journey* narrative — keep per CONVENTIONS rule 5, but add a link to the canonical home). `03-MLX/mtp-self-speculative-mlx.md:64` and `03-MLX/README.md:98` repeat the warning — trim to a pointer. | **P3** |
| **Bandwidth ceiling / O(1)-per-token framing** (13.7 GB / 546 GB/s = 25.1 ms floor; "spec decode is not capped at 2×") | `03-MLX/bandwidth-and-dispatch.md` (M4 Max numbers) + `05-THEORY/research-frontiers.md §0` (the O(1) ceiling framing); glossary entries already point here | The math recurs across ~8 MLX/theory docs but mostly as brief contextual mentions, which is acceptable. Only watch `03-MLX/README.md` and `03-MLX/mtp-self-speculative-mlx.md` for over-explaining the floor; trim to a sentence + link. | **P3** |

**Not duplicated (in good shape, recorded to avoid over-collapsing):** the **cache-bookkeeping bug** (canonical `02-LLAMACPP/the-bug.md`; everything else links to it — incl. theory §4 which correctly cross-references rather than re-explains) and the **precision-divergence saga** (canonical `04-VLLM-GPU/precision-divergence.md`; `modal-self-speculative.md`, `vllm-patches-and-strategies.md`, and theory §4 all link out correctly). Leave these as-is.

## 4. Navigation gaps  (P2×4, P3×1)

| Gap | Detail | Fix | Pri |
|---|---|---|---|
| **No top-level `MTP/README.md`** | The corpus has no front door. CONVENTIONS lists it as "master entry + navigation (orchestrator)". | Create (iter-4): one-paragraph what-is-this; the headline results table (link to `00-OVERVIEW/results.md`); a map of the 6 domains with one-line each + link; an "if you read one file, read X" pointer; link to glossary + the-big-picture. | **P2** |
| **No `00-OVERVIEW/the-big-picture.md`** | CONVENTIONS: "the whole arc as one narrative." Does not exist yet. | Create (iter-4): the end-to-end story — the model is a 3:1 hybrid → MTP head is the shipped drafter → recurrent state makes rollback hard → split-recurrence + confidence gating solve it → per-platform results. Should be readable straight through, linking to canonical docs (glossary already drafts an "orientation path" §7–19 that can seed this). | **P2** |
| **No `00-OVERVIEW/corpus-map.md`** | CONVENTIONS: "provenance / where everything came from." Does not exist yet. | Create (iter-4): the 10-source-repo → MTP-domain mapping table (the table in CONVENTIONS is the seed), plus the de-dup collapses. Cross-link to 99-LEDGER inventories. | **P2** |
| **`01-ARCHITECTURE/` has no folder README** | 02/03/04/06 each have a `README.md` entry page; 01 does not. A reader entering the folder sees 4 loose docs (hybrid-arch, mtp-head-anatomy, tensor-layout, weight-extraction) with no ordering. | Add `01-ARCHITECTURE/README.md`: 2–3 lines + ordered links (read hybrid-architecture → mtp-head-anatomy → tensor-layout → weight-extraction; subfolders `extract/`, `tensor-diffs/`). | **P2** |
| **`05-THEORY-AND-DESIGNS/` has no folder README** | Same as above; this folder has 6 docs + 2 src subfolders and is the most-linked-to domain. | Add `05-THEORY-AND-DESIGNS/README.md`: order the reads (speculative-decoding-on-hybrid-models → recurrent-rollback-technique → per-position-heads-design / cascade-mtp-training / training-mtp-heads → research-frontiers; src under `recurrent-rollback-src/`, `training/`). | **P2** |
| **Missing "Related docs" footers** | Only 4 docs carry an explicit see-also/related footer (`mtp-self-speculative-mlx.md`, `recurrent-rollback-technique.md`, `speculative-decoding-on-hybrid-models.md`, `08-GH200-AGENT-INSTALL.md`). The other ~40 rely on inline links + a Provenance footer only. Inline linking is decent, but a consistent footer aids navigation. | Add a short, uniform "Related / see also" footer to the leaf docs (mirror the glossary's canonical-pointer style). Low effort, high consistency payoff. | **P3** |

Note: `00-OVERVIEW/glossary.md` and `00-OVERVIEW/results.md` now **exist** (created concurrently during this audit) and are well-formed — glossary has a clean alphabetized term list with canonical-home pointers and an orientation path; both have valid links. Treat as in-progress per instructions.

## 5. Layering check  (P3×2)

CONVENTIONS rule 4 (narrative-on-top, reproducible-ops-below) is **well-followed overall** —
sampled docs (`qwen35-hybrid-architecture.md`, `mtp-head-anatomy.md`, `the-bug.md`,
`the-recipe.md`, `precision-divergence.md`, `quantization.md`, `weight-extraction.md`) all
open with a blockquote/narrative intro and push exact ops/tensor-maps/Provenance lower. No
P1/P2 offenders. Minor:

| File | Issue | Fix | Pri |
|---|---|---|---|
| `04-VLLM-GPU/deployment.md` | Opens straight into "Deployment Runbook … exact commands" with only a 2-line intro; it is ops-first by nature (a runbook), but has no narrative framing of *why* GH200 vs RTX 5090 or what the reader will achieve. | Add 2–3 narrative sentences up top (what each platform is for, expected result) before Part 1. | **P3** |
| `02-LLAMACPP/infrastructure-patches.md` | Jumps quickly into the ordered patch list / base-commit mechanics. The "what these patches accomplish" narrative is thin relative to the per-patch table. | Add a short narrative paragraph (the arc: base MTP layer → qwen35 port → rollback → the bookkeeping fix) above the apply-order detail. | **P3** |

---

*Agent 3E, read-only consistency auditor. This file is the only file written. No existing
doc was modified. Findings reflect corpus state at audit time (2026-05-27); note other
agents are editing concurrently — `00-OVERVIEW/glossary.md` and `results.md` appeared mid-audit.*
