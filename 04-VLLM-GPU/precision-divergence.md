# The Precision-Divergence Saga (Resolved)

> **This is the #1 evolving thread in this corpus.** The understanding changed
> three times and the sources contradict each other in sequence. This document
> presents the **resolved current picture first**, then preserves the full
> journey so nothing is lost and nobody is misled. The contradiction is real: the
> modal-mtp README literally contains an earlier section ("FP16 Accuracy Update"
> → "The attention layers are doing real work at FP16") that is **superseded** by
> a later section in the *same file* ("Critical Finding: CPU vs GPU Divergence").

## The question

Modal MTP drafts by skipping all 16 full-attention layers and running only the
48 DeltaNet layers + MLPs (see [`modal-self-speculative.md`](modal-self-speculative.md)).
Does the DeltaNet-only forward produce the same tokens as the full model? If
yes, near-perfect speculative acceptance is possible. If not, why?

---

## RESOLVED: current understanding (latest, authoritative)

**The divergence is a CUDA-kernel numerical-behavior difference, NOT a precision
problem and NOT a design/representational problem.**

The full test matrix is the smoking gun:

| Device | Precision | Match rate | Result |
|---|---|---|---|
| CPU | FP32 | **100%** (5/5 prompts, 50 tok) | Perfect |
| CPU | FP16 | **100%** (12/12 prompts, 100 tok) | Perfect |
| GPU | FP16 | **~15%** (12/12 prompts) | Diverges token 1–10 |
| GPU | BF16 | **~50%** (8/8 prompts) | Mixed — 4/8 at 100%, 4/8 early diverge |

The decisive comparison is the **CPU FP16 row vs the GPU FP16 row**: *same
precision, different device* → 100% vs ~15%. Precision is held constant and the
result flips entirely. Therefore the cause cannot be precision.

**Mechanism:** the fused DeltaNet CUDA kernels — `causal_conv1d` and
`chunk_gated_delta_rule` — produce slightly different intermediate values than
the CPU PyTorch fallback. When the attention corrections are *absent* (draft
mode), those tiny kernel-level differences are no longer periodically washed out
by attention, so they accumulate through the recurrence and flip token argmax
within a few steps. The CPU PyTorch reference path does not exhibit this because
it uses unfused, reference-accurate math.

**Consequences:**
- The architecture is **proven correct at all precisions** (both CPU runs are
  100%). DeltaNet-only drafting is representationally sound.
- The GPU divergence is **kernel-level, not design-level**.
- It is **fixable** by matching kernel accumulation behavior, **or tolerable** if
  the speculative-decode acceptance rate stays high enough (rejected drafts are
  just re-verified; correctness is never compromised).

> Note the FP16-GPU output is unusable for non-spec generation, but speculative
> decoding *verifies every draft token against the full model*, so a low draft
> match rate only costs throughput, never correctness.

### Proposed (still-untested) fix: FP32 state accumulation

vLLM already exposes `--mamba-ssm-cache-dtype float32`, which keeps the DeltaNet
recurrent state in FP32 while compute stays BF16/FP16. Cost: **150 MB per request**
(vs 77 MB at BF16) — negligible. The hypothesis is that FP32 accumulation in the
recurrence would damp the kernel-level differences before they flip argmax.

**Status: untested in vLLM.** It can only be validated once the modal_mtp model-
runner draft loop is wired up, because HuggingFace `generate()` (used for the CPU
validation) does not expose the state-dtype control. The target combination is
**BF16 compute + FP32 state**.

---

## The journey (preserved — how the understanding evolved)

### Stage 1 — FP32/CPU: 100% match → "architecture is sound"

The first validation (`validate_draft_accuracy.py`, FP32 on CPU) drafted 50
tokens across 5 prompts with attention skipped and compared to the full model:

| Prompt | Tokens | Match |
|---|---|---|
| "The theory of relativity states that" | 50 | **100%** |
| "def fibonacci(n):\n    " | 50 | **100%** |
| "In 1969, humans first" | 50 | **100%** |
| "The chemical formula for water is" | 50 | **100%** |
| "Once upon a time in a dark forest," | 50 | **100%** |

Every DeltaNet-only token matched the full model. Conclusion at the time:
*architecture sound, near-perfect acceptance possible.*

### Stage 2 — FP16/GPU: early divergence → (WRONG) "it's a precision problem"

Re-running at FP16 (production precision) on GPU showed dramatic early divergence
(12 prompts, the "FP16 Accuracy Update" table):

| # | Match | Div@ | Prompt |
|---|---|---|---|
| 1 | 34% | 10 | def merge_sort(arr): |
| 2 | 18% | 4 | function debounce(fn, delay) { |
| 3 | 12% | 2 | Solve step by step: train at 60 mph… |
| 4 | 19% | 8 | derivative of f(x) = x^3 * ln(x) |
| 5 | 3% | 1 | The last human on Earth… |
| 6 | 4% | 1 | Write a haiku about the ocean |
| 7 | 7% | 1 | Three laws of thermodynamics |
| 8 | 84% | 79 | Name the planets in order |
| 9 | 6% | 5 | Explain recursion to a 5 year old |
| 10 | 2% | 2 | 用中文解释什么是人工智能 |
| 11 | 24% | 7 | Relativité générale (French) |
| 12 | 10% | 5 | SELECT u.name, COUNT(o.id)… |

> **SUPERSEDED interpretation (do not trust this framing):** the README's
> Stage-2 text concluded *"The attention layers are doing real work at FP16
> precision. DeltaNet recurrence at half precision accumulates errors faster than
> at FP32… it's a precision problem, not a representational one."* This is
> **wrong** — Stage 3 disproves it. Kept here only to document the dead-end.

A related (also-superseded) artifact is the **"FP32 State Accumulation Path"**
section, which framed `--mamba-ssm-cache-dtype float32` as the fix *for a
precision problem*. The fix itself may still help (FP32 state damps kernel
noise), but the *reason given* (precision) was the wrong diagnosis.

### Stage 3 — the reversal: CPU FP16 = 100% → "NOT precision, it's the kernels"

The decisive experiment was `validate_extended.py` run **on CPU at FP16** (12
prompts × 100 tokens, `device_map="cpu"`, `dtype=torch.float16`). Result
(`validate_extended_output.log`): **1146/1146 tokens matched = 100.00%, 12/12
prompts perfect, zero divergences.** The log even notes the fast fused kernels
were unavailable and it fell back to the torch implementation — i.e. the
reference math path.

Same FP16 precision as Stage 2, but on CPU → 100% instead of ~15%. **Precision
held constant; device changed; result flipped.** That kills the precision
hypothesis and points squarely at the fused GPU kernels. This is the resolved
understanding above.

(An intermediate BF16/GPU run added the "~50% (8/8), mixed" data point — better
than FP16/GPU but still not the 100% of CPU — consistent with kernel-numerics,
not precision.)

## The diagnostic tool

`optimizations/diagnose_divergence.py` (copied from `modal-mtp/`) is the
layer-by-layer instrument built to localize the divergence. It runs one prompt
through the GPU FP16 model twice — normal vs skip-attention — capturing hidden
states after every layer via forward hooks, and reports:
- first layer where any divergence appears,
- first **DeltaNet** (`linear_attention`) layer to diverge (the real signal —
  proves "fused DeltaNet kernels amplify upstream differences"),
- the top-5 most divergent DeltaNet layers and a growth-pattern histogram.

It expects to confirm that divergence originates/amplifies in DeltaNet layers
(fused-kernel sensitivity), not merely at the skipped attention layers.

## Bottom line for an engineer

1. **Trust the matrix, not the prose.** CPU 100% (both precisions) vs GPU
   ~15–50% = kernel numerics.
2. Don't "fix precision" — that diagnosis was retracted. Investigate the
   `causal_conv1d` / `chunk_gated_delta_rule` fused kernels' accumulation, or
   accept the lower draft match rate (spec decode stays correct regardless).
3. The FP32-state path (`--mamba-ssm-cache-dtype float32`, +73 MB/req) is the
   first thing to try once the modal_mtp draft loop is wired up — **but it has
   never been run in vLLM**, so treat it as a hypothesis.

## Related

- [`modal-self-speculative.md`](modal-self-speculative.md) — the draft mechanism whose GPU divergence this saga explains.
- [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) — DeltaNet recurrent state + `--mamba-ssm-cache-dtype` (the proposed FP32-state fix).
- [`vllm-patches-and-strategies.md`](vllm-patches-and-strategies.md) — `diagnose_divergence.py`, the validate scripts, and the decisive CPU-FP16 log.
- [`../99-LEDGER/open-questions.md`](../99-LEDGER/open-questions.md) — the precision saga as an open thread (untested FP32-state path).
- [`../00-OVERVIEW/glossary.md`](../00-OVERVIEW/glossary.md) — DeltaNet/GDN, `causal_conv1d`, `chunk_gated_delta_rule`.

---
*Provenance: `modal-mtp/README.md` (all four evolving sections), the byte-distinct
`qwen-ops/research/findings/modal-mtp-precision-divergence.md` (which already
presents the resolved CPU-vs-GPU framing), `modal-mtp/validate_extended_output.log`
(the decisive CPU-FP16-100% evidence), `modal-mtp/validate_draft_accuracy.py`,
`modal-mtp/validate_extended.py`, `modal-mtp/diagnose_divergence.py`.*
