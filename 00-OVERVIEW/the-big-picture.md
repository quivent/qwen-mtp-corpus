# The Big Picture — the whole arc, end to end

> Read this straight through to understand everything. It is the connective tissue: the
> story of why MTP speculative decoding on Qwen3.5-27B is interesting, why it is *hard*,
> what was invented to make it work, where it actually landed on each platform, and what
> is still open. Every claim links to the canonical doc that owns the detail (and the
> numbers). For the vocabulary, keep [`glossary.md`](glossary.md) open; for the cited
> numbers, [`results.md`](results.md).

---

## 1. The model is a 3:1 hybrid

Qwen3.5-27B is not a plain transformer. Of its 64 layers, **48 are DeltaNet** (gated
linear-attention, *recurrent*) and **16 are full attention**, interleaved in a strict
**3:1 pattern** — three DeltaNet layers, then one attention layer, repeating
(`full_attention_interval = 4`). The DeltaNet layers are O(1) per token: each carries a
small fixed-size recurrent state (an `ssm_state` plus a `conv_state`, ~77 MB/request in
BF16 across the 48 layers) instead of a growing KV cache. The attention layers
periodically restore global recall. This shape is the root of everything that follows.

→ [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md)

## 2. It ships with an MTP head — that every framework strips

Stored on disk as **"layer 64"** is a trained **Multi-Token Prediction (MTP) head**: a
single extra transformer block (RMS-norm → concat `[embedding ‖ hidden]` → `eh_proj` →
gated self-attention → SiLU MLP → shared `lm_head`, 15 weight tensors). It predicts the
**+1** token directly from the main model's hidden state. The catch: **every framework
— the MLX converter, HuggingFace transformers, vLLM, the llama.cpp GGUF converter —
silently strips the MTP head on load** (the block count never gets bumped past 64). So
across all three platforms, **job #1 was re-attaching the head** the framework threw away.

→ [`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md)

## 3. That head is a built-in self-drafter

The MTP head is exactly what speculative decoding needs: a cheap draft of the next token.
Because it lives *inside* the checkpoint and reads the main model's own hidden state,
Qwen3.5-27B can do **speculative decoding without a separate draft model** — the target
model drafts for itself. Draft the +1 token with the head (~3 ms vs a ~34 ms main forward
on M4 Max), then verify it in the next batched forward; commit the matches, roll back the
rest. The stock single head accepts roughly **79%** of its +1 drafts. That is the whole
promise: more tokens per weight read, for almost free.

→ [`../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md)

## 4. But the hybrid architecture breaks standard spec-decode

Standard speculative decoding assumes you can **cheaply roll back** a rejected draft —
which on a KV cache is trivial (`KV = KV[:i+1]`). DeltaNet has no such property. Its
nonlinear delta-rule state update is **irreversible**: there is no "state at position k",
only the state after processing the whole batch, and you cannot invert it to trim it. So
on rejection you must **snapshot and restore** the recurrent state of all 48 DeltaNet
layers, trim the attention KV, and re-decode — and the naive form of that (restore + redo
the forward) costs ~7 ms/step, eating the win. **Rollback is the hard problem** of this
whole effort.

→ [`../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md) §2–3

## 5. Two key inventions

Two ideas turned the hard problem into measured speedups:

- **Split-recurrence rollback** (the corpus's signature technique). Batch the DeltaNet
  *projections* at T=N for verify, but split **only the recurrence** into per-token steps,
  saving a zero-copy reference to the state after each step. Because MLX arrays are
  immutable, a "checkpoint" is just keeping a Python reference; on rejection you restore
  the reference and trim the KV — **no redo**, ~6 ms/step saved. This took MLX from 36.9
  to 42.7 tok/s. It is a *generalizable* technique for any recurrent/hybrid model.
  → [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md)

- **Chained-recurrent MTP with confidence gating.** The shipped head predicts only +1;
  chaining it through its own output to draft +2, +3 is out-of-distribution (accuracy
  decays: +1 ≈ 80%, +2 ≈ 60%, +3 ≈ 40%). The trick is to **gate the chain by the head's
  own top-1 probability** — chain another step only when `max(softmax) ≥ threshold`. This
  is the `stacked_v2` recipe from MLX, ported to llama.cpp as two env vars
  (`MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`). K=2 is the sweet spot; K=3's marginal third
  step doesn't pay back. Zero training cost.
  → [`../02-LLAMACPP/the-recipe.md`](../02-LLAMACPP/the-recipe.md),
  [`../03-MLX/mtp-self-speculative-mlx.md`](../03-MLX/mtp-self-speculative-mlx.md)

## 6. The bug that ate a session

On llama.cpp, with drafting working, the verify/rollback loop produced **garbage** — digit
loops like "1. 2. 3. … 1000000" within ~10 tokens of any rejection. The team spent most of
a session chasing *numerical* suspects (DeltaNet kernel divergence, RoPE positions, state
leaks). All red herrings. The actual culprit was **one line of host-side cache
bookkeeping**: after a batched rollback re-decode, the code set `id_last = corr`, but
`corr` was already the last slot written — so the next verify wrote it a *second* time,
shifting every subsequent cache slot by one. The model saw corrupted context.

The cruelest part: this bug had been **masking every measurement for the entire session**.
The chained-chain threading and the confidence gate were committed *early* but every test
ran on broken text, so they were dismissed as ineffective. After the one-line fix (patch
11), re-running the previously-dismissed env vars delivered the **1.99×**. The biggest win
of the session was a *re-validation*, not a new feature. The lesson — *always verify spec
output against plain decode* — is baked into every benchmark table in this corpus.

→ [`../02-LLAMACPP/the-bug.md`](../02-LLAMACPP/the-bug.md)

## 7. The precision saga (it was kernel numerics, not precision)

On vLLM, the **modal** idea (§ below) drafts by skipping the 16 attention layers and
running only the 48 DeltaNet layers. On CPU this is **100% accurate**. On GPU it diverged
to ~15%. The understanding flipped three times — the modal-mtp README literally contains an
earlier "FP16 is the problem" section *superseded* by a later one in the same file. The
resolved truth comes from holding precision constant: **CPU FP16 = 100%, GPU FP16 = ~15%**
— same precision, different device, result flips entirely. So the cause is **not precision**
and **not a design flaw**: it is a **CUDA-kernel numerical-behavior difference** (the fused
`causal_conv1d` / `chunk_gated_delta_rule` kernels produce slightly different intermediates
than the CPU PyTorch fallback). With the attention corrections absent in draft mode, that
tiny difference compounds.

→ [`../04-VLLM-GPU/precision-divergence.md`](../04-VLLM-GPU/precision-divergence.md)

## 8. The per-position-heads correction

An early belief in this corpus was that **MLX's 1.73× proved trained per-position heads
work** (a DeepSeek-V3-style design with N independent heads, each predicting a fixed
offset). That was **wrong**, and it is corrected here. The shipped Qwen3.5 checkpoint
contains exactly **one** MTP block (`mtp.layers.0.*` — there is no `layers.1`). The MLX win
came from **chained recurrent application of that single head plus confidence gating** (the
`stacked_v2` runtime strategy, zero new weights), *not* from trained per-position heads.
Per-position heads remain a **design-only** forward direction in this corpus — no GPU was
spent training them — and read §0 of that doc before the rest, because it reframes everything.

→ [`../05-THEORY-AND-DESIGNS/per-position-heads-design.md`](../05-THEORY-AND-DESIGNS/per-position-heads-design.md)

## 9. Where it landed, per platform (and the honest reality)

| Platform | Best | What it means |
|---|---|---|
| **llama.cpp** (M4 Max) | **13.98 tok/s** | 1.99× over K=1 vanilla — *but still 0.78× of plain decode (17.90)*. The win is real but does not yet cross plain decode; the gap is per-forward-pass overhead (snapshot/restore vs MLX's cache slicing), **not algorithm**. |
| **MLX** (M4 Max) | **51.1 tok/s** | 1.73× over stock 29.5 — split-recurrence rollback + adaptive confidence chain. The cleanest single-request win in the corpus. |
| **vLLM** (GH200) | **186 / 1030** | Stock MTP spec=7. The 5.54× is *batch* concurrency scaling; **none** of ~14 experimental strategies (modal, tree, sibling heads, cascade, transplant, PLV…) beat stock. |
| **vLLM** (RTX 5090) | **151 tok/s** | GPTQ W4A16 + MTP=5. GPTQ > AWQ for MTP serving (51% vs 31% acceptance — Hessian-optimal rounding preserves draft quality). |

**The honest reality, said plainly:** on this bandwidth-bound hybrid model, single-request
spec decode **often lost to plain decode**. llama.cpp's best is below plain decode; vLLM's
experimental strategies all underperformed stock; and the cross-platform speedups are each
relative to a different baseline and are not comparable. The wins are genuine *within their
baseline* — they are not a free 2×.

→ [`results.md`](results.md) (every configuration, cited)

## 10. What's still open — and the O(1)-per-token ceiling

The single most important framing for the open work: **spec decode is not capped at 2×**.
Decode is memory-bandwidth-bound — every token reads all the weights once (13.7 GB / 546
GB/s = a 25.1 ms floor on M4 Max). With N drafts at 100% acceptance, one T=(N+1) forward
reads the weights *once* and yields N+1 tokens (e.g. N=8 → ~213 tok/s). The real limit is
**draft accuracy at depth**, and a single MTP head cannot sustain depth (acceptance decays
geometrically: 87% → 68% → 54% → 39% → 28% → 21% → 16%). That is *why* the open directions
exist — per-position heads, cascade training, parallel MTP drafts, megakernels, ANE offload
— all aimed at holding accuracy at larger N so the O(1) ceiling becomes reachable.

→ [`../05-THEORY-AND-DESIGNS/research-frontiers.md`](../05-THEORY-AND-DESIGNS/research-frontiers.md),
[`../99-LEDGER/open-questions.md`](../99-LEDGER/open-questions.md)

---

## Related docs

- [`results.md`](results.md) — the numbers behind every claim above.
- [`glossary.md`](glossary.md) — every term, with a pointer to its owning doc.
- [`corpus-map.md`](corpus-map.md) — where everything came from; "where do I find X?".
- [`../README.md`](../README.md) — the corpus front door.

*Synthesis narrative. Asserts nothing new — each section links to the canonical doc that
owns the claim and its numbers.*
