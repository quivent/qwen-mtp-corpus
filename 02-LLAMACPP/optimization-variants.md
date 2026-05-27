# Optimization variants — the 9 patches

> Nine speculative-decoding optimization variants explored on top of the
> [infrastructure substrate](infrastructure-patches.md). Copied verbatim in
> [`patches/optimizations/`](patches/optimizations/). **Variant 01 (adaptive chained
> MTP) is the winner: 1.99× over K=1 vanilla.** Author `quivent`, 2026-04-08.

## ⚠️ The honesty caveat (pre-fix vs post-fix)

For most of the session, **every variant looked like a win** (1.16×, 1.72×, 2.5×).
**All of those measurements were on degraded text** — the [cache-bookkeeping
bug](the-bug.md) was corrupting output within ~10 tokens of any rejection, and
nobody validated text coherence. Only after **patch 11** (the fix) do measurements
mean anything. Where we have post-fix numbers, they're labeled. Where we don't, the
variant is honestly marked "needs post-fix re-validation". The patches are preserved
as the **implementation record** — the infrastructure work in each (the new memory
primitives, the in-graph AR loop, the tree plumbing) is reusable regardless of
whether the variant itself won.

---

## Status table (post-bug-fix)

| # | Variant | Output coherent? | Speedup vs K=1 vanilla | Status |
|---|---|---|---|---|
| 01 | **Adaptive chain** (`MTP_CHAIN_THRESH`, `MTP_CHAIN_KMAX`) | ✓ | **1.99×** (13.98 vs 7.02 tok/s mean) | 🏆 **WINNER** — the MLX `stacked_v2` recipe |
| 02 | Debug verify (`MTP_DEBUG_VERIFY`) | n/a (diagnostic) | — | Essential debugging primitive |
| 03 | Drift refresh (`MTP_REFRESH_EVERY=N`) | ✓ (pre-fix) | 10× accept jump pre-fix | Redundant post-fix (chain already bounds drift via re-decode) |
| 04 | Predictive hidden draft (`MTP_PREDICTIVE_ALPHA`) | ✓ (pre-fix) | K=2 accept 8→83% on short prefixes pre-fix | Superseded by variant 01 |
| 05 | Perturbed-head ensemble (`MTP_ENSEMBLE_K=N`) | ✓ | needs post-fix re-validation | Orthogonal to 01 — tree-fork variant of the same idea |
| 06–07 | Branching speculative tree (`MTP_TREE_B`, `MTP_TREE_DEPTH`) | ✓ | needs post-fix re-validation | Orthogonal — multi-sequence parallelism |
| 08 | Ensemble fast-path (`MTP_ENSEMBLE_FAST`) | **✗** | — | **BROKEN** — recurrent contamination on this hybrid model |
| 09 | Stacked hidden-noise (`MTP_STACK_N`) | ✓ | **0.58× to 0.69×** | **DECISIVELY NEGATIVE** — MTP head is saturated |

> The fast-path #08 is the one variant we have *post-fix* evidence about, and it
> **breaks output**. #09 is decisively negative. #03/#04 were "wins" pre-fix only.
> #05/#06-07 await post-fix numbers.

---

## The variants in detail

### 01 — Adaptive chain 🏆 **THE WINNER** (commit `7ac89131e`)
Env: `MTP_CHAIN_THRESH` (top-1 prob cut-off in [0,1], default 1.0 = K=1 only),
`MTP_CHAIN_KMAX` (hard depth cap, default = n_draft). After each MTP draft step,
compute top-1 probability from `softmax(logits)`; stop chaining when it falls below
threshold. On a chained step, feed the MTP head's own output hidden
(`t_mtp_out_hidden`) as the next step's `prev_hidden` — the recurrent step. **This is
the same recurrent-stack technique MLX `stacked_v2.py` uses.** Combined with the
patch-11 fix, delivers **1.99× over K=1 vanilla** at `MTP_CHAIN_KMAX=2
MTP_CHAIN_THRESH=0.85` with coherent output. No behavior change by default. Full
recipe + benchmark in [`the-recipe.md`](the-recipe.md).

### 02 — Debug verify (commit `0eb4d026c`)
`MTP_DEBUG_VERIFY` dumps draft-vs-target argmax per verify step so you can see
exactly where the drafter diverges from the main model. Pure diagnostic. Comparing
top-K logits (not just argmax) is the cheapest way to detect drift before it cascades.

### 03 — Drift refresh (commit `a96ffc806`)
`MTP_REFRESH_EVERY=N` (default 0 = off): every N committed tokens, skip the MTP
draft and do a plain T=1 decode of `id_last`. That single-token decode traverses
`build_delta_net_autoregressive` cleanly and **bounds the `mtp_prev_hidden` drift**
that otherwise compounds across verify iterations. The cleanest single-knob variant;
works on any hybrid model. **Status: redundant post-fix** — the chained path's
re-decode already bounds drift. (Pre-fix it showed a 7%→75% / "10× accept jump".)

### 04 — Predictive hidden draft (commit `475aa869b`)
`MTP_PREDICTIVE_ALPHA` gates Option A: predict the next-step hidden as
`h_{t+k} ≈ h_main + alpha · Σ_{j<k} embed(draft_token_j)` instead of running the
main model just to get a fresh hidden. Replaces the (then-broken) MTP-output-hidden
chaining for k≥1. **Status: superseded by variant 01** (which got the real chaining
working post-fix). Pre-fix it showed K=2 accept 8→83% (+75pt) on short prefixes.

### 05 — Perturbed-head ensemble (commit `268e325f0`)
`MTP_ENSEMBLE_K=N`: emit top-N **sibling** candidates for position t+1 from **one**
MTP forward pass (instead of chaining). Verify as a tree fork — slot-0 logits predict
the true next token; whichever draft matches is accepted. Top-1 hit (happy path): one
forward, commits 2 tokens (d_0 + slot-1 lookahead). Alt-hit/miss: snapshot-rollback +
T=2 re-decode, 2 forwards for 2 tokens. K=1 default untouched. **Status: orthogonal to
01; needs post-fix re-validation.**

### 06–07 — Branching speculative tree (commits `d4af5df2c`, `ca1595f37`)
`MTP_TREE_B` (branches), `MTP_TREE_DEPTH` (depth). Full B×D tree with multi-sequence
batching: (1) draft top-B roots; (2) chain each root to depth D-1 reusing the shared
main hidden; (3) fork seqs 1..B-1 from seq 0; (4) build one multi-seq verify batch;
(5) find longest accepted prefix; (6) restore recurrent state, purge aux seqs,
re-decode committed prefix AR.

Patch **07** is the fix that makes it run end-to-end: bump `n_parallel` to tree_B,
force `kv_unified=true` (cross-seq `seq_cp` with partial range needs unified KV),
and **duplicate `id_last` per branch** rather than sharing a coupled slot (the hybrid
memory's `split_equal(sequential=true)` rejects tokens belonging to multiple
sequences). Verify batch resized 1+B*D → B*(1+D). B=2/D=1, B=2/D=2, B=3/D=2 all
complete. **Output: first ~1-3 tokens clean, then DeltaNet state drifts → token
repetition** — same split-rec-v2 root cause as the linear path (multi-token verify on
hybrid memory routes DeltaNet through chunking; forked seqs corrupt aliased recurrent
cells). **Status: orthogonal; needs post-fix re-validation.**

### 08 — Ensemble fast-path skip (commit `bca807788`) — **BROKEN**
`MTP_ENSEMBLE_FAST` (default on when `MTP_ENSEMBLE_K` set; =0 restores bit-exact
path). On a top-1 ensemble hit, **skip** the snapshot-restore + T=2 commit re-decode:
trim attn KV at `[P+2, +inf)` and force the recurrent tail position back to P+1 via
the new `llama_memory_seq_force_recurrent_pos` primitive (overwrites
`cells[seq_id].tail->pos` without touching hidden state). The recurrent hidden
**retains contamination** from d_1..d_{K-1}. Skips the second forward on ~55% of
cycles. **Pre-fix benchmark** (5 prompts, n=64, draft-max=3): K=1 vanilla 4.94, K=3
ensemble slow 6.74, K=3 ensemble **fast 8.50 tok/s (1.72×)**. **Post-fix verdict:
output is incoherent** — the "close enough" recurrent contamination assumption fails
on this hybrid model. **This is the one variant with post-fix evidence, and it breaks.**
(Same finding as `MTP_SKIP_ROLLBACK` in [`the-recipe.md`](the-recipe.md): contamination
raises the reject rate on subsequent iterations, cancelling the savings.)

### 09 — Stacked hidden-noise validator (commit `6f85e8817`) — **DECISIVELY NEGATIVE**
`MTP_STACK_N` (default 1 = byte-identical vanilla), `MTP_STACK_NOISE` (sigma, default
1e-3). Run the MTP head N times per draft step with small Gaussian noise added to
`prev_hidden`, sum logits before argmax, ensemble-vote. **Hypothesis:** noisy logits
average into a more reliable argmax. **Result: doesn't work.** The MTP head is
**structurally saturated** — small perturbations don't move the argmax *at all*
(accept count byte-identical at N=1, 2, 4, 8), large perturbations only shift it
within run-to-run noise. The bottleneck isn't noisy logits; it's that the head's top-1
is **structurally wrong ~90% of the time**. Per-pass cost makes it strictly worse:
**5.4 tok/s at N=2 vs 7.8 vanilla; 0.58× to 0.69×**. Optimal N=1 (don't enable).
**Published as the implementation record of a clean negative result.**

---

## Why ensembling can't help here (the deep lesson of #09)

Ensembling decorrelates **independent** errors. The MTP head's errors are not noise
around a correct answer — its top-1 is *structurally* wrong most of the time because a
single head chained out-of-distribution to +2/+3 is fundamentally mis-predicting.
You cannot average your way out of a systematic bias. This is the same reason K=2 beats
K=3 in the recipe: the marginal chained step is too inaccurate to pay back its cost.

## Why publish negative/pending variants?

The *infrastructure work* in each patch is reusable: the tree path's discovery that
hybrid recurrent memory needs `kv_unified=true` for `seq_cp`; the
`llama_memory_seq_force_recurrent_pos` primitive; the in-graph AR loop; the
snapshot/restore plumbing. All correct, hard-won work the next person attacking spec
decoding on a hybrid model shouldn't have to rediscover.

---

## Related

- [`the-recipe.md`](the-recipe.md) — the winner (variant 01) in full: env vars, benchmark, levers.
- [`infrastructure-patches.md`](infrastructure-patches.md) — the substrate these 9 variants sit on top of.
- [`the-bug.md`](the-bug.md) — why every pre-fix "win" here was on corrupted text.
- [`patches/optimizations/`](patches/optimizations/) — the 9 variant patches, copied verbatim.
- [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) — the rollback technique the tree/ensemble variants stress.
- [`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md) — where the 1.99× sits among all platforms.

---
*Provenance: distilled from `qwen-mtp-optimizations/README.md`,
`qwen-mtp-research/README.md`, and the commit messages in
`qwen-ops/llamacpp/optimizations/*.patch` (= `qwen-mtp-optimizations/patches/*`,
byte-identical).*
