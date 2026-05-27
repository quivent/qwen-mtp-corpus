# The bug that ate the session — cache-bookkeeping double-write

> A one-line host-side cache-bookkeeping bug that corrupted every speculative-decode
> measurement for an entire session. Six agents missed it while hunting numerical
> bugs. The fix is [infrastructure patch 11](infrastructure-patches.md). This is the
> canonical home of the bug story.
>
> **Note on provenance:** the source repo `qwen-mtp-research/README.md` references a
> `docs/the-bug.md` file — **that file does not exist** in the source repo. The bug
> story lives only in the README narrative and the patch-11 commit message. This
> document reconstructs the full story from both.

---

## The symptom

For most of the session, every optimization variant looked like a win — speedups of
1.16×, 1.72×, 2.5× depending on prompt. **All of those measurements were on degraded
text** that diverged from plain decode within **~10 tokens after any rejection**. The
characteristic failure: the model collapsed into digit loops —
`"1. 2. 3. ... 1000000"`.

## The root cause

In `examples/mtp-speculative/mtp-speculative.cpp`, after a **batched rollback
re-decode** of `[id_last, drafts…, corr]` (the re-decode introduced by
[patch 10](infrastructure-patches.md)):

```cpp
// after a batched rollback re-decode of [id_last, drafts..., corr]:
n_past  += n_commit;
id_last = corr;          // BUG: corr is already in the cache as the last batch slot
```

`corr` (the correction token) **was already written to the cache as the last slot of
that re-decode batch**. Setting `id_last = corr` meant the *next* iteration's verify
batch wrote `corr` a **second time** at position `n_past + n_commit` — duplicating the
token and **shifting every subsequent token by one slot**. From that point on the
model saw garbage context: each position's KV held the wrong token. Output degenerated
into digit loops.

The bug only manifests with the **batched** re-decode (patch 10). The earlier
one-token-at-a-time re-decode path didn't trigger it the same way — which is part of
why it appeared late and looked like a regression from an optimization.

## The fix (patch 11)

Mirror the accept-all branch: read the **tail logits** of the re-decoded batch and use
their `argmax` as the new `id_last`. `n_past` advances by exactly `n_commit`; no
double-write.

```cpp
n_past  += n_commit;
id_last = argmax(tail_logits);   // the token the re-decode actually predicts next
```

After the fix, K=1 spec output matches plain decode structurally on all benchmark
prompts (planets / photosynthesis / fibonacci / haiku / translate). K=1 spec mean
throughput: **7.64 tok/s** (plain decode **17.90 tok/s**).

## Why six previous agents missed it

All six were hunting **forward-pass numerical bugs** — the symptom was "wrong logits",
so the instinct was to stare at the graph:

- chunking-vs-AR DeltaNet divergence in fp16 (→ patch 09, proven a red herring)
- RoPE positions
- recurrent state leak across decodes (→ patch 06's private scheduler)
- hidden-state pipe corruption (`mtp_prev_hidden` capture)
- the `kq_mask` scheduler buffer bug (→ patch 04 diagnostic naming)

The bug was **three lines below all the things they kept staring at**, in the
host-side bookkeeping, not the numerics. It was found only when an isolated debugger
compared **plain-decode vs spec slot-0 logits at every iteration with the SAME
`id_last`** (the `MTP_DEBUG_SLOT0` knob added in patch 11). The divergence proved the
**cache history was wrong**, not the math being computed on it.

---

## The three lessons

### 1. Validate output text against ground truth on *every* measurement
Throughput numbers without coherence checks are meaningless. Six agents reported
speedups; **zero validated text quality.** Every "win" was on broken output. Always
diff spec output against plain decode (greedy spec decode is supposed to be
exactness-preserving) before believing a tok/s number. The corollary tooling: a
`MTP_FORCE_AR`-style always-on ground-truth path (patch 08) and a token-by-token diff.

### 2. Mutual drift convergence is real
When **both** drafter and target are corrupted by the **same** upstream bug, accept
rates can paradoxically *climb* — the two corrupted distributions **agree on garbage**.
A rising accept rate is not evidence of correctness if the corruption is shared. This
is why "the accept rate went up" was actively misleading the optimization agents.

### 3. Bookkeeping bugs hide behind numerical bugs
When the symptom is "wrong logits", the instinct is to look at the graph. But the
graph can be perfectly correct while being **fed the wrong input**. Always check that
the graph is being handed the context you *think* it is — verify the cache history /
position bookkeeping before deep-diving the kernels. "Wrong output" ≠ "wrong math".

---

## The meta-lesson (consequence for everything else)

The cache bug had been **masking every measurement** for the entire session. The
winning recipe — `MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85` — was committed *early* and
**dismissed as ineffective** because its measurements were on corrupted text. After
fixing a bug that affects measurement: **re-run every experiment previously dismissed
as ineffective**, especially the ones that "almost worked but didn't quite pay back."
The session's biggest win (1.99×) was a **re-validation, not a new feature.** See
[`the-recipe.md`](the-recipe.md).

---

## Related

- [`infrastructure-patches.md`](infrastructure-patches.md) — patch 11 is the fix; patch 10 is the batched re-decode that exposed it.
- [`the-recipe.md`](the-recipe.md) — the recipe this bug had been masking the whole session.
- [`optimization-variants.md`](optimization-variants.md) — the "wins" that turned out to be on corrupted text.
- [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) — the rollback path whose bookkeeping this bug lived in.
- [`../00-OVERVIEW/glossary.md`](../00-OVERVIEW/glossary.md) — *cache-bookkeeping bug* entry (points back here).

---
*Provenance: reconstructed from `qwen-mtp-research/README.md` ("The bug that ate the
session") and the patch-11 commit message in
`qwen-ops/llamacpp/infrastructure/11-fix-mtp-rollback-re-decode-bookkeeping-id_last-fro.patch`.
The referenced `qwen-mtp-research/docs/the-bug.md` does not exist in the source.*
