# Integration plan — per-position MTP heads (DeepSeek V3 style)

> The alternative path if the [chained recipe](the-recipe.md) doesn't cross plain
> decode: train **N=4 per-position MTP heads** so one main forward pass amortizes over
> up to 4 committed tokens. Six phases, with a **spend gate between each** — the user
> decides whether to proceed. **Phase 0 is a kill-or-proceed gate before any GPU spend.**

## Why this path exists

Post-fix, single-head MTP spec runs at **0.43× of plain** (K=1 = 7.64 tok/s, plain =
17.90). The MTP draft pass costs ≈ a main forward pass, so single-head spec can't win.
Two ways forward: (1) make the draft pass cheaper (graph caching, fusing — see
[the-recipe.md](the-recipe.md)); (2) **per-position heads** — multiple cheap heads, each
predicting +1, +2, … +N from the same hidden state, so one main forward amortizes over
N committed tokens. This is the DeepSeek V3 design.

> **The MLX truth (correction to an earlier belief):** the MLX implementation that hits
> 1.68×/1.73× is **NOT** per-position trained heads — its checkpoint contains exactly
> **one** MTP block, the same single head this llama.cpp port uses. Its speedup comes
> from the *runtime* chained-recurrent strategy (the recipe), a 0.8B companion, and
> confidence gating — **zero training cost**. Per-position heads remain a *separate*,
> training-based alternative path documented here. (See `MTP/05-THEORY-AND-DESIGNS/` for
> the full resolution; flagged in the ledger.)

## Per-position heads — design headlines

- **N=4 heads**, each a single transformer block, sharing the main model's embedding
  and LM head.
- **Tensor naming**: `blk.<64+k>.nextn.*` for k=0..3, slotted into the existing GGUF
  layer table.
- **Training**: warm-start each head from the existing single MTP layer's weights,
  freeze the main model, train heads jointly with summed cross-entropy at offsets +1..+4
  on a **1B-token corpus**.
- **Inference math**: 1 main forward (60 ms) + 4 head forwards (4×10 ms) = **100 ms per
  cycle**, commits up to 4 tokens → theoretical **40 tok/s** vs plain 17.9 = **2.23×
  speedup ceiling**.
- **Estimated cost**: ~**$165** in GPU time (Phase 1 data + Phase 2 training).
- **Honest risk**: the "10 ms per head" assumption needs measurement; on the hybrid
  model the heads still need to walk DeltaNet recurrent state, which may not be as cheap
  as a pure-attention head. **This is exactly what Phase 0 measures.**

---

## The six phases (with spend gates)

### Phase 0 — Validate the cost assumption (NO GPU, NO TRAINING) — **the gate**
**Goal**: prove the head forward pass is cheap (≤ 20 ms) before spending training money.
**Actions**: instrument `build_mtp_head` in `src/models/qwen35.cpp` with
`ggml_time_us()` markers; run the current single-head path 1000 iterations, record
p50/p99 head wall time; compare to main forward (56 ms baseline).
**Success**: `head_fwd ≤ 0.25 × main_fwd` (i.e. ≤ 14 ms).
**Failure**: if `head_fwd ≈ main_fwd` (**likely**, given K=1 spec runs at 0.43× of
plain), the entire per-position approach is **dead** — no amount of training fixes it,
because the fixed overhead per draft pass (KV bookkeeping, DeltaNet state, graph alloc)
dominates, not the head's FLOPs. Re-evaluate (switch to MLX-style 0.8B stacked draft, or
shelve spec decode for this model).
**Rollback**: none — read-only.

### Phase 1 — Build training data (GPU spent)
**Script**: `scripts/build_training_data.py`. **Cost**: ~24 H100-h, ~$85. **Disk**:
~36 GB tokens-only (or ~4 TB with cached hidden states — pick tokens-only, recompute
hidden on-train). **Failure modes**: OOM at 2048 seq_len × batch 8 → halve batch;
tokenizer mismatch → verify `vocab_size == 151936` before starting. **Rollback**: delete
output dir.

### Phase 2 — Train heads (GPU spent)
**Script**: `scripts/train_per_position_heads.py`. **Cost**: ~23 H100-h, ~$80.
**Total Phase 1+2**: ~**50 H100-hours, ~$165**. **Failure modes**: loss plateaus at
random-init level → warm-start didn't load; head 3/4 loss much higher than 1/2 →
expected, still useful if p_4 ≥ 0.4; catastrophic main-model drift → main is frozen, so
that's a bug. **Checkpoint**: every 5k steps, keep last 3. **Rollback**: discard, re-run
from last good step.

### Phase 3 — Quantize + GGUF append (CPU-only)
Extend `convert_hf_to_gguf.py` per the tensor-layout doc (loop over `num_mtp_layers`);
run converter → `qwen35-27b-mtpN4.gguf` (fp16); `llama-quantize` to Q4_K_M; verify
`gguf-dump | grep nextn` shows 4 sets and KV metadata `qwen3.nextn_predict_layers == 4`.
**Failure**: tensor shape mismatch → converter dims off (diff against tensor-layout doc).
**Rollback**: use existing single-head GGUF.

### Phase 4 — Land loader + graph + inference patches
Apply `loader-stub.patch`, `graph-stub.patch`, `inference-stub.patch` (copied into
[`patches/per-position-heads-stubs/`](patches/per-position-heads-stubs/), origin
`qwen-mtp-research/patches/`) as starting points → real commits. **Order**: loader (MTP
dispatches head_0 only) → graph (`mtp_head_idx` plumbing) → inference (draft loop
iterates K heads). **Validation each step**: plain decode must still produce identical
output. **Rollback**: `git revert`.

### Phase 5 — Validate output coherence end-to-end
`MTP_NUM_HEADS=1` → must match pre-patch single-head MTP exactly. `MTP_NUM_HEADS=4` →
must match plain decode under greedy (spec decode is exactness-preserving under greedy —
**this is the correctness test**). Run 10 prompts × 500 tokens, compare byte-for-byte.
**Do not proceed to benchmarks until byte-exact.** **Rollback**: `MTP_NUM_HEADS` disables.

### Phase 6 — Benchmark
**Metrics**: tok/s (plain), tok/s (K=1 single-head = today's 7.64), tok/s (K=4
per-position), per-head accept rate, cycle-time breakdown (main + head_0..3).
**Success**: K=4 tok/s > **1.2 × plain** (21.5 tok/s). **Stretch**: ≥ 1.5× (26.9 tok/s).
**Failure**: K=4 < plain → head cost or accept rate worse than modeled; debug with the
breakdown, then retrain (lower K, more data, longer schedule) or shelve.
**Rollback**: `MTP_NUM_HEADS=0`/unset keeps the path dormant.

---

## Summary of spend gates

| Phase | GPU-h | $ | Decision after |
|---|---|---|---|
| 0 | 0 | 0 | Is head_fwd actually cheap? |
| 1 | 24 | 85 | Data looks healthy? |
| 2 | 23 | 80 | Loss curves match expected? |
| 3 | 0 | 0 | GGUF loads? |
| 4 | 0 | 0 | Plain decode still exact? |
| 5 | 0 | 0 | Spec output byte-exact? |
| 6 | 0 | 0 | Actual speedup ≥ 1.2×? |

**Total committed through Phase 2: ~50 GPU-hours, ~$165.** Everything after Phase 2 is
free (CPU/validation). Phase 0 is the cheapest and most important gate.

---

## Related

- [`the-recipe.md`](the-recipe.md) — the zero-training chained path this plan is the alternative to.
- [`../01-ARCHITECTURE/tensor-layout-hf-to-gguf.md`](../01-ARCHITECTURE/tensor-layout-hf-to-gguf.md) — the `blk.<64+k>.nextn.*` layout Phase 3 appends (already N>1-capable).
- [`../05-THEORY-AND-DESIGNS/per-position-heads-design.md`](../05-THEORY-AND-DESIGNS/per-position-heads-design.md) — the full per-position-head design + training scripts.
- [`../05-THEORY-AND-DESIGNS/cascade-mtp-training.md`](../05-THEORY-AND-DESIGNS/cascade-mtp-training.md) — the cascade/depth-trained variant of the same idea.
- [`patches/per-position-heads-stubs/`](patches/per-position-heads-stubs/) — the loader/graph/inference stub patches Phase 4 starts from.

---
*Provenance: distilled from `qwen-mtp-research/docs/integration-plan.md` and the
per-position-heads headlines in `qwen-mtp-research/README.md`. The full per-position-head
design + training scripts (`build_training_data.py`, `train_per_position_heads.py`) and
`docs/tensor-layout.md` are owned by `MTP/01-ARCHITECTURE/` and
`MTP/05-THEORY-AND-DESIGNS/`.*
