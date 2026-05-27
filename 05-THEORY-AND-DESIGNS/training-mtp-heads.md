# Training / Improving MTP Heads

How to train or improve the MTP draft head(s) for Qwen3.5-27B so speculative decoding
accepts more, deeper. Covers the **distillation-from-logits** principle (no labels needed),
the freeze-main/train-head setup, the expected accept-rate gains and resulting tok/s, and
the **cascade-corrective** training idea. The two reference scripts are copied into
[`training/`](training/).

> Related designs: [`per-position-heads-design.md`](per-position-heads-design.md) (the
> architecture being trained), [`cascade-mtp-training.md`](cascade-mtp-training.md) (the
> depth-specific cascade variant), and
> [`speculative-decoding-on-hybrid-models.md`](speculative-decoding-on-hybrid-models.md) §7
> (why training matters on this model).

---

## 1. The core idea: distill from the main model's own logits (no labeled data)

The MTP head is a *drafter*. It only needs to agree with the **main model's argmax** — it
does not need to be "correct" in any external sense. So training needs **no labeled data**:

- **Freeze the main model.** Train only the head(s).
- **Teacher = the main model's own hidden states + next-token targets** drawn from a plain
  text corpus. The "label" for position t is simply `token_{t+1}` from the corpus (and
  `token_{t+k}` for head_k). The head learns to reproduce, from `h_t`, what the main model
  would say next.
- This is cheap: only ~15 tensors / ~480–800M params are trained while the 27B main model
  runs forward-only.

The hard-won constraint (from the cascade experiments, see
[`cascade-mtp-training.md`](cascade-mtp-training.md)): **the hidden states used for
training MUST come from the EXACT serving checkpoint** that will verify drafts at inference.
Training on hidden states from a different model load (e.g. a separate bf16 vs the W4A16
serving copy) gives **0% acceptance** — the head produces outputs mismatched to what the
verifier expects. Same dtype, same checkpoint, same forward path.

---

## 2. Setup (freeze main, train head)

From [`training/train_per_position_heads.py`](training/train_per_position_heads.py)
(design-only; requires an H100-class GPU):

- **Main model:** frozen Qwen3.5-27B (bf16), `requires_grad_(False)`, `eval()`.
- **Heads:** N=4 new MTP blocks, each warm-started from the existing single trained
  `mtp.layers.0` (load `mtp_weights_vanilla.safetensors`), then perturbed with `N(0, 0.02)`
  noise so the heads don't collapse to identical functions.
- **Objective:** sum of per-head cross-entropy against ground-truth `target_{t+k}`.
- **Teacher forcing:** head_k receives ground-truth `token_{t+k-1}` (scheduled sampling
  optional in the last 10% of steps).

### 2.1 Hyperparameters (DeepSeek V3 reference)

```
lr           = 1e-4 peak, cosine decay to 1e-5
warmup       = 2000 steps
total steps  = 50000   (~400M–800M tokens; the script states 800M at 16k tok/step)
batch        = 8 × 2048 = 16k tokens/step
optimizer    = AdamW, beta=(0.9, 0.95), wd=0.1
grad_clip    = 1.0
precision    = bf16 mixed precision, fp32 master weights
frozen       = everything except the N new MTP blocks
trained      = ~120M params/head × 4 = 480M params
head init    = load mtp.layers.0 into all N heads, then add N(0, 0.02) noise
```

### 2.2 Forward sketch (per training step)

```python
with torch.no_grad():
    out  = main(input_ids, output_hidden_states=True)
    h_all = out.hidden_states[-1]            # [B, T, n_embd]
total_loss = 0
for k, head in enumerate(heads):
    h_shift  = h_all[:, :-num_heads-1, :]
    prev_tok = input_ids[:, k : k + h_shift.size(1)]
    target   = input_ids[:, k+1 : k+1 + h_shift.size(1)]
    logits   = main.lm_head(main.model.norm(head(h_shift, main.model.embed_tokens(prev_tok))))
    total_loss = total_loss + F.cross_entropy(logits.flatten(0,1), target.flatten())
total_loss.backward()
```

**Tap-point caveat:** the existing MTP head does its own normalization, so the desired
hidden tap is the last **block output** (pre-final-norm), not the post-norm state. Use
`output_hidden_states=True` and take the last block output — see the transformers
`Qwen3ForCausalLM` internals (noted in the data-builder script).

---

## 3. The data pipeline

From [`training/build_training_data.py`](training/build_training_data.py) (design-only;
~100 GPU-hours to process 1B tokens). Produces `(h_t, target_{t+1..t+N})` pairs for every
position.

**Output row (parquet):** `hidden_state` float16 `[n_embd=5120]` (optional),
`targets` int32 `[num_heads=4]` (`token_{t+1}..token_{t+N}`), `prev_tokens` int32
`[num_heads=4]` (`token_t..token_{t+N-1}`; head_k trains on `embed(prev_tokens[k])`),
`seq_id` int64, `pos` int32.

**Storage math (1B tokens):** row ≈ `5120·2 + 4·4 + 4·4 + 8 + 4 ≈ 10.3 KB` →
~10.3 TB uncompressed, ~4 TB with zstd parquet (hidden states dominate, fp16 entropy ~6
bits). **Recommended alternative:** store only token ids (36 B/row → **36 GB total**) and
**recompute hidden states on the fly** during training — hidden states at scale are too big
to store.

**Recommended corpus:** FineWeb-Edu (`HuggingFaceFW/fineweb-edu`) 10B sample, mixed 80%
FineWeb-Edu + 10% code (StarCoder subset) + 10% math (OpenWebMath); tokenized with the
Qwen3.5 tokenizer, packed into 2048-length sequences.

---

## 4. Expected accept-rate gains and resulting tok/s

### 4.1 Per-head convergence (from the training script's projections)

| Head (offset) | Start CE (nats) | Converged CE (nats) | Accept p (argmax vs main argmax) |
|---|---|---|---|
| head_0 (+1) | ~3.2 | ~2.6 (≈ main-model CE) | p_1 ≈ 0.78 |
| head_1 (+2) | ~5.0 | ~3.4 | p_2 ≈ 0.68 |
| head_2 (+3) | ~6.0 | ~4.0 | p_3 ≈ 0.58 |
| head_3 (+4) | ~6.5 | ~4.4 | p_4 ≈ 0.48 |

DeepSeek V3 reported similar geometric decay. These p-values feed directly into the
inference cost model in [`per-position-heads-design.md`](per-position-heads-design.md) §5.

### 4.2 The accept-rate → tok/s story (single-head improvement)

The current single Qwen3.5 MTP head gives **79% acceptance** on the main model's own hidden
states (MLX/M4 Max). The improvement ladder and its rationale (from the rollback
TECHNIQUE.md future-work and the frontiers doc):

- **79% (today)** — single transformer-layer MTP head, untuned. Achieves 42.7 tok/s with
  split-recurrence rollback (1.45× over 29.5 baseline).
- **85–90% (fine-tuned)** — fine-tune the single head on the main model's *own* hidden
  states (distillation, no labeled data). Cheap: freeze main, train only ~15 tensors
  (~800M params).
- **95% (multi-layer / per-position)** — multi-layer MTP heads (or per-position heads)
  sustain accuracy at greater draft depth, enabling N>1 drafts without geometric collapse.

Higher accept rate moves the achieved tok/s toward the ceiling (which is **O(1) per token**,
not 2× — see [`research-frontiers.md`](research-frontiers.md)). The gap that better heads
close is the "21% rejection → 1 token instead of 2" term (~12% of the lost throughput) plus
enabling deeper N.

### 4.3 Cost to train

From the script: 50k steps × 16k tokens = 800M train tokens; main forward dominates
(~54 GFLOP/token × 1.15 for head fwd+bwd) ≈ **5.0e19 FLOPs (50 ExaFLOPs)**. On an H100 SXM5
(989 TFLOPS bf16 @ 60% MFU = 594 TFLOPS sustained): wall ≈ `5.0e19 / 5.94e14 ≈ 84,000 s ≈
23 GPU-hours ≈ $80` at $3.50/H100-hour. Add the data pipeline (~24 GPU-hours forward-only):
**Phase 1+2 total ≈ ~50 GPU-hours, ~$175** (~$165 cited in the research README).

---

## 5. The cascade-corrective variant (pointer)

This doc covers the **general** distill-from-logits training of MTP heads (one head, or N
per-position heads sharing one main hidden). A **distinct, deeper variant** for *chained*
drafting — train a **depth-specific corrective head per chain step**, adding an MSE term on
hidden states so the chain self-corrects instead of degrading with depth — is a different
design and is owned in full by [`cascade-mtp-training.md`](cascade-mtp-training.md): the
degradation ladder, the `CE + λ_h·MSE` loss, the three what-failed experiments, the GPU
memory plan, the weight-swap deployment, and the config. It is **not** repeated here.

The two share exactly one principle, which this doc owns and §1 already states: **training
hidden states must come from the exact serving checkpoint** (same dtype, same forward path);
a mismatch gives 0% acceptance.

---

## 6. Reference scripts (copied here)

[`training/`](training/) contains both design-only scripts (neither was executed; both
guard with a `[DESIGN-ONLY] ... NOT RUNNING` exit). See [`training/README.md`](training/README.md)
for origins.

- `build_training_data.py` — corpus → `(h_t, targets, prev_tokens)` parquet shards;
  storage math; tokens-only-recompute recommendation.
- `train_per_position_heads.py` — freeze main, warm-start + noise the N heads, summed CE,
  teacher forcing; hyperparameters, expected loss curves, cost estimate.

> The cascade scripts referenced in [`cascade-mtp-training.md`](cascade-mtp-training.md)
> (`collect_hidden_states.py`, `cascade_mtp_corrective.py`) are described in that doc; they
> live in the vLLM/GH200 ops domain (Agent D) where the W4A16 serving checkpoint runs.

---

## Related

- [`per-position-heads-design.md`](per-position-heads-design.md) — the N-head architecture these scripts train, plus the Phase-0 kill-gate.
- [`cascade-mtp-training.md`](cascade-mtp-training.md) — the depth-specific cascade-corrective variant (CE + MSE).
- [`research-frontiers.md`](research-frontiers.md) §A.2 / §0 — the training frontier and the O(1)/token ceiling better heads chase.
- [`speculative-decoding-on-hybrid-models.md`](speculative-decoding-on-hybrid-models.md) §7 — why training matters on this hybrid model.
- [`recurrent-rollback-technique.md`](recurrent-rollback-technique.md) — the rollback that delivers 79%→42.7 tok/s today (the baseline to improve on).
- [`training/`](training/) — the two copied design-only scripts.

## Provenance

- `qwen-ops/training/build_training_data.py` — copied to [`training/`](training/).
- `qwen-ops/training/train_per_position_heads.py` — copied to [`training/`](training/).
- `recurrent-rollback/docs/TECHNIQUE.md` §7.3 — "train better MTP heads" (79%→85-90%,
  distillation, freeze main, ~15 tensors / ~800M params).
- `qwen-ops/research/designs/CASCADE-MTP-TRAINING.md` — the cascade-corrective variant is
  owned in full by [`cascade-mtp-training.md`](cascade-mtp-training.md); this doc only
  points to it (§5).
- `qwen-mtp-research/README.md` — $165 / Lambda H100 figure.
