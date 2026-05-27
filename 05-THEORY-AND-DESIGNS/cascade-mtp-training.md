# Cascade MTP Training Strategy

Depth-specific cascade heads for *chained* MTP drafting: train a separate head for each
chain depth so acceptance doesn't collapse as the chain gets deeper. Distilled from
`qwen-ops/research/designs/CASCADE-MTP-TRAINING.md`.

> Context: this is the "improve chained single-head drafting" path. The chained-single-head
> recipe is what actually delivered the MLX/llama.cpp wins (see
> [`per-position-heads-design.md`](per-position-heads-design.md) §0). Cascade heads aim to
> push its depth ceiling. General head-training principles are in
> [`training-mtp-heads.md`](training-mtp-heads.md).

---

## 1. The problem

Qwen3.5's MTP head was trained for **depth-1** prediction. Chained 7 times, acceptance
degrades sharply because the head sees drifted hidden states at deeper positions it was
never trained on:

```
depth:    1     2     3     4     5     6     7
accept:  87%   68%   54%   39%   28%   21%   16%
```

## 2. The solution: depth-specific cascade heads

Train a **separate head for each chain depth**. Each head is fine-tuned from the stock MTP
weights on the hidden states produced by the **exact chain of predecessors** it will follow
at inference:

```
Qwen hidden → Head₀ → Head₁(Head₀ output) → Head₂(Head₁ output) → …
```

## 3. Architecture (each head = the stock MTP head)

- `pre_fc_norm_embedding` (RMSNorm 5120)
- `pre_fc_norm_hidden` (RMSNorm 5120)
- `fc` (Linear 10240 → 5120, concatenates `[embed, hidden]`)
- 1 decoder layer (full attention + MLP, **849 MB**)
- `norm` (RMSNorm 5120)

**Shared across all heads:** `embed_tokens` (2.5 GB) and `lm_head` (2.5 GB).

## 4. Training loss — CE + the critical MSE term

```
L = CE(token_pred, target) + λ_h · MSE(output_hidden, ideal_hidden)
```

- **CE** — predict the correct next token at this depth.
- **MSE** — produce a hidden state close to what the FULL model would produce at this
  position.

**The MSE term is critical.** Without it, each head optimizes its own token prediction but
lets the hidden state drift; the next head in the cascade gets degraded input and error
compounds across depths. With it, each head actively reduces accumulated drift, the next
head gets cleaner input, and the cascade **self-corrects**.

## 5. Training data — from the SERVING checkpoint (non-negotiable)

**Source: the SAME model checkpoint that will verify drafts at inference.** If training
hidden states don't match inference hidden states, the head produces mismatched outputs —
**verified 0% acceptance when using cached hidden states from a different run.**

**Collection:**
1. Run the **W4A16 model** (the exact serving checkpoint) on N prompts.
2. At each token position capture: `hidden_states[-1][pos]` (the ideal hidden state) and
   `input_ids[pos+1]` (the token at this position).

**For depth-D training:**
1. Run heads 0..D−1 on the collected hidden states to get the actual chain output.
2. That chain output is the **input** to head D.
3. Target: predict `input_ids[pos+D+1]` and match `hidden_states[-1][pos+D+1]`.

## 6. GPU memory plan

| Component | VRAM |
|---|---|
| Serving model (W4A16, reduced KV) | ~35 GB |
| Training (MTP head + optimizer + embeddings) | ~15 GB |
| Headroom | ~48 GB |
| **Total** | **~50 GB of 98 GB** |

The serving model **must run during training** to generate correct hidden states. Limit KV
with `--gpu-memory-utilization 0.35 --max-model-len 4096`.

## 7. Training config

```bash
# Collect hidden states from the SERVING model via API
python3 collect_hidden_states.py \
    --endpoint http://localhost:8001 \
    --num-prompts 500 \
    --output /tmp/ideal_hidden_states.pt

# Train cascade heads
python3 cascade_mtp_corrective.py \
    --hidden-states /tmp/ideal_hidden_states.pt \
    --stock-head /path/to/stock_mtp_head.safetensors \
    --output-dir /path/to/cascade-heads/ \
    --num-depths 7 \
    --epochs 3 \
    --lr 5e-5 \
    --lambda-h 1.0 \
    --batch-size 8
```

## 8. Deployment — weight swap per chain step

At inference, swap the MTP head weights at each chain step:

```python
for step in range(num_speculative_tokens):
    cascade_heads[step].copy_weights_to(mtp_model)   # swap to this depth's head
    draft_token, hidden = mtp_model.forward(...)     # one draft step
```

Weight swap via `copy_()` is **CUDA-graph compatible** (verified: graphs capture pointers,
not values). Cost: **~0.2ms per swap on GH200.** (CUDA-graph weight-swap mechanics are
Agent D's [`../04-VLLM-GPU/`](../04-VLLM-GPU/).)

## 9. What failed and why

1. **Sibling heads (diversity training)** — trained 3 copies with noise + diversity loss.
   Pushed apart but not toward better predictions. **0% acceptance** — diverged from the
   target distribution.
2. **Agent's cascade (CE-only)** — trained on cached hidden states from a DIFFERENT forward
   pass than inference. **0% acceptance** — hidden-state mismatch between training and
   serving.
3. **Corrective version (fc+norm only)** — skipped the decoder layer. **0% accuracy** — the
   decoder layer (attention + MLP) is essential, not optional.

## 10. What must be different (the five rules)

1. **Hidden states from the SERVING model** — not from a separate bf16 load.
2. **Full decoder layer** in each head — not just fc+norm.
3. **MSE loss on hidden states** — not just CE on tokens.
4. **Same dtype as serving (bfloat16)** — not float32 from training.
5. **Test after EACH depth** — not train all 7 then discover it's broken.

## 11. Expected outcome

If trained correctly, each depth head should maintain **~70–80% acceptance** at its position
(vs the stock head's 87→16% degradation). This would raise expected accepted tokens from
**3.1 to ~5.2**, yielding **~40% throughput improvement**.

---

## Related

- [`training-mtp-heads.md`](training-mtp-heads.md) — general distill-from-logits head training (the principle this variant builds on).
- [`per-position-heads-design.md`](per-position-heads-design.md) — the DeepSeek-V3 per-position alternative (shared hidden vs depth-specific chain).
- [`speculative-decoding-on-hybrid-models.md`](speculative-decoding-on-hybrid-models.md) §8 — the O(1)/token ceiling that deeper accept rates chase.
- [`research-frontiers.md`](research-frontiers.md) §A.2 — where deeper heads sit in the open-directions map.
- [`../04-VLLM-GPU/`](../04-VLLM-GPU/) — CUDA-graph weight-swap mechanics and the W4A16 serving checkpoint these heads train against.
- [`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md) — measured cross-platform numbers this training would aim to beat.

## Provenance

- `qwen-ops/research/designs/CASCADE-MTP-TRAINING.md` — full source (distilled losslessly).
  Cross-references: general training principles → [`training-mtp-heads.md`](training-mtp-heads.md);
  CUDA-graph weight-swap → Agent D's [`../04-VLLM-GPU/`](../04-VLLM-GPU/).
