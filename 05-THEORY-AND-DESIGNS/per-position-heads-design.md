# Per-Position MTP Heads — Design (DeepSeek-V3 style)

**Status: design-only, pre-training. No GPU spent on training.** This is the full spec for
replacing Qwen3.5's single MTP head with N independent per-position heads — *plus* the
resolved truth about what actually delivered the measured MLX wins (it was **not** this
design). Read §0 first; it changes how you read the rest.

---

## 0. CRITICAL — the resolved truth (read this first)

> **The shipped Qwen3.5 checkpoint has exactly ONE MTP block. The MLX 1.68×/1.73× did NOT
> come from trained per-position heads. It came from CHAINED RECURRENT application of the
> single head plus confidence gating — the `stacked_v2` recipe — at zero training cost.**

Earlier docs in this corpus assumed the MLX speedup proved per-position trained heads work.
That assumption was **wrong** and is corrected here. The evidence:

- The MLX safetensors `~/mlx-fork/mtp_weights_vanilla.safetensors` contains exactly **one**
  MTP block: `mtp.layers.0.*` (self-attn, MLP, norms, `mtp.fc` eh-projection). There is no
  `mtp.layers.1` or higher. Same single head as the llama.cpp GGUF.
- The 1.68× on MLX comes from `~/optimizations/qwen-mtp-inference/stacked_v2/stacked_v2.py`,
  a **stacked speculative decoding** runtime strategy, not new weights:
  1. **Chained recurrent application of the single MTP head** — feed the head's own output
     hidden as the next step's `prev_hidden`. The *same* head is called K times; each call
     uses the previous call's post-MTP hidden (not the shared main-model hidden). Code
     signal: `mtp2_logits, h_mtp2 = mtp_head(h_mtp1, embed_d1, lm_head)` — same `mtp_head`,
     second call with the NEW hidden from the first.
  2. **A small (~0.8B) companion draft model** stacked on top: when the MTP chain commits
     to length 2, the 0.8B model drafts position 3 so the main model's T=4 verify can
     commit up to 4 tokens. Code: `draft_model(d1.reshape(1,1), cache=d_cache)`.
  3. **Confidence gating** — chain length is gated by `max(softmax) ≥ threshold`
     (`c1 = mx.max(mtp1_probs); if c1.item() >= confidence_threshold: chain`). Low-confidence
     positions use a short T=2 verify; high-confidence use T=3/T=4.
  4. **Zero training cost.**

The same recipe ported to llama.cpp (components 1 and 3) delivered **1.99× over K=1 vanilla**
(`MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`) — see the chained-recipe writeup
(Agent B / [`training-mtp-heads.md`](training-mtp-heads.md)) and
[`speculative-decoding-on-hybrid-models.md`](speculative-decoding-on-hybrid-models.md) §7.

**Implication for this design:** chained-single-head is **cheaper and lower-risk** than
per-position heads. Per-position heads remain a valid *alternative* path if chaining
plateaus — but they are not what produced the headline numbers, and they must clear a
kill-gate before any GPU is spent.

### 0.1 The Phase-0 kill-gate

> **Measure the wall time of one head-only forward (`build_mtp_head`) on this hybrid model.
> If `head_fwd ≈ main_fwd`, per-position heads CANNOT win regardless of accept rate.**

This is non-negotiable and goes *before* any training. On a hybrid model, a `T=1` decode is
dominated by **fixed per-pass overhead** (KV bookkeeping, DeltaNet recurrent state update,
graph allocation), not by the FLOPs of the head's block. The diagnostic that killed
single-head MTP — "the draft pass costs about as much as a main pass" — proved this: K=1
MTP spec runs at **0.43× of plain decode** (7.64 vs 17.90 tok/s). This is **measured
llama.cpp evidence**, not a projection — see the recipe writeup
[`../02-LLAMACPP/the-recipe.md`](../02-LLAMACPP/the-recipe.md) (plain-decode baseline 17.90
tok/s; chained MTP only 0.78× of plain) and the headline table in
[`../02-LLAMACPP/README.md`](../02-LLAMACPP/README.md) (K=1 spec = **0.43×**, 7.64 tok/s).
If a head-only graph that
bypasses the main model layers and runs *only* the single MTP block (as
`LLM_GRAPH_TYPE_MTP` already does in `src/models/qwen35.cpp:21`) can hit ~10–15ms vs a
~56ms main pass, the design is viable. If not, **neither per-position heads nor chaining can
win** on this model without a different primitive (e.g. pruning the MTP attention to a
single linear layer). Phase 0 is a kill-or-proceed gate.

---

## 1. Goal and shape

Replace the single MTP head with **N independent heads (N=4)**, each predicting a *fixed
relative offset* from a **shared** main-model hidden state — the DeepSeek-V3 design.

```
main fwd → h_t
  head_1(h_t, tok_t)            → logits for t+1
  head_2(h_t, tok_{t+1}_pred)   → logits for t+2
  head_3(h_t, tok_{t+2}_pred)   → logits for t+3
  head_4(h_t, tok_{t+3}_pred)   → logits for t+4
verify (t+1..t+4) in ONE T=5 main pass → commit up to 5 tokens
```

One main forward amortizes over K head forwards. Heads are cheap (a single transformer
block each) — **if** Phase 0 confirms it.

**Target model:** Qwen3.5-27B Q4_K_M (hybrid DeltaNet + full-attention), llama.cpp branch
`mtp-dispatch` @ `b070bed01`. **Baseline:** plain decode 17.90 tok/s; K=1 vanilla MTP 7.64
tok/s (0.43×).

---

## 2. Per-head structure

Each `head_k` is a single dense transformer block, identical in shape to the existing
`mtp.layers.0`:

- `input_layernorm` (RMS, `[n_embd]`)
- `self_attn.{q,k,v,o}_proj` + `q_norm`, `k_norm` (GQA: n_head=40, n_head_kv=8, head_dim=128)
- `post_attention_layernorm`
- `mlp.{gate,up,down}_proj` (intermediate = 17408)
- `eh_proj` `[2·n_embd, n_embd]` — fuses `concat(hnorm(h_t), enorm(embed(prev_tok)))` down to n_embd
- `hnorm`, `enorm` — pre-concat RMS norms
- **Shared with main model:** `embed_tokens`, final `norm`, `lm_head`

> Note: the per-position-heads doc uses `n_embd = 5120` for the shared-head param math and
> the GQA dims above (40/8/128, intermediate 17408). The cascade doc uses RMSNorm 5120 and
> `fc` 10240→5120. Exact authoritative tensor shapes are Agent A's
> [`../01-ARCHITECTURE/`](../01-ARCHITECTURE/); this design carries the numbers as the
> source stated them.

### 2.1 Shared vs dedicated LM head

| Option | Params added per head | Quality | Notes |
|---|---|---|---|
| **Shared LM head** (recommended) | 0 | +0 | matches current Qwen3.5 setup (`mtp_use_dedicated_embeddings: False`); head_k output → main `norm` → main `lm_head` |
| Dedicated LM head | 5120 × 151936 ≈ 778M fp16 / 195M Q4 | marginal | only if heads diverge in output distribution; DeepSeek V3 shares |

**Decision: share.** Per-head params ≈ 480M fp16 / 120M Q4. N=4 → ~480M Q4 added to the
16GB model ≈ **3% overhead**.

### 2.2 Tensor naming

`blk.<64+k>.nextn.*` for k=0..3, slotted into the existing GGUF layer table after the 64
base layers. (GGUF tensor-mapping details are Agent A's domain.)

### 2.3 llama.cpp stub patches (the design-stub diffs)

Three **design-stub** diffs sketch the llama.cpp loader/graph/inference changes this design
would need (load N independent NextN heads; a draft graph generalized to select head `k`; a
drafting entry point that dispatches to head `k` per position instead of chaining the single
head). They are NOT part of the working infrastructure series — they are incomplete stubs.
The canonical copy lives in Agent B's
[`../02-LLAMACPP/patches/per-position-heads-stubs/`](../02-LLAMACPP/patches/per-position-heads-stubs/)
(`loader-stub.patch`, `graph-stub.patch`, `inference-stub.patch` + README). Do not duplicate
them here; reference that copy.

---

## 3. Training objective

Per training example (sequence length L):

1. Run the **frozen** main model once; collect final-layer hidden states `h_1..h_L`
   (pre-norm, the same tap point that feeds the existing MTP head).
2. For each position t and each head k ∈ {1..N}:
   - **Input:** `h_t`, `embed(token_{t+k-1})` — the previous committed token along the
     chain (ground-truth at train time, head_{k-1} argmax at inference).
   - **Target:** `token_{t+k}`.
   - **Loss:** cross-entropy via shared `norm` + `lm_head`.

**Total loss** = `Σ_k w_k · CE_k`, with `w_k = 1.0` for all heads (DeepSeek V3 uses uniform
weights). **Freeze the main model;** train only the N new MTP blocks.

**Teacher forcing:** heads train with ground-truth previous tokens. At inference, the chain
uses argmax from head_{k-1} — a train/test skew. Mitigation: scheduled sampling in the last
10% of training (20% probability of replacing GT with head_{k-1} argmax). Optional;
DeepSeek V3 skipped it and it still worked.

**Warm-start:** initialize each head from the existing single MTP layer's weights, then add
small `N(0, 0.02)` noise so heads don't collapse to identical functions.

---

## 4. N=4 justification

- DeepSeek V3 uses 4.
- Acceptance decays roughly geometrically. At per-token accept `p ≈ 0.75`, expected
  accepted per cycle ≈ `p + p² + p³ + p⁴ = 2.05` drafted-accepted, + 1 bonus = **3.05
  committed**.
- N=6+ hits diminishing returns (`p⁶ ≈ 0.18`) while still costing 6 head forwards.
- Memory: N=4 heads ≈ 480 MB Q4, fits comfortably.

---

## 5. Inference cost model

Let main forward = M ms, head forward = H ms, per-position acceptance = p.

- Cycle time = `M + K·H`
- Expected committed tokens/cycle = `1 + Σ_{k=1}^{K} p^k` (bonus + geometric accepts)
- Throughput = committed / cycle_time

At **M=56ms, H=12ms, K=4, p=0.7**:
- cycle = 56 + 48 = **104 ms**
- committed = 1 + 0.7 + 0.49 + 0.343 + 0.24 = **2.77**
- tok/s = 2.77 / 0.104 = **26.6 tok/s → 1.49× over plain 17.9**

| Scenario | committed | tok/s | speedup |
|---|---|---|---|
| Optimistic (p=0.8) | 3.36 | 32.3 | **1.80×** |
| Base (p=0.7) | 2.77 | 26.6 | **1.49×** |
| Pessimistic (p=0.6) | 2.30 | 22.1 | **1.24×** |

**All of these assume H=12ms.** If `H ≈ M` (as today with single-head spec), throughput is
**worse than plain regardless of p**. This is the Phase-0 gate restated as math.

### 5.1 The README headline variant (M=60ms, H=10ms)

The research README states the ceiling slightly differently: **1 main forward (60ms) + 4
head forwards (4×10ms) = 100ms per cycle, commits up to 4 tokens → theoretical 40 tok/s vs
plain 17.9 = 2.23× speedup ceiling.** The "10ms per head" is an *assumption needing
measurement* — on a hybrid model the heads still walk DeltaNet recurrent state, which may
not be as cheap as a pure-attention head.

### 5.2 Estimated GPU cost

**~$165 in Lambda H100 time** total (Phase 1 data + Phase 2 training). Breakdown from the
training scripts: ~23 GPU-hours for 50k training steps (~$80) + ~24 GPU-hours for the data
pipeline (forward-only over the corpus) ≈ ~50 GPU-hours, ~$175. See
[`training-mtp-heads.md`](training-mtp-heads.md) and the copied scripts in
[`training/`](training/).

---

## 6. Critical risks on this hybrid model

1. **DeltaNet recurrent state is NOT captured by `h_t` alone.** The MTP head inherits the
   main model's final-layer hidden `h_t`, which contains the *attention* output at t — but
   DeltaNet layers also update a recurrent state `S_t` depending on full history. Each
   head_k is a dense-attention block, so it doesn't need recurrent state for its own
   compute, but it **implicitly assumes `h_t` summarizes everything up to t**. Same
   assumption the existing single head makes — *not new* risk — but it means **accept rates
   are lower than on pure-attention models** (the 7.64 tok/s single-head baseline proves
   the head struggles).
2. **Per-head cost vs main-pass cost** — the Phase-0 kill-gate (§0.1).
3. **KV cache per head** — each head has its own self-attention and KV cache growing at
   1/main. N=4 heads at seq 2048 ≈ `4 × (2048 × 8 × 128 × 2 × 2 bytes) ≈ 16 MB` —
   negligible. **The heads' KV caches must be rolled back on misprediction** the same way
   the main KV is rolled back today.
4. **Shared `h_t` for all heads → stale hidden.** head_4 uses the same `h_t` as head_1 even
   though it predicts 4 positions ahead. DeepSeek V3 shows this works because the head learns
   to compensate using `embed(token_{t+3})` as the dominant signal — but the mismatch may
   hurt more on the hybrid model.

---

## 7. Non-goals

- No new MTP loss formulation, no auxiliary losses beyond per-head CE.
- No dedicated LM heads.
- No tree search beyond the existing verify path.
- No training run delivered with the design — scripts are provided
  ([`training/`](training/)); the user decides whether to execute.

---

## 8. How this relates to the other paths

| Approach | Training cost | Code change | Expected speedup |
|---|---|---|---|
| 1. Chained single-head MTP (MLX-style, the one that shipped) | $0 | moderate | 1.3–1.5× IF head_fwd ≪ main_fwd |
| 2. + Stacked 0.8B draft | $0 | larger | 1.5–1.7× (MLX-observed 1.68×) |
| 3. Per-position heads (this doc) | ~$165 | moderate | 1.4–1.8× IF training lands |
| 4. All combined | ~$165 | large | speculative |

**Recommendation:** do Phase 0 first. If head_fwd is cheap, try approach 1 (chaining)
before approach 3 (per-position heads). If head_fwd is *not* cheap, neither works and spec
decode is dead on this hybrid model without a different primitive. The cascade variant
(depth-specific heads) is in [`cascade-mtp-training.md`](cascade-mtp-training.md).

---

## Related

- [`training-mtp-heads.md`](training-mtp-heads.md) — how to actually train these heads (distill-from-logits, freeze-main, cost).
- [`cascade-mtp-training.md`](cascade-mtp-training.md) — the depth-specific cascade alternative for chained drafting.
- [`speculative-decoding-on-hybrid-models.md`](speculative-decoding-on-hybrid-models.md) §7 — why `head_fwd ≈ main_fwd` is the load-bearing kill-gate.
- [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) + [`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md) — authoritative tensor shapes for the head.
- [`../02-LLAMACPP/the-recipe.md`](../02-LLAMACPP/the-recipe.md) — the chained-single-head recipe that actually shipped (the cheaper path).
- [`../02-LLAMACPP/patches/per-position-heads-stubs/`](../02-LLAMACPP/patches/per-position-heads-stubs/) — the llama.cpp design-stub diffs for this design.

## Provenance

- `qwen-mtp-research/docs/per-position-heads.md` — the design spec (structure, objective,
  N=4, cost model, risks, non-goals). **Collapsed** with byte-identical duplicate
  `qwen-ops/research/designs/per-position-heads.md`.
- `qwen-mtp-research/README.md` — §"The MLX truth", §"Per-position MTP heads — design",
  Phase-0 gate, $165, tensor naming, 2.23× ceiling math.
- `qwen-ops/research/findings/mlx-reference.md` — the resolved truth (single head,
  `stacked_v2`, chained recurrence, 0.8B companion, confidence gating, the technique
  hierarchy table).
