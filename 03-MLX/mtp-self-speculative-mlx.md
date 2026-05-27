# MTP self-speculative decoding on MLX

The model drafts its own next token using a small MTP (Multi-Token Prediction) head,
then verifies in a single T=2 forward pass — **no separate draft model needed**. This is
the MLX-specific implementation: the head, the generation loop, the split-recurrence
rollback as it fits the M4 Max generation loop, the 79% acceptance, and the
where-time-goes ceiling.

Code: [`src/mtp_head.py`](src/mtp_head.py), [`src/generate.py`](src/generate.py)
(`mtp_generate`), [`src/fused_kernels_t2.py`](src/fused_kernels_t2.py).

> The **general** split-recurrence rollback technique (not MLX-specific) is owned by
> Theory: [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md).
> The Qwen3.5 MTP head architecture/tensors are owned by Architecture:
> [`../01-ARCHITECTURE/`](../01-ARCHITECTURE/). This doc cross-links rather than re-derives.

---

## The MTP head (15 tensors, one transformer layer)

A single gated-attention transformer layer that predicts token *t+2* from (a) the main
model's last hidden state at position *t* and (b) the embedding of the token at position
*t+1*. Architecture (`MTPHead` in `mtp_head.py`):

1. RMSNorm hidden + RMSNorm embedding.
2. **Concatenate `[embed, hidden]`** (not `[hidden, embed]` — matches GGUF `eh_proj`).
3. FC projection 10240 → 5120.
4. One gated-attention transformer layer (gated Q with split `[queries, gate]`, q/k norms,
   partial RoPE `rope_dims = head_dim·0.25`, SDPA, output × `sigmoid(gate)`, then MLP).
5. Final RMSNorm → **shared** lm_head → logits.

It returns `(logits, h)` where `h` is the **pre-norm hidden** — this is what lets the head
be *chained* (feed `h` back as the next call's `prev_hidden`).

Weight tensors (shapes for the 27B):
```
pre_fc_norm_hidden.weight        [5120]
pre_fc_norm_embedding.weight     [5120]
fc.weight                        10240 -> 5120   (quantized)
input_layernorm.weight           [5120]
q_proj.weight                    5120 -> 12288   (gated: 24 heads × 256 × 2)
k_proj.weight                    5120 -> 1024
v_proj.weight                    5120 -> 1024
o_proj.weight                    6144 -> 5120
q_norm.weight, k_norm.weight     [256] each
post_attention_layernorm.weight  [5120]
gate_proj.weight, up_proj.weight 5120 -> 17408
down_proj.weight                 17408 -> 5120
norm.weight                      [5120]   (final, before shared lm_head)
```
Config defaults: hidden 5120, 24 heads, 4 kv-heads, head_dim 256, intermediate 17408,
rms_eps 1e-6, rope_theta 100000.0, partial_rotary_factor 0.25, group_size 64, bits 4.
Loaded size ~265 MB (31 tensors incl. scales/biases).

### Extraction — they're hidden in the checkpoint
Stock `mlx_lm.convert` (and HF transformers, and vLLM) silently **drop** the 15 MTP
tensors because the Python model class has no `mtp` field — unknown keys discarded without
warning. `extract_weights.py` / `extract_mtp_huihui.py` pull them from the raw safetensors
shards, apply the **`+1.0` norm shift** (Qwen3.5 stores RMSNorm `w` where compute is
`x·(1+w)`), quantize 2D matrices > 1024 elements to 4-bit (group_size=64), keep small
tensors/norms as bf16, and save a standalone ~265 MB file that `load_mtp()` consumes.

### The concat-order bug (resolved)
The FC input must be concatenated `[embed, hidden]`, not `[hidden, embed]` (otherwise:
garbage logits). Canonical explanation — the GGUF `eh_proj` convention and why
embedding-first — lives in
[`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md) (the
`eh_proj` concat insight).

---

## The generation loop (`mtp_generate`, 42.7 tok/s)

```
1. Prefill (T=1), argmax → token.
2. First MTP draft: draft = argmax(mtp_head(h_last, embed(token))).
3. Loop:
   a. T=2 verify of [token, draft] via fwd_t2_rollback (batched matmuls, split GDN).
      verify = argmax(logits2[:,0]);  bonus = argmax(logits2[:,1]).
   b. Pipeline: optimistically compute next MTP draft from h2[:,-1] + embed(bonus).
   c. Single mx.eval(verify, bonus, next_draft) — all resolve together (async pipeline).
   d. If verify == draft:  ACCEPT — keep [draft, bonus] (2 tokens), token=bonus, draft=next_draft.
      Else:                REJECT — rollback() to after token 0, keep verify (1 token),
                           re-draft from h2[:,0] + embed(verify).
   e. Break on EOS.
```

Step (b)+(c) are the **async eval pipeline**: the next MTP draft is enqueued
*optimistically* (assuming accept) before the accept/reject decision is known, and a
single `mx.eval` resolves verify, bonus, and the next draft together — so the accept path
adds near-zero latency for the next draft.

### Split-recurrence rollback as it fits the M4 Max loop
Qwen3.5 is hybrid: **48 DeltaNet (recurrent) layers + 16 attention layers.** Rollback on
reject works differently for each:
- **DeltaNet layers:** recurrent state is captured *during* the T=2 verify by the fused
  `fused_gdn_step_with_intermediate` / `fused_conv1d_silu_t2` kernels, which snapshot the
  state after token 0 (`conv_mid`, `rnn_mid`). On reject, `c[0], c[1] = (conv_mid, rnn_mid)`
  — restoring already-computed references. **Zero-copy** because MLX arrays are immutable.
- **Attention layers:** the KV-cache offset is recorded before the layer, and on reject
  set to `offset + 1` (keep token 0, "un-see" the rejected token). No data to restore.

The accepted path does exactly the same work as non-speculative decode; the rejected path
simply never ran the second recurrence step. No checkpoint copy, no redo. This is what
took the head from **36.9 tok/s** (naive checkpoint/restore + 7 ms redo per reject) to
**42.7 tok/s**. The two T=2 fused kernels also cut DeltaNet verify dispatches from 4 to 2
per layer (~1 ms across 48 layers). Kernel details: [`kernel-fusion.md`](kernel-fusion.md).

---

## Where the time goes (T=1 vs T=2)

```
T=1 baseline:         34 ms → 1 token   → 29.5 tok/s
T=2 split-recurrence: 38 ms → 1.8 tokens → 42.7 tok/s (1.45×)

Breakdown of the 38 ms:
  34 ms  weight reads (13.7 GB at 546 GB/s — same as T=1, bandwidth bound)
   3 ms  MTP head forward (one transformer layer)
   1 ms  split-recurrence overhead (48 extra GDN kernel dispatches)
```

The T=2 forward reads the **same 13.7 GB** of weights as T=1 (bandwidth-bound) but
produces up to 2 tokens — that is the entire point. Only +4 ms (38 vs 34, a 1.12×
overhead ratio) buys a second token at 79% acceptance.

### Acceptance & tokens-per-step
- Single-head, n=1: **79% acceptance** → 1.79 tokens/step (at the 42.7 measurement).
- Adaptive chain (revalidated 2026-04-08, threshold=0.8, max_chain=2): **tokens/step
  2.29, avg chain 1.27, 0 rollbacks / 56 steps** → 51.1 tok/s. (Huihui: 2.17 / 1.17 →
  49.5; see [`huihui-abliterated.md`](huihui-abliterated.md).)

### The 1.45× vs 2.0× gap (n=1)
Theoretical max at 100% acceptance, zero overhead: 34 ms → 2.0 tokens → **58.8 tok/s
(2.0×)**. Achieved 1.45×; the 0.55× lost to:
- 4 ms step overhead (MTP head + split dispatches): ~15%
- 21% rejection (1 token instead of 2): ~12%
- eval sync + Python loop per step: ~5%

---

## The theoretical ceiling — it's O(1) per token, not 2×

The general O(1)-per-token framing (and the bandwidth-floor derivation) is canonical in
[`../05-THEORY-AND-DESIGNS/research-frontiers.md`](../05-THEORY-AND-DESIGNS/research-frontiers.md)
§A and [`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md); below is the MLX instantiation.

With N draft tokens and 100% acceptance, one T=(N+1) forward pass produces N+1 tokens.
Weight reads are bandwidth-bound: T=N costs ~1× (same 13.7 GB, just more output).
DeltaNet adds N × ~0.02 ms per layer = N × ~1 ms total.
```
At N=8: 34 ms + 8 ms = 42 ms for 9 tokens = 4.7 ms/tok = 213 tok/s
```
The only limit is **MTP accuracy at depth**, which degrades because a *single* MTP layer
can't predict 8 tokens ahead. Multi-layer MTP heads (Qwen3.5 ships only 1) would sustain
accuracy at greater depth. This reframes the prize from "2× speculative speedup" to
"amortize the fixed bandwidth cost over as many tokens as the head stays accurate."

---

## Future optimization potential (from the repo)

**Reduce the 38/34 overhead ratio (1.12×):**
- Compile the MTP head into the main model's `mx.compile` graph (eliminate 3 ms MTP
  dispatch).
- Fuse the GDN split into the existing Metal kernels (eliminate 1 ms split overhead).
- Target: 34.5 ms/step → 1.8 / 0.0345 = **52.2 tok/s**.

**Eliminate eval-sync + Python-loop overhead:**
- Async pipeline (start MTP draft while GPU finishes verification — partly done).
- Batch multiple steps into one `mx.eval`.
- Move accept/reject to GPU (argmax + compare as GPU ops).
- Target: save 1–2 ms/step.

**Improve acceptance beyond 79%:**
- The head is a single transformer layer (~265 MB, ~800M params for the 15 tensors).
- Fine-tune on the target model's own outputs (freeze main model, train only the head on
  next-token prediction from the main model's hidden states). Even logit distillation
  (no labels) should push acceptance to 85–90%.
- At 90% acceptance: 1.9 tokens/step → ~48 tok/s. At 95%: 1.95 → ~50 tok/s.

**Parallel MTP (n=2) and ANE offload** — see the frontiers in
[`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md): n=2 projects to 52.6 tok/s
(break-even at 22% second-draft acceptance), and the MTP head runs on ANE at 2.24 ms with
zero GPU-bandwidth interference (CoreML conversion confirmed).

---

## Related

- [`kernel-fusion.md`](kernel-fusion.md) — the T=2 split-recurrence Metal kernels (`fused_gdn_step_with_intermediate`) that make the rollback zero-cost.
- [`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md) — the bandwidth floor, dispatch-barrier profiling, and the n=2 / ANE frontiers this doc points to.
- [`the-journey.md`](the-journey.md) — how the MTP head was discovered (Phase 5) and the chained-head win (Phase 7).
- [`huihui-abliterated.md`](huihui-abliterated.md) — the same loop run on the abliterated variant.
- Cross-domain: [`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md) (the head's tensors + the `eh_proj` concat insight), [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) (the general technique), [`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md).

---

*Sources: `mlx-qwen-mtp/README.md`, `qwen-inference-lab/docs/TIMELINE.md` (Phases 5–6),
`qwen-inference-lab/docs/RESEARCH_FRONTIERS.md`, and the code in `src/mtp_head.py`,
`src/generate.py`, `src/fused_kernels_t2.py`, `src/extract_weights.py`.*
