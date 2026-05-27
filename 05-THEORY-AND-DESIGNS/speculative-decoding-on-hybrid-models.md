# Speculative Decoding on Hybrid Attention/Recurrent Models

**The core theory.** Why standard speculative-decode assumptions break on a model that
interleaves full-attention layers with recurrent (DeltaNet) layers, and the principles
that every platform-specific implementation in this corpus had to obey. Platform code
lives elsewhere — this doc is the architecture-independent *why*.

> Cross-links: the generalizable rollback mechanism is in
> [`recurrent-rollback-technique.md`](recurrent-rollback-technique.md); the Qwen3.5
> architecture and exact tensor shapes are Agent A's
> [`../01-ARCHITECTURE/`](../01-ARCHITECTURE/); llama.cpp host-side rollback is Agent B's
> [`../02-LLAMACPP/`](../02-LLAMACPP/); MLX kernels are Agent C's
> [`../03-MLX/`](../03-MLX/); vLLM/modal self-spec is Agent D's
> [`../04-VLLM-GPU/`](../04-VLLM-GPU/).

---

## 1. The narrative (read this straight through)

Speculative decoding accelerates autoregressive LLM inference by **drafting** several
tokens cheaply, then **verifying** them in one batched forward pass through the target
model. When the target disagrees with a drafted token, the system **rolls back** to the
last accepted position and continues. The whole scheme rests on one assumption:

> **Rollback is cheap.** You can throw away the work done on the rejected tokens at
> near-zero cost and resume from the accepted prefix.

For pure-attention transformers this assumption is trivially true: the KV cache is an
**append-only sequence**. Token `t`'s key/value vectors are an independent row that never
modifies token `t-1`'s. Rolling back to position `i` is `KV_cache = KV_cache[:i+1]` — an
O(1) pointer/length update.

Qwen3.5-27B is **not** a pure-attention transformer. It is a **hybrid**: 48 DeltaNet
(recurrent) layers + 16 full-attention layers in a strict **3:1 pattern** (three DeltaNet
layers, then one attention layer, repeating 16 times to 64 layers total). The DeltaNet
layers carry a fixed-size **recurrent state matrix** that is updated nonlinearly at every
timestep. That state is **irreversible** — there is no "state at intermediate position",
only the state after processing all tokens in a batch. This single fact breaks the cheap
rollback assumption and forces every implementation in this corpus to do something
non-trivial.

Five distinct consequences follow, each of which bit at least one implementation:

1. **The rollback problem** — recurrent state cannot be trimmed like a KV cache.
2. **Chunking-vs-AR kernel divergence** — the T≥2 batched DeltaNet kernel is
   mathematically equivalent to the T=1 autoregressive kernel but numerically divergent
   in FP16.
3. **Cross-stream `seq_cp` is alias-only on recurrent memory** — you cannot fork a
   speculative tree by copying recurrent state; copies are aliases that get stomped.
4. **The architecture IS the speculative schedule** — the 3:1 DeltaNet:attention layout
   is itself a draft/verify rhythm (3 cheap recurrent tokens, 1 expensive correction).
5. **Single-head spec is hard to win on a hybrid model** — the draft pass costs about as
   much as a main pass, because fixed per-pass overhead (state update, KV bookkeeping,
   graph alloc) dominates the FLOPs, not the head's small block.

---

## 2. Why DeltaNet state is irreversible

### 2.1 The delta-rule recurrence

A DeltaNet layer maintains a per-head state matrix `S ∈ R^{d_k × d_v}` updated each timestep
by the **gated delta rule** (for Qwen3.5-27B the concrete shape is `head_k_dim=128 ×
head_v_dim=128` per head, `num_v_heads=48` heads → `ssm_state = 786,432 elements/layer`,
plus a `conv_state = (conv_kernel_size−1=3, conv_channels=10240)`; ~78 MB/request over 48
DeltaNet layers in BF16. Authoritative shapes:
[`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md)):

```
S_{t+1} = g_t · S_t + beta_t · k_t · (v_t - k_t^T @ S_t)
```

where `g_t` is a scalar decay gate (sigmoid of a learned projection), `beta_t` is a
learned scaling factor (sigmoid), `k_t ∈ R^{d_k}` is the key (after conv1d + SiLU), and
`v_t ∈ R^{d_v}` is the value. The term `k_t^T @ S_t` is the **retrieval**: what the state
currently remembers for key `k_t`. The bracketed term `(v_t - k_t^T @ S_t)` is the
**delta** — the prediction error between the desired value and the current retrieval,
exactly analogous to the classical Hebbian/Widrow-Hoff delta rule.

Expanding, this is an **affine** recurrence:

```
S_{t+1} = A_t @ S_t + B_t
  A_t = g_t · I - beta_t · k_t @ k_t^T      (rank-1 perturbation of scaled identity)
  B_t = beta_t · k_t @ v_t^T
```

`A_t` is **state-independent but input-dependent** — it changes every timestep based on
the current token's projections.

### 2.2 Why it cannot be inverted (and therefore cannot be trimmed)

To "undo" one step (recover `S_t` from `S_{t+1}`) you would need
`S_t = A_t^{-1} @ (S_{t+1} - B_t)`. This fails in practice for two reasons:

- **You must store `A_t, B_t` (or `g_t, beta_t, k_t, v_t`) for every step** — at which
  point you might as well store the intermediate state directly.
- **`A_t` may be singular.** By the matrix determinant lemma,
  `det(A_t) = g_t^{d_k} · (1 - beta_t·||k_t||² / g_t)`; when `beta_t·||k_t||² = g_t` the
  matrix is singular and inversion fails entirely. Even when invertible, computing
  `A_t^{-1}` is a `d_k × d_k` inverse per step per head per layer — far more expensive
  than saving the state.

The state-dependence of the retrieval (`k_t^T @ S_t` depends on what is *read* from `S_t`,
not just what is written) is what makes this strictly harder than a pure linear recurrence
`S_t = g·S_{t-1} + k·v^T` (which *is* invertible when `g ≠ 0`).

**Bottom line:** the state at time `t` is a lossy compression of the entire history
`0..t`. There is no "remove the last token's contribution" operation. This is the
attention/recurrent asymmetry that makes hybrid spec-decode hard.

### 2.3 This is not unique to DeltaNet

Every recurrent/SSM architecture has the same non-trimmable property. The full
applicability table and per-architecture reasoning are in
[`recurrent-rollback-technique.md`](recurrent-rollback-technique.md#5-applicability);
summary:

| Architecture | State update | Why non-invertible |
|---|---|---|
| DeltaNet | `g·S + k·(v − k^T@S)·beta` | nonlinear (retrieval in delta) |
| Mamba / Mamba-2 | `A_bar·S + B_bar·x` (selective SSM) + conv | `A_bar`, `B_bar` input-dependent; conv is sliding window |
| RWKV | `exp(−w)·S + k·v^T` | exponential decay mixes all history |
| Griffin | `a·S + (1−a)·(W_x@x)`, RG-LRU | input-dependent gate `a` |
| Jamba | Mamba layers interleaved with attention | Mamba layers carry SSM state |

---

## 3. The rollback problem and its solutions

Because state can't be trimmed, on rejection you must recover the correct pre-draft state.
The implementations in this corpus chose among four strategies:

1. **Checkpoint + restore + redo** — snapshot the recurrent state before drafting; on
   rejection restore it and **redo** the forward pass over the accepted tokens. Correct
   but expensive: a full forward is ~34ms (Qwen3.5-27B, M4 Max); at a 21% per-token
   reject rate the expected redo is **~7–8.7ms per step**, nearly cancelling the spec
   gains. This is the baseline that the rollback technique improves on.
2. **Split-recurrence rollback** — batch the matmuls at T=N but split *only* the
   recurrence into T=1 steps, saving zero-copy state refs after each step; restore the ref
   on reject. **No redo.** Net ~6ms/step saved, +2 tok/s. This is the canonical
   generalizable technique — full writeup in
   [`recurrent-rollback-technique.md`](recurrent-rollback-technique.md).
3. **In-graph AR loop** — run a T=1 autoregressive loop *inside one graph dispatch*
   (mathematically equivalent to T=1 verify), bypassing the chunking kernel. In llama.cpp
   this is `MTP_VERIFY_FORCE_AR` (commit `987541157`); see §4 for why it solved a
   non-problem.
4. **Accept state contamination as cost** — some "fast-path" variants skip the second
   forward and let the recurrent state carry the rejected token's contribution. Proven
   **broken** on the hybrid model post-bug-fix (see Agent B's
   [`../02-LLAMACPP/the-bug.md`](../02-LLAMACPP/the-bug.md)).

The host-side mechanics — `llama_memory_seq_force_recurrent_pos`, the snapshot/restore
primitives, the cache trim — are platform-specific (Agent B: llama.cpp; Agent C: MLX;
Agent D: vLLM/modal). This doc owns only the principle: **rollback on a hybrid model is
never free; it is either a redo, a saved-ref restore, or an accepted error.**

---

## 4. Chunking-vs-AR kernel divergence (the red herring)

For T≥2 batches, the DeltaNet recurrence is computed by a **chunking** kernel
(`build_delta_net_chunking` in llama.cpp; `chunk_gated_delta_rule` in the vLLM/CUDA
stack). For T=1 it uses an **autoregressive** kernel (`build_delta_net_autoregressive` /
`causal_conv1d`). The two are **mathematically equivalent** but **numerically divergent in
FP16** — the chunked parallel scan accumulates in a different order than the sequential
scan, producing a few ulps of difference.

This was **hypothesis #1** in the llama.cpp investigation for why MTP spec produced garbage
output. It turned out to be a **red herring**: the divergence is real but bounded (a few
ulps), and the actual fault was a one-line host-side cache-bookkeeping bug (see Agent B's
[`../02-LLAMACPP/the-bug.md`](../02-LLAMACPP/the-bug.md)).
Two lessons crystallize here:

- **`MTP_VERIFY_FORCE_AR` doesn't fix what people think it fixes.** The in-graph AR loop
  was added to bypass chunking divergence in the verify path, but throughput barely
  changed under non-contention conditions because chunking was never the bottleneck. A
  useful primitive aimed at the wrong target.
- **The divergence still matters at scale / low precision.** The modal/vLLM precision saga
  (Agent D) shows DeltaNet recurrence at FP16 accumulates error faster than FP32, and the
  attention corrections are *essential* — not merely drift-correcting. The same fused
  CUDA DeltaNet kernels produce slightly different intermediate values when attention
  corrections are absent vs present; the CPU PyTorch fallback does not exhibit this.

---

## 5. Cross-stream `seq_cp` is alias-only on recurrent memory

Tree/multi-sequence speculative variants want to fork B branches and verify them in one
batch. On a pure-attention model you fork by copying KV cells. On a hybrid model,
**`llama_memory_seq_cp` for the recurrent half creates an alias, not a copy.** All forked
branches share one recurrent cell, which gets **stomped by whichever ubatch writes last**.

Consequences for any tree-spec design on a hybrid model:

- True per-branch recurrent state requires either a **deep memory-layer rewrite** or a
  fallback to **per-branch `id_last` duplication** (carrying each branch's last token and
  re-deriving), which in turn drags in the `kv_unified=true` requirement and `n_parallel`
  bumping.
- This is why the branching-tree and perturbed-head-ensemble variants in the corpus never
  produced clean wins: the recurrent state could not be honestly forked.

Principle: **recurrent memory has no cheap fork.** Any breadth in the speculative search
(trees, ensembles, parallel drafts) costs either correctness or a memory rewrite on a
hybrid model.

---

## 6. The architecture IS the speculative schedule

The structural insight: **the 3:1 DeltaNet:attention layout is itself a draft/verify
rhythm — 3 cheap recurrent steps, 1 expensive correction.** (The hybrid layout — 48
DeltaNet + 16 attention in a strict 3:1 pattern over 64 layers — is recapped in §1; the
authoritative layer table is Agent A's
[`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md).)

The attention layers exist to **periodically correct drift in the recurrent state**. Each
DeltaNet state matrix carries a compressed representation of the full sequence that is
*sufficient on its own for short-horizon drafting* (1–5 tokens) — the model does not need
the attention corrections to stay coherent over a short window. So the same model can run
in two modes — a cheap DeltaNet-only "draft" pass and a full 64-layer "verify" pass — with
no separate draft model and no extra weights. The speed advantage grows with sequence
length because attention is O(N) while DeltaNet is O(1).

This is the principle behind **Modal MTP**, Agent D's self-speculative design. The full
mechanism — the draft-mode / verify-mode layer skip, the layer diagram, the
100%-of-50-tokens validation on Huihui-Qwen3.5-27B-abliterated, the GH200 numbers, and the
precision caveat — is the canonical home in
[`../04-VLLM-GPU/modal-self-speculative.md`](../04-VLLM-GPU/modal-self-speculative.md).

---

## 7. Single-head spec is hard to win on a hybrid model

The honest, post-bug-fix single-head numbers (llama.cpp, M4 Max, Qwen3.5-27B Q4_K_M):

| Config | tok/s | vs plain |
|---|---|---|
| Plain decode | 17.90 | 1.00× |
| K=1 MTP spec | 7.64 | 0.43× |

The MTP draft pass costs **about as much as a main forward pass** on this model, so
single-head spec loses. The reason is structural, not algorithmic: on a hybrid model,
`llama_decode(T=1)` is dominated by **fixed per-pass overhead** — KV cache bookkeeping,
the DeltaNet recurrent state update, and graph allocation — *not* by the FLOPs of the
head's own block. A head whose block is cheap in FLOPs still pays the full fixed overhead.

This is the **load-bearing assumption** behind both forward designs:

- It is why per-position heads need a **Phase-0 kill-gate**: if `head_fwd ≈ main_fwd`, no
  number of heads and no accept rate can win (see
  [`per-position-heads-design.md`](per-position-heads-design.md)).
- It is why the **chained-single-head** recipe (MLX `stacked_v2`) is the lower-risk,
  zero-training-cost path that actually delivered the wins (1.68×/1.73× MLX; 1.99× over
  K=1 vanilla in llama.cpp). See [`per-position-heads-design.md`](per-position-heads-design.md)
  §"The resolved truth".

---

## 8. The O(1)-per-token ceiling (it's not 2×)

A common misconception: MTP/spec-decode caps at 2× because you draft one extra token. The
real ceiling is **O(1) per token**, set by memory bandwidth, not by a fixed multiple.

With **N draft tokens and 100% acceptance**, one **T=(N+1)** forward pass reads the model
weights **once** and produces **N+1** tokens. The DeltaNet recurrence adds only
`N × ~0.02ms` per layer. For Qwen3.5-27B on M4 Max:

- At N=1: 34ms produces ~1.8 tokens (79% accept) → the measured 42.7 tok/s.
- At N=8 with perfect accept: `34ms + 8ms = 42ms` for 9 tokens = **4.7ms/tok = 213 tok/s**.

The practical limit is **draft accuracy at depth**, not the verify cost. A *single* MTP
layer cannot sustain accuracy at large N — acceptance decays geometrically with depth (the
stock head degrades 87% → 68% → 54% → 39% → 28% → 21% → 16% over 7 chained steps). This is
exactly the gap that depth-specific cascade heads and per-position heads aim to close. Full
projection and the frontier directions are in
[`research-frontiers.md`](research-frontiers.md).

---

## 9. Principles checklist (for any new hybrid-model spec-decode implementation)

1. **Never assume cheap rollback.** Budget for redo, or implement saved-ref restore.
2. **The recurrent state is the hard part, not the KV cache.** Design rollback around it.
3. **Trust nothing measured before validating output coherence.** Mutual-drift convergence
   makes corrupted drafter+target *agree on garbage*, inflating apparent accept rates.
4. **Chunking≠AR numerically in FP16** — bounded, usually a red herring, but matters at
   low precision when attention corrections are absent.
5. **No cheap fork of recurrent memory** — trees/ensembles cost correctness or a rewrite.
6. **Measure `head_fwd` vs `main_fwd` before any training spend.** Fixed per-pass overhead,
   not FLOPs, decides whether spec can win at all.
7. **The ceiling is O(1)/token, set by bandwidth and draft depth-accuracy**, not a fixed 2×.

---

## Related

- [`recurrent-rollback-technique.md`](recurrent-rollback-technique.md) — the generalizable split-recurrence rollback (the §3 solution in full).
- [`research-frontiers.md`](research-frontiers.md) — the open directions, esp. the O(1)/token ceiling (§8 here).
- [`per-position-heads-design.md`](per-position-heads-design.md) / [`training-mtp-heads.md`](training-mtp-heads.md) — the head designs the §7 kill-gate governs.
- [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) — authoritative 48/16 layer table and DeltaNet state shapes.
- [`../02-LLAMACPP/the-bug.md`](../02-LLAMACPP/the-bug.md) — the cache-bookkeeping bug behind the §4 red-herring.
- [`../04-VLLM-GPU/modal-self-speculative.md`](../04-VLLM-GPU/modal-self-speculative.md) — canonical home for the §6 modal draft/verify mechanism; [`../03-MLX/`](../03-MLX/) and [`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md) for measured numbers.

## Provenance

Synthesized from:
- `modal-mtp/README.md` (3:1 pattern, modal draft/verify, 100%-match validation, precision)
- `qwen-ops/README.md` §"Key Architectural Insight" (3:1, irreversibility, chunking red herring)
- `qwen-mtp-research/README.md` (hybrid notes, chunking-vs-AR, `seq_cp` alias, single-head numbers, MLX truth)
- `recurrent-rollback/README.md` + `docs/TECHNIQUE.md` (delta-rule math, non-invertibility, O(1) ceiling)
- `qwen-ops/research/findings/RESEARCH_FRONTIERS.md` (bandwidth wall, O(1)/token)
- `qwen-ops/research/findings/mlx-reference.md` (single-head truth)
