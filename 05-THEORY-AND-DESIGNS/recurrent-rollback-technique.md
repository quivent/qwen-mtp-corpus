# Split-Recurrence Rollback — The Generalizable Technique

**Zero-cost speculative-decoding rollback for hybrid attention/recurrent models.** This is
the canonical, architecture-agnostic writeup of the split-recurrence rollback technique.
The reference MLX/DeltaNet implementation is copied into
[`recurrent-rollback-src/`](recurrent-rollback-src/) and pointers to the platform
integrations are at the bottom.

> Theory context: why recurrent state is irreversible in the first place is in
> [`speculative-decoding-on-hybrid-models.md`](speculative-decoding-on-hybrid-models.md).
> This doc assumes that and explains how to roll back anyway at near-zero cost.

---

## 1. The problem in one paragraph

Speculative decoding drafts K tokens, verifies them in one batched forward, and on
rejection rolls back to the last accepted position. For pure-attention models, rollback is
`KV_cache = KV_cache[:i+1]` — O(1), append-only. Hybrid models (DeltaNet, Mamba, RWKV,
Griffin, Jamba) add **recurrent layers** whose fixed-size state matrix is updated
nonlinearly: `state_t = gate · state_{t-1} + key · (value − key^T @ state_{t-1}) · beta`.
This state **cannot be trimmed** — token `t`'s information is irreversibly mixed in. So the
cheap-rollback assumption that speculative decoding depends on does not hold.

---

## 2. Previous approach: checkpoint + restore + redo (the baseline)

Before the verification forward, deep-copy the recurrent state. On rejection at position
`i`, restore the checkpoint (the state before *any* drafted token) and **redo** the forward
pass for the accepted tokens `0..i`.

```python
checkpoint = mx.array(model.rnn_state)   # force a real copy (NOT a Python ref)
logits = model.forward(draft_tokens)     # updates recurrent state
accepted = verify(logits, drafts)
if accepted < len(drafts):
    restore_state(model, checkpoint)
    model.forward(draft_tokens[:accepted])  # REDO accepted tokens
```

**Cost (Qwen3.5-27B, M4 Max, 546 GB/s):**

| Operation | Cost |
|---|---|
| Full forward pass (64 layers) | ~34ms |
| State checkpoint (~78 MB copy, BF16) | ~0.3ms (≈0.14ms by bandwidth math) |
| State restore | ~0.3ms |
| **Redo forward pass** | **~34ms × (accepted/total)** |

With K=3 draft tokens at a 21% per-token rejection rate:
- Rejection probability per step ≈ `1 − 0.79³ ≈ 51%`
- Average redo when rejecting ≈ `34ms × 1.5/3 ≈ 17ms` (re-process ~half the tokens)
- **Expected redo cost per step ≈ `0.51 × 17ms ≈ 7–8.7ms`**

At a 30 tok/s baseline (33ms/tok) the redo penalty consumes **~26%** of the time budget,
making spec decoding barely break even. The **deep-copy detail matters**: a naive Python
reference copy (`checkpoint = model.state`) is *wrong* — the state gets mutated underneath
you; you must force a real array copy. The copy itself is cheap (~78 MB / 546 GB/s ≈
0.14ms); **the redo dominates.**

> **State-size note (adjudicated).** The per-request DeltaNet recurrent state on
> Qwen3.5-27B is `ssm_state = (num_v_heads=48, head_v_dim=128, head_k_dim=128) = 786,432
> elements/layer` plus `conv_state = (conv_kernel_size−1=3, conv_channels=10240)`, over 48
> DeltaNet layers → **75.5 MB ssm-only / ~78 MB with conv (BF16); 151 MB FP32**. Authoritative
> source: [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md).
> The source repo's README/`delta_net_rollback.py` used a *wrong illustrative shape*
> (`14 heads × 256 × 256 = 84 MB`); the per-head `head_k_dim × head_v_dim` matrix structure
> it describes is correct, only the head count (14 vs 48) and per-head dims (256 vs 128)
> were off. All memory figures in this doc use the adjudicated 48·128·128 shape.

---

## 3. The solution: split the recurrence, batch the matmuls

**Key insight: matmuls don't care about sequence length, but recurrences do.** A hybrid
recurrent layer is:

```
input_proj (batched matmul) → recurrence (sequential) → output_proj (batched matmul)
```

The projections process all `T` tokens independently (`Y[b,t,:] = X[b,t,:] @ W`). Only the
recurrence is inherently sequential. So:

1. **Batch the matmuls at T=N** (all draft tokens together) — same cost as before, better
   GPU utilization.
2. **Split only the recurrence into T=1 steps** — minimal overhead, since the recurrence
   was already sequential by nature.
3. **Capture an intermediate state ref after each step** — zero-copy because arrays are
   immutable.
4. **On rejection at position `i`: restore state ref `i`, trim the KV caches** — no redo.

```
Before: input_proj(T=N) ── recurrence(T=N) ── out_proj(T=N)
                              ↑ only final state available → must redo on rejection

After:  input_proj(T=N) ── rec(T=1)→rec(T=1)→…→rec(T=1) ── out_proj(T=N)
                            ↑save     ↑save        ↑save
                            state[0]  state[1]     state[N-1]

        On reject at i: restore state[i-1], trim KV. No redo. No recomputation.
```

### 3.1 Correctness proof (matmuls distribute over sequence concatenation)

For `X = concat(x_0, …, x_{T-1})` along the sequence axis and weight `W`:

```
matmul(concat(x_0,…,x_{T-1}), W) = concat(x_0@W, …, x_{T-1}@W)
```

because each `(b,t)` slice is computed independently: `Y[:,t,:] = X[:,t,:] @ W = x_t @ W`.
Therefore `in_proj` and `out_proj` give identical results whether applied to the full
`(B,T,D)` tensor or to per-token slices concatenated afterward. The recurrence is
sequential either way, so splitting it changes nothing about the computation — only the
**granularity of state capture**. We still batch the matmuls because batched matmuls have
strictly better GPU utilization; splitting them would add overhead for no benefit.

### 3.2 Why zero-copy refs work (array immutability)

In MLX (and JAX), arrays are **immutable** — operations produce new arrays, never mutate in
place. When the recurrence computes `new_state = f(old_state, input)`, `old_state` is
untouched; both arrays coexist in GPU memory until garbage collected. Saving a reference is
therefore **free**:

```python
saved_states = []
for t in range(N):
    output, state = gdn_step(input_t, state)
    saved_states.append(state)   # reference to an immutable array — no copy
```

After the loop, `saved_states[0]` still points to the state after token 0 even though
`state` has been updated N times. Each saved ref adds one 8-byte pointer to a Python list;
the state arrays already exist (they'd have been computed and discarded in the standard
forward). The split merely *retains* N intermediate arrays instead of just the final one.

**Memory overhead** for Qwen3.5-27B: state per recurrent layer ≈ 1.573 MB (ssm-only, BF16;
786,432 elements × 2 bytes), ~1.634 MB including conv; additional retained states =
`(N−1) × 48 layers × ~1.6 MB`. At N=4: `3 × 48 × 1.573 MB ≈ 226 MB` ssm-only (~235 MB with
conv) — modest vs the model's 13.7–15.3 GB weight footprint and the KV cache. In **PyTorch**
(mutable tensors) you would need an explicit `.clone()`, but the cost is still small —
just the state matrices, not the full hidden states.

---

## 4. Cost analysis (the payoff)

### 4.1 Headline cost table (README form, N draft tokens)

| Component | Cost |
|---|---|
| Extra GDN kernel dispatches | N tokens × 48 layers × ~0.02ms = **~1ms** |
| Saved redo (eliminated) | 34ms × 21% reject = **~7ms** |
| **Net savings** | **~6ms per verification step** |

The split adds ~1ms of dispatch overhead but eliminates ~7ms of redo, for a **net ~6ms per
step**. At 30 tok/s, that is roughly **+2 tok/s**.

### 4.2 Detailed cost model (TECHNIQUE.md form, N=4)

Definitions: `C_recurrence_dispatch` = one T=1 recurrence dispatch ≈ 0.02ms;
`C_recurrence_fused` ≈ 0.01ms (fused T=4, marginal over T=1); `N` = 4 (1 verified + 3
draft); `C_forward` ≈ 34ms; `P_reject` ≈ 0.51 (K=3 draft, 21% per-token);
`E[accepted/N]` ≈ 0.5.

- **Split cost:** `C_batched_matmul + N·C_recurrence_dispatch`
- **Original cost:** `C_batched_matmul + C_recurrence_fused + C_redo`, where
  `C_redo = C_forward · P_reject · E[accepted/N]`
- **Extra dispatch overhead per layer:** `4·0.02 − 0.01 = 0.07ms`; over 48 recurrent
  layers: `48 × 0.07 = 3.4ms`
- **Eliminated redo:** `34 × 0.51 × 0.5 = 8.7ms`
- **Net savings: `8.7 − 3.4 = 5.3ms per verification step.`**

> Two numbers (5.3ms and 6ms; 2.9ms and 3.4ms overhead) appear because the README and the
> deep-dive use slightly different per-layer dispatch accounting (per-token diff vs
> fused-vs-split diff). Both land in the **~5–6ms net saving** range. Reported elsewhere as
> the rollback cost itself being O(48) reference assignments ≈ **~0.001ms**.

### 4.3 Break-even condition

The split wins when its overhead is less than the expected redo:

```
N · C_recurrence_dispatch − C_recurrence_fused  <  C_forward · P_reject · E[accepted/N]
```

For Qwen3.5-27B: `4 × 0.02 × 48 = 3.84ms  <  34 × 0.51 = 17.3ms` — comfortably above
break-even.

### 4.4 Measured end-to-end (MLX, 79% acceptance, split-recurrence rollback)

```
Achieved: 42.7 tok/s  (1.45× over 29.5 baseline)
Ceiling:  58.8 tok/s  (2.0×, with 100% acceptance and zero overhead)

Gap breakdown (0.55× lost):
  MTP head + split dispatch overhead (4ms/step):  ~15%
  21% rejection (1 token instead of 2):           ~12%
  eval sync + Python loop per step:                ~5%
```

The ceiling is **not 2×** — it is **O(1) per token** (see
[`research-frontiers.md`](research-frontiers.md) and the theory doc §8). The practical
limit is draft accuracy at depth.

---

## 5. Applicability

The technique applies to **any architecture with non-trimmable recurrent state**. Batch the
input/output projections, split the recurrence into per-token steps, save state refs.

| Architecture | Recurrent state | State update | Why non-invertible / notes |
|---|---|---|---|
| **DeltaNet** | `ssm_state` per-head `head_k_dim × head_v_dim` (Qwen3.5: `48 × 128 × 128`) + `conv_state` | `g·S + k·(v − k^T@S)·beta` | nonlinear (retrieval in delta) |
| **Mamba / Mamba-2** | `conv_state` + `ssm_state` | conv (FIFO) + selective SSM `A_bar·S + B_bar·x` | `A_bar`, `B_bar` input-dependent; conv *could* be trimmed by popping/restoring but it's simpler to save the ref |
| **RWKV** | `time_state` | `exp(−w)·S + k·v^T` | channel-wise exponential decay accumulates all history |
| **Griffin** | `rg_lru_state` | `a·S + (1−a)·(W_x@x)`, RG-LRU | input-dependent real gate `a` |
| **Jamba** | mixed Mamba + attention | Mamba layers use SSM state | Mamba layers non-trimmable; attention layers trim normally |

**Mamba detail:** `conv_state` is a FIFO sliding window (last `d_conv−1` inputs) —
trimmable in principle by saving the evicted entry, but the split-recurrence save-the-ref
path is simpler. `ssm_state` is a linear recurrence with **time-varying input-dependent
coefficients** (`A_bar = exp(delta·A)`, `B_bar = delta·B`, delta input-dependent) — not
invertible because `A_bar_t` depends on the unstored input. **RWKV detail:** undoing a step
needs the exact `k_t·v_t^T` that was added, which is not retained. **Griffin detail:** the
input-dependent gate `a = sigmoid(W_a@x)` is the non-invertible element.

### Generic adaptation pattern

```python
class MyArchRollbackLayer:
    def input_proj(self, x):                 # batch all linear projections at T=N
        return self.module.in_proj(x)
    def recurrence_step(self, projected, state):
        new_state = f(projected, state)      # MUST return a NEW state, never mutate
        return output, new_state
    def output_proj(self, rec_out):          # batch output projection at T=N
        return self.module.out_proj(rec_out)
    def get_state(self): return self._state
    def set_state(self, s): self._state = s
```

---

## 6. When it does NOT help (limitations)

1. **Pure-attention models** — KV trimming is already O(1); no recurrent state.
2. **Very long draft sequences (K ≫ 10)** — retained-state memory grows linearly with K;
   at K=32 on a large model this could be significant.
3. **Very low rejection rates (< 5%)** — redo is already negligible; split overhead may
   exceed the saving. *Adaptive note:* below **26.5% acceptance**, standard T=2 with
   checkpoint/redo is cheaper — track accept rate at runtime and switch modes.
4. **Fused recurrence+matmul kernels** (e.g. FlashLinearAttention) — splitting breaks the
   fusion; the per-dispatch overhead may exceed the assumed ~0.02ms.

---

## 7. Dispatch-barrier note (why some fusions help and some don't)

A profiling discovery relevant to the cost model: the gap between theoretical bandwidth
time (25.1ms) and actual GPU time (36.1ms) is dominated by **dispatch barriers between
kernels** (L2 cache-coherency syncs), not kernel execution. Measured chain:

```
Matmul chain only:    22.4 ms   (weights pipelined, no barriers)
Matmul + norm chain:  31.0 ms   (+8.6ms from norm dispatch barriers)
Full model:           36.1 ms   (+5.1ms from GDN, activations, reshapes)
```

Each barrier is ~5–15µs but 500+ per forward aggregate to ~14ms. **Key consequence for the
split:** reducing dispatches *within* a small subsystem (fusing two GDN T=1 calls into one
T=2) shows ~0ms improvement because those barriers hide behind adjacent matmul work — which
is exactly why the per-token split adds so little real time. Reducing barriers *between
matmuls* (fusing rms_norm into the matmul kernel) saves real time: measured **2.0ms** saved
(256 barrier eliminations) via a custom kernel. Deeper MLX-specific profiling is Agent C's
domain.

---

## 8. Reference implementation (copied here)

[`recurrent-rollback-src/`](recurrent-rollback-src/) contains the MLX reference (origin:
`recurrent-rollback/`). See that folder's README for file origins. Summary:

- `src/split_recurrence.py` — architecture-agnostic protocol + forward pass.
  - `RecurrentLayer` protocol: `input_proj`, `recurrence_step` (returns NEW state),
    `output_proj`, `get_state`/`set_state`.
  - `RollbackPoint` dataclass: `position`, `recurrent_states {layer→state ref}`,
    `kv_cache_lengths {layer→len}`.
  - `split_recurrence_forward(...)` → `SplitForwardOutput(hidden_states, logits,
    rollback_points)`. For each layer: attention layers run normally and record KV length
    per position; recurrent layers do `input_proj(T=N)` → per-token `recurrence_step`
    saving a state ref into `rollback_points[t]` → `concat` → `output_proj(T=N)`.
  - `rollback_to(point, layers, cache)` — O(num_layers) ref assignments: `set_state` for
    recurrent layers, `_trim_kv_cache` for attention layers.
- `src/delta_net_rollback.py` — DeltaNet-specific (`DeltaNetState{conv_state, rnn_state}`,
  `DeltaNetRollbackLayer`). The `recurrence_step` does causal conv1d (shift-and-append
  conv_state) + SiLU, then the GDN delta rule, returning new `(conv_state, rnn_state)`.
  `wrap_model_layers(model)` inspects each layer and tags it attention/recurrent.
- `examples/mtp_speculative_decode.py` — full MTP-drafted spec loop with split-recurrence
  rollback and a `stats` dict (accept/reject, rollbacks, estimated redo time saved).
  Uses Qwen EOS token `151643`.
- `examples/simple_example.py` — minimal correctness check: prefill, process 2 tokens,
  `rollback_to(rollback_points[0])`, verify the restored state matches exactly (diff < 1e-8).

---

## 9. Platform integrations (cross-links)

- **MLX (Apple Silicon)** — the live implementation, kernel fusion, and the M4 Max journey
  are Agent C's [`../03-MLX/`](../03-MLX/). The reference code in
  [`recurrent-rollback-src/`](recurrent-rollback-src/) is MLX.
- **llama.cpp (host-side rollback)** — the snapshot/restore primitives,
  `llama_memory_seq_force_recurrent_pos`, and the cache-bookkeeping bug that this technique
  side-steps are Agent B's [`../02-LLAMACPP/`](../02-LLAMACPP/).
- **vLLM / modal (GPU)** — DeltaNet state snapshot/restore at GH200 scale (~78 MB/request BF16)
  inside the modal self-spec loop is Agent D's [`../04-VLLM-GPU/`](../04-VLLM-GPU/).
- **Theory siblings** — the irreversibility argument and the four rollback strategies are in
  [`speculative-decoding-on-hybrid-models.md`](speculative-decoding-on-hybrid-models.md)
  (§2–§3); the O(1)/token ceiling and open directions are in
  [`research-frontiers.md`](research-frontiers.md). Measured numbers:
  [`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md).

---

## 10. Citation

```bibtex
@software{kornreich2026recurrent_rollback,
  author = {Kornreich, Josh},
  title  = {Split-Recurrence Rollback for Speculative Decoding in Hybrid Models},
  year   = {2026},
  url    = {https://github.com/joshkornreich/recurrent-rollback}
}
```

License: MIT.

---

## Provenance

- `recurrent-rollback/README.md` — problem, solution, cost table, applicability, citation.
- `recurrent-rollback/docs/TECHNIQUE.md` (499 lines) — delta-rule math, non-invertibility
  proof, correctness proof, detailed cost model, break-even, deep-copy detail, memory
  overhead, dispatch-barrier analysis, limitations, measured 42.7 tok/s.
- `recurrent-rollback/src/*`, `recurrent-rollback/examples/*` — copied to
  [`recurrent-rollback-src/`](recurrent-rollback-src/).
