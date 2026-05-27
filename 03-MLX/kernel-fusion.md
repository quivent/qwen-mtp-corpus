# Kernel fusion — the Metal kernels, and why custom fusion often lost

The MLX track spent weeks (Phase 1, V2–V6) on Metal kernel fusion for the **net result
of +1.7%**. The detailed *why* — the GPU is bandwidth-bound and `mx.compile` already
fuses elementwise ops — is the load-bearing finding. This doc covers the kernels that
**survived** into the shipped path, the ones that **lost**, and the rms_norm+matmul
fusion that targets the 8.6 ms dispatch-barrier overhead.

Code: [`src/generate.py`](src/generate.py), [`src/fused_kernels_t2.py`](src/fused_kernels_t2.py),
and the full V2–V7 exploration in [`src/fused_gdn.py`](src/fused_gdn.py).

---

## The kernels that survived (shipped, used in `mtp_generate`)

### Fused input projection (4 matmuls → 1)
Each DeltaNet (GatedDeltaNet) layer does 4 input-projection matmuls: QKV, Z, B, A.
`patch_model()` concatenates their quantized weights/scales/biases along axis 0 once at
patch time, so decode does a single `mx.quantized_matmul` then slices the result. Saves 3
GPU dispatches per layer × 48 layers. Slicing afterward is (in principle) free pointer
arithmetic — though profiling found slices still cost ~5 µs each as graph nodes (see
[`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md)).

### `fused_conv1d_silu` (decode, S=1)
One Metal kernel: causal conv1d (kernel size 4) over the 3-wide conv state + the new
token, then SiLU, then shift the conv state `[s1, s2, x]`. Uses `fma` and `fast::exp`.
Replaces a conv + activation + state-shift chain. Per-token T=1 path.

### `fused_gdn_step` (the GDN recurrence)
One Metal kernel does the whole gated-delta recurrence step per token:
softplus(`a + dt_bias`) → `g = exp(-A_exp · softplus)` (with `A_exp = exp(A_log)`
precomputed at patch time) → `beta = sigmoid(b)` → RMS-norm + scale of q and k (via
`simd_sum`) → state decay `state *= g` → delta rule `delta = (v − kv_mem)·beta` → state
update + output. Replaces 6+ separate dispatches. Has an optional masked variant
(`fused_gdn_step_v2_mask`) for prefill. Grid `(32, Dv, B·Hv)`, threadgroup `(32, 4, 1)`.
At ~22 µs/layer this is already efficient and memory-bound on the state tensor
`[1,28,128,64]` = 229K elements.

### The T=2 verify kernels (split-recurrence) — `fused_kernels_t2.py`
For speculative verification of `[token, draft]` in one pass, with **intermediate state
capture for zero-cost rollback**:
- **`fused_conv1d_silu_t2`** — processes both tokens through conv1d+SiLU in one dispatch;
  returns `(conv_out_0, conv_out_1, mid_state, final_state)` where `mid_state` is the
  conv state after token 0 (the rollback checkpoint).
- **`fused_gdn_step_with_intermediate`** — runs the T=2 GDN recurrence in one dispatch;
  the T-loop runs twice, snapshots the state registers into `state_mid` after t=0, writes
  `state_out` after t=1. Returns `(output_T2, final_state, mid_state)`.

Together these cut DeltaNet verify dispatches from 4 to 2 per layer, saving ~48 kernel
launches (~1 ms) across the 48 layers. The captured `mid_state` is what makes rollback
free: on reject, restore `c[0], c[1] = (conv_mid, rnn_mid)` — already-computed references,
no recompute. Full loop in [`mtp-self-speculative-mlx.md`](mtp-self-speculative-mlx.md).

---

## The rms_norm + quantized_matmul fusion (the 8.6 ms barrier)

**The discovery** (profiling, 2026-04-05): **8.6 ms per forward pass** is spent on
*dispatch barriers* between RMS-norm kernels and quantized-matmul kernels — not on the
norm compute itself.

- Individual norm dispatch in isolation: ~5 µs.
- Pipelined between two matmuls: ~33 µs per barrier (GPU stalls waiting for L2 coherency
  between norm output and matmul input).
- 256 norm+matmul pairs × 33 µs ≈ **8.6 ms** — a ~170× amplification over the isolated
  cost (5 µs × 256 = 1.3 ms expected vs 8.6 ms measured). In isolation the pipeline is
  empty and the norm executes instantly; in the real model the norm sits on the critical
  path between two matmuls.

**The fix — `fused_rms_norm_qmv`** (two-pass within one kernel dispatch):
1. Pass 1: compute `sum(x²)` for the RMS (x already in L2, ~0.01 ms), `rms_inv = rsqrt(...)`.
2. Pass 2: load `x · norm_weight · rms_inv` and do the normal `qdot` against quantized
   weights (4-bit unpack with `0x000f`/`0x00f0`/… masks and the `/16,/256,/4096` scaling).

**Measured (via `mx.fast.metal_kernel`, 256 matmuls):**
```
Separate: 30.9 ms
Fused:    28.9 ms
Saved:    2.0 ms (6.6%)
```
This is a **lower bound** — `mx.fast.metal_kernel` is slower per-dispatch than the stock
qmv. Full MLX metallib integration (modifying `qmv_fast` itself) is expected to save more;
the upstream `quivent/mlx-fused-qmv` work reports up to **10.4 ms (30%)** on the matmul
pipeline, projecting toward ~75 tok/s when stacked with MTP.

> **Evolving thread (caution).** `RESEARCH_FRONTIERS.md` lists rms_norm-into-matmul under
> "what we are NOT pursuing," noting a synthetic benchmark showed 10.4 ms savings but the
> **real model showed +0.76 ms slower** because `mx.compile` already handles the dispatch
> pipeline. Resolution: the fusion is a *real* win only with full metallib integration
> (replacing the stock kernel); the `mx.fast.metal_kernel` prototype gives +2.0 ms in a
> 256-matmul microbenchmark but does not beat `mx.compile`'s scheduling on the full
> model. It is documented, prototyped, and **not in the shipped 42.7 tok/s path.**

---

## The kernels that LOST (V6 — failed custom fusion)

V6 wrote three custom Metal kernels to fuse ops *between* matmuls:
- `fused_add_rms_norm` — residual add + RMS norm.
- `silu_mul_rms_norm` — SiLU gate × RMS norm.
- `dual_rms_norm` — two RMS norms in one dispatch.

With a deferred-residual pattern (layers return `(h2, m)` so the loop fuses add+norm
across layer boundaries). Target: ~192 fewer GPU launches (930 → 738 per token).

**Result: 28.92 tok/s vs stock 29.36 (−1.5%).** They made things *slower*.

**Why:** `mx.compile` already fuses elementwise ops with RMS norm into fewer dispatches.
The "930 kernel count" was *operations in the traced graph*, not actual GPU dispatches;
after `mx.compile` fusion the real dispatch count is far lower. The opaque
`mx.fast.metal_kernel` calls **broke** the fusion graph — `mx.compile` can't see through
them, so it stopped fusing around them.

**The lesson (stated three ways across the corpus, because it kept mattering):**
- Don't manually fuse ops that `mx.compile` already fuses.
- Profile *actual GPU dispatches*, not traced graph operations.
- `mx.compile` is a **dispatch scheduler, not a kernel fuser** — it reorders and
  coalesces elementwise work but cannot merge across kernel *types* (norm↔matmul,
  activation↔matmul). The dispatch barrier between kernel types is the only real target.

---

## What `mx.compile` does and doesn't do (the mental model)

From the dispatch-barrier profiling:
- It **partially fuses elementwise chains**: silu+mul, sigmoid+mul, add+norm.
- It **cannot fuse across kernel types**: norm+matmul, activation+matmul. Those barriers
  are the optimization target — which is exactly what `fused_rms_norm_qmv` attacks.
- The compiled-layer / monolithic-compile work (V3/V4/V5) only removes *Python dispatch*
  overhead, which `async_eval` already hides behind GPU execution — hence the +1.7%
  async / +4.7% sync split, and the 16 GB memory cost of V5's graph cache.

---

## Actionable fusions still on the table (profiled, estimated)

From the dispatch-barrier profiling, the remaining fusion budget for the 5.1 ms
non-weight, non-norm overhead:

| Fusion | Dispatches eliminated | Est. savings |
|---|---|---|
| rms_norm into matmul (prototyped) | 256 | 2.0 ms measured |
| conv+silu INTO gdn_step | 48 | ~0.46 ms |
| silu_gate as single kernel | 64 | ~0.42 ms |
| silu×rms_norm as single kernel | 48 | ~0.26 ms |
| **Total** | **416** | **~3.1 ms** (+2.0 = ~5.1 ms recoverable) |

Recovering all 5.1 ms would bring the forward pass 36.1 ms → ~31.0 ms; at 1.8 tokens/step
(MTP) that is **58.1 tok/s (1.97× baseline)**. Full breakdown in
[`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md).

---

## Related

- [`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md) — the dispatch-barrier profiling (the 8.6 ms norm barrier, the 5.1 ms budget) these kernels target.
- [`mtp-self-speculative-mlx.md`](mtp-self-speculative-mlx.md) — how the surviving T=2 kernels plug into the generation loop.
- [`the-journey.md`](the-journey.md) — Phase 1 (V2–V6) in chronological context, with the failed V6 fusion.
- [`README.md`](README.md) — the 03-MLX entry page + results table.
- Cross-domain: [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) (the zero-cost rollback the intermediate-state kernels enable).

---

*Sources: `mlx-qwen-mtp/README.md`, `qwen-inference-lab/docs/TIMELINE.md` (V2–V6),
`qwen-inference-lab/docs/DISPATCH_BARRIER_PROFILING.md`, `qwen-inference-lab/docs/RESEARCH_FRONTIERS.md`
(the "NOT pursuing" correction), and the code in `src/generate.py`,
`src/fused_kernels_t2.py`, `src/fused_gdn.py`.*
