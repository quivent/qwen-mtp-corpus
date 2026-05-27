# The journey — every approach in order

Chronological record of every approach tried on the M4 Max, what worked, what failed,
and *why*. The dead ends are the interesting parts — they are what tell you the GPU is
bandwidth-bound and that speculative decoding is the only real lever.

Hardware throughout: Apple M4 Max, 16-core GPU, 128 GB unified, 546 GB/s, model
Qwen3.5-27B-4bit (13.7 GB weights). Stock baseline **29.5 tok/s**.

---

## Summary table (the whole arc)

| Approach | tok/s | Change | Verdict |
|---|---|---|---|
| Stock `mlx_lm` (async_eval, greedy) | 29.5 | baseline | — |
| V5 monolithic compile | 30.0 | +1.7% | marginal; +16 GB memory |
| V6 custom Metal fusion kernels | 28.9 | −1.5% | **dead end** — broke `mx.compile` fusion |
| Stock + spec decode (0.8B draft) | 37.6 | +27% | works; needs separate draft model |
| qmv_fast `results_per_simdgroup=8` | 28.3 | −4.2% | **dead end** — register pressure |
| qmv_fast `num_simdgroups=4` | — | −1.5% | **dead end** — register pressure |
| group_size=128 quantization | 30.3 | +2.8% | **dead end** — 2.3× quant error |
| GPU-resident loop (V7) | 29.5 | +0% | **dead end** — `mx.compile` ≠ kernel fuser |
| CPU draft model | 3716 ms/tok | — | **dead end** — no Metal on CPU |
| CoreML/ANE draft | — | — | **dead end** (then later partial win, see below) |
| MTP head (self-speculative) | 36.9 | +25% | works; no draft model |
| MTP + split-recurrence rollback | **42.7** | **+45%** | the shipped win |
| Adaptive MTP chain + batch verify | **51.1** | **+73%** | corpus best (sibling project) |

The biggest gains came from speculative decoding (MTP or draft model), **not** kernel
optimization. The GPU is memory-bandwidth-bound at ~80% of theoretical. The headroom is
in tokens-per-forward-pass, not in faster forward passes.

---

## Phase 1 — Kernel fusion (V2–V6). Net +1.7%.

### V2 — Fused DeltaNet projections + custom Metal kernels
The DeltaNet layers do 4 separate matmuls for input projection (QKV, Z, B, A). Fused
them into 1 `quantized_matmul` by concatenating weight matrices (saves 3 GPU dispatches
per layer). Plus two custom Metal kernels:
- `fused_conv1d_silu`: conv1d + SiLU + conv-state shift in one pass.
- `fused_gdn_step`: RMS norm + scale + gating (softplus + exp) + beta (sigmoid) +
  recurrent state update, all in one kernel.
- Pre-computed `A_exp = exp(A_log)` at patch time; pre-flattened conv weights
  `[conv_dim,1,4] → [conv_dim,4]`.

Result: modest per-layer improvement, masked by other costs. (These kernels *did* survive
— they're in the shipped `generate.py`. See [`kernel-fusion.md`](kernel-fusion.md).)

### V3 — `mx.compile` each DeltaNet+MLP layer
Wrap each DeltaNet layer + MLP into one `@mx.compile` function; weights captured as
closures (constants in the graph). Also fuses MLP gate+up → 1 matmul. Fixed decode
shape B=1, S=1. Eliminated ~50 µs/layer of Python dispatch. Small but real.

### V4 — Compiled attention layers
Same for the 16 attention layers: `_make_compiled_attn_pre` (input_layernorm + fused QKV
3→1 + q/k RMS norms + head transpose) and `_make_compiled_attn_post` (gate·output +
o_proj + residual + post_norm + MLP). SDPA itself stays as
`mx.fast.scaled_dot_product_attention` (already optimal). Incremental gain.

### V5 — Monolithic compile (ONE `mx.compile` for all 64 layers)
`monolithic_decode(h, offset, *flat_cache) -> (logits, *updated_cache)`. All 48 DeltaNet
+ 16 attention layers + final norm + lm_head in one compiled graph. Zero Python dispatch
during decode. KV cache via `mx.put_along_axis` (compilable); RoPE with `mx.array`
offset; flat cache passed as positional args.
- **Result: 29.5 → 30.0 tok/s (+1.7% async); 27.6 → 28.9 tok/s (+4.7% sync).**
- **Memory cost: 15.3 GB → 31.7 GB** (compiled graph cache eats ~16 GB).
- Lesson: async improvement is tiny because the GPU is the bottleneck. `async_eval`
  hides Python overhead, so CPU savings only show in sync mode. The 4.7% sync gain
  confirms real Python overhead was removed, but the GPU doesn't care.

### V6 — Custom Metal fusion kernels. **FAILED (−1.5%).**
Wrote `fused_add_rms_norm` (residual add + RMS norm), `silu_mul_rms_norm` (SiLU gate ×
RMS norm), `dual_rms_norm` (two RMS norms in one dispatch), with a deferred-residual
pattern so layers return `(h2, m)` and the loop fuses add+norm across layer boundaries.
Target: ~192 fewer GPU launches (930 → 738 per token).
- **Result: 28.92 tok/s — slower than stock 29.36.**
- **Why it failed:** `mx.compile` already fuses elementwise ops with RMS norm. The
  "930 kernel count" was *operations in the traced graph*, not actual GPU dispatches;
  after `mx.compile` the real dispatch count is much lower. Our opaque `metal_kernel`
  calls broke the fusion graph that `mx.compile` couldn't see through.
- **Key lesson: don't hand-fuse what `mx.compile` already fuses. Profile actual GPU
  dispatches, not traced graph operations.**

---

## Phase 2 — Speculative decoding (the "almost gave up" story). 29.5 → 37.6.

### Broken benchmark (the false negative)
Setup: Qwen3.5-0.8B-4bit draft, measured with `stream_generate`. **Result: 26.5 tok/s**
— slower than baseline. Conclusion at the time: *"spec decode is dead for DeltaNet — the
verify step can't batch the recurrence."* **This was wrong** — `stream_generate` has
per-token Python overhead that measures differently than `generate`.

### Sanity check on Llama-3.3-70B
To confirm the spec-decode setup worked at all, tested on a standard transformer (no
DeltaNet): **11.9 → 20.1 tok/s (1.69×)**. Spec decode works fine with `mlx_lm`.

### Re-measurement with the correct harness
Back to Qwen with `mlx_lm.generate`: **29.5 → 37.6 tok/s (1.27×)**. It works.
**Lesson: measure correctly before declaring something dead.**

---

## Phase 3 — qmv_fast kernel tuning. **Dead ends.**

The quantized matrix-vector kernel `qmv_fast` dominates decode time. Stock config: 2
simdgroups × 4 rows = 8 output rows per threadgroup; ~219–234 GB/s per matmul; ~439 GB/s
pipelined (80% of 546).
- `results_per_simdgroup=8` (double work per simdgroup to amortize launch): **−4.2%**.
  Register pressure killed occupancy — fewer threadgroups run simultaneously.
- `num_simdgroups=4` (more simdgroups for latency hiding): **−1.5%**. Same issue.
- Both required building/installing a modified MLX from source (rebuilt Python bindings).
- **Conclusion: Apple's `qmv_fast` is well-tuned. 80% bandwidth is near the practical
  ceiling.** The remaining 20% is memory-controller overhead, launch latency, and
  non-matmul compute (see [`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md)).

---

## Phase 4 — Other dead ends.

### group_size=128 quantization. **Dead end.**
Larger groups = fewer scale/bias params = less metadata to read. **Result: +2.8% speed
but 2.3× quantization error.** Unacceptable quality trade. group_size=64 stays.

### GPU-resident autoregressive loop (V7). **Dead end (+0%).**
Unroll N decode steps inside `mx.compile`: embed + forward + argmax + cache update in
ONE compiled graph, CPU dispatches once for N tokens (`_build_gpu_loop`, `gpu_generate`).
**Result: ~0% gain.** `mx.compile` is a **dispatch scheduler, not a kernel fuser** — it
doesn't merge kernels across loop iterations; each iteration still dispatches the same
GPU work. The only savings would be CPU dispatch overhead, which `async_eval` already
hides.

### CPU draft model. **Dead end.**
Run the 0.8B draft on CPU cores while the main model runs on GPU. **Result: 3716 ms/tok.**
MLX's CPU backend has no Metal acceleration — pure CPU BLAS. Unusable.

### CoreML / ANE draft model. **Dead end (initially).**
Compile the 0.8B draft to CoreML for the Neural Engine. Problems: (1) `coremltools`
broken on Python 3.14 (the environment); (2) DeltaNet ops (gated delta recurrence,
custom conv1d) unsupported by the CoreML converter; (3) ANE has limited precision
(fp16) and unknown latency for this arch. Abandoned without a prototype.
- **Later partial win:** the *MTP head* (pure attention + MLP, no DeltaNet) **does**
  convert. CoreML conversion succeeded: 424.7M params, 849.5 MB mlpackage, **2.24 ms
  median latency on ANE**, exact output match with MLX. ANE has a separate memory bus →
  zero GPU-bandwidth interference. See [`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md)
  (frontiers) — this is a research frontier, not yet in the shipped path.

---

## Phase 5 — MTP discovery. 36.9 tok/s.

### Finding the hidden weights
Qwen3.5 was trained with MTP heads, but both MLX's converter and HF transformers strip
them during conversion (the Python model class has no `mtp` field → unknown keys silently
discarded). The weights exist in the original safetensors. Discovery: llama.cpp's GGUF
conversion preserves 4 simplified MTP tensors; the original HF model has the full **15
weight tensors** per MTP head; the architecture had to be reverse-engineered from shapes.

### Architecture (from weight shapes)
Norm both inputs → concat `[embed, hidden]` → FC projection (10240→5120) → one
gated-attention transformer layer (q/k/v/o + q/k norms + MLP) → final norm → shared
lm_head. Full tensor list and shapes: [`mtp-self-speculative-mlx.md`](mtp-self-speculative-mlx.md)
and [`../01-ARCHITECTURE/`](../01-ARCHITECTURE/).

### The critical bug — concat order
Initially concatenated `[hidden, embed]` → garbage logits. **Fix: must be
`[embed, hidden]`** to match the GGUF `eh_proj` convention (the FC weight was trained
with embeddings in the first half of the input dim). Found by comparing argmax outputs
against llama.cpp's MTP — swapping the order made them match. (Canonical:
[`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md) — the
`eh_proj` concat insight.)

### MTP performance
~3 ms per MTP forward (vs ~34 ms main forward); **79% acceptance**; no draft model;
~200–265 MB extra weights (one transformer layer + norms + projection).

---

## Phase 6 — Split-recurrence rollback. 36.9 → 42.7 tok/s (1.45×).

The problem: on draft rejection you must roll back model state. Attention is trivial
(don't advance the KV offset). DeltaNet layers mutate recurrent state during the forward
pass and must be restored.

- **Attempt 1 — checkpoint + restore + redo: 36.9 tok/s.** Copy all DeltaNet states
  before speculation; on reject restore and re-run accepted tokens. The redo costs
  34 ms × 21% reject ≈ 7 ms average overhead per step.
- **Insight 1 — MLX arrays are immutable.** `state = new_state` leaves the old array
  unchanged; you don't need `.copy()`, just save the Python reference. Checkpoint
  becomes free: `saved = [(c[0], c[1]) for c in cache if is_delta]` is just pointers.
- **Attempt 2 — split the GDN recurrence.** Run the verify batch `[token, draft]`
  through all matmuls at T=2 (efficient, same weight reads). Run the GDN recurrence for
  token 0 only, snapshot intermediate state, check acceptance; if accepted run the
  recurrence for token 1, if rejected the state is *already* correct.
- **Result: zero-cost rollback. Final throughput 42.7 tok/s (1.45×).** Accepted path
  does exactly the non-speculative work; rejected path just skips the second recurrence.

The general technique is documented in
[`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md);
the MLX implementation (`fused_gdn_step_with_intermediate`, `mtp_generate`) is in
[`mtp-self-speculative-mlx.md`](mtp-self-speculative-mlx.md) and `src/`.

---

## Phase 7 — Adaptive MTP chain + batch verify. → 51.1 tok/s (1.73×).

The final win comes from the sibling `parallel-mtp-voting` project (`adaptive_mtp.py` /
`stacked_v2.py`; only its logs are in this corpus):
1. **Adaptive MTP chain:** recurrently feed the *single* MTP head its own previous
   output to draft 1–6 positions before verification. Chain length gated by confidence
   `max(softmax) ≥ threshold`. This is **not** per-position heads — it's the same head
   applied K times. (See the evolving-thread note below.)
2. **Stacked 0.8B draft:** when the MTP chain commits to length 2, the small 0.8B Qwen
   drafts position 3 so the main model's T=4 verify can commit up to 4 tokens.
3. **Confidence gating:** low-confidence → short verify (T=2); high-confidence → T=3/T=4,
   amortizing verify cost.

Revalidated 2026-04-08 at threshold=0.8, max_chain=2, max_tokens=128:
**vanilla 51.1 tok/s** (tokens/step 2.29, avg chain 1.27, 0 rollbacks / 56 steps);
**Huihui abliterated 49.5 tok/s** (see [`huihui-abliterated.md`](huihui-abliterated.md)).

> **Evolving thread (resolved).** An early note claimed "the MLX 1.68× comes from trained
> per-position heads." That was **corrected**: the MLX `mtp_weights_vanilla.safetensors`
> contains exactly **one** block (`mtp.layers.0.*`); there is no `layers.1`. The win is
> *algorithmic* — chained recurrent application of the single existing head + a stacked
> 0.8B draft + confidence gating — achieved with **$0 training cost**. The hierarchy of
> techniques (chained single-head vs per-position heads) is analyzed in
> [`../05-THEORY-AND-DESIGNS/`](../05-THEORY-AND-DESIGNS/) (per-position-heads design).

---

## Related

- [`README.md`](README.md) — the 03-MLX entry page + the canonical results table.
- [`kernel-fusion.md`](kernel-fusion.md) — Phase 1 in depth: the kernels that survived vs. lost.
- [`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md) — why the GPU is bandwidth-bound (the finding the dead ends point to); the ANE/CoreML frontier.
- [`mtp-self-speculative-mlx.md`](mtp-self-speculative-mlx.md) — Phases 5–7 in depth: the MTP head + generation loop.
- [`huihui-abliterated.md`](huihui-abliterated.md) — the abliterated-variant benchmark (Phase 7).
- Cross-domain: [`../01-ARCHITECTURE/`](../01-ARCHITECTURE/) (MTP head + the `eh_proj` concat insight), [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) (the general split-recurrence technique).

---

*Sources: `qwen-inference-lab/docs/TIMELINE.md` (canonical), `qwen-inference-lab/README.md`,
`qwen-mtp-research/docs/mlx-reference.md`, `qwen-inference-lab/docs/RESEARCH_FRONTIERS.md`
(ANE/CoreML confirmation). The `qwen-ops/research/findings/TIMELINE.md` copy is
byte-identical and collapsed here.*
