# Research Frontiers — Open Directions

The open research directions for MTP speculative decoding on Qwen3.5-27B. This doc is the
**canonical home for the architecture / training / general forward directions** and the
**O(1)-per-token ceiling argument**. Hardware/MLX-specific frontiers (entropy-coded weights,
ANE offload, megakernel, die topology, the bandwidth-gap breakdown) are summarized here for
losslessness but their platform detail and live measurements are Agent C's
[`../03-MLX/`](../03-MLX/) (esp. `bandwidth-and-dispatch.md`).

> Source: `RESEARCH_FRONTIERS.md` (byte-identical in `qwen-ops/research/findings/` and
> `qwen-inference-lab/docs/` — collapsed). Dated 2026-04-05, M4 Max (128 GB, 546 GB/s, 40
> GPU cores, 48 MB SLC).

---

## 0. The ceiling argument: it's O(1) per token, not 2×

The single most important framing for all of these directions. A common misconception is
that MTP/spec-decode caps at 2× (one drafted token). The real ceiling is **O(1) per token**,
set by **memory bandwidth and draft accuracy at depth** — not a fixed multiple.

**The bandwidth wall:** Qwen3.5-27B-4bit weights are 13.7 GB; at ~80% of 546 GB/s that is
**25.1 ms per token** theoretical floor (39.8 tok/s). Measured stock baseline is 29.5 tok/s
(~33.9 ms/tok). **MTP does not reduce time per token — it extracts more tokens per weight
read.**

**The O(1) argument:** with N draft tokens and **100% acceptance**, one **T=(N+1)** forward
reads the weights *once* and produces **N+1 tokens**. DeltaNet recurrence adds only
`N × ~0.02ms` per layer. Therefore:

- At N=8 with perfect accept: `34ms + 8ms = 42ms` for 9 tokens = **4.7 ms/tok = 213 tok/s**.

The practical limit is **draft accuracy at depth**. A **single MTP layer cannot sustain
depth** — acceptance decays geometrically (stock head: 87% → 68% → 54% → 39% → 28% → 21% →
16% over 7 chained steps). This is exactly why the forward training directions
([`per-position-heads-design.md`](per-position-heads-design.md),
[`cascade-mtp-training.md`](cascade-mtp-training.md)) exist: to hold accuracy at larger N so
the O(1) ceiling becomes reachable.

**Current state table (M4 Max):**

| Configuration | tok/s | Bottleneck |
|---|---|---|
| Stock baseline | 29.5 | 13.7 GB weight reads at 80% of 546 GB/s |
| + MTP n=1 w/ rollback | 42.7 | Same bandwidth wall, 1.8 tokens per read |
| Theoretical (100% BW) | 39.8 | Can't exceed bandwidth / weight_size |

---

## A. Architecture & algorithm directions (this domain)

### A.1 Parallel MTP predictions (n=2, n=3)

Run two independent MTP drafts, verify in one T=3 pass. With 79% per-token acceptance:

```
P(both correct)     = 0.79² = 0.624 → 3 tokens
P(first ok, 2nd no) = 0.79 × 0.21 = 0.166 → 2 tokens
P(first wrong)      = 0.21 → 1 token
P(both wrong)       = 0.04 (4%, NOT 20%)

E[tokens/step] = 2.414 (vs 1.79 for n=1)
T=3 verify ≈ 45.9 ms → 52.6 tok/s (+23% over n=1)
```

**Break-even:** the second draft needs only **22% acceptance** to beat n=1. At 79% it is
3.6× above break-even. **n=3 parallel:** 58.3 tok/s (2× baseline) — diminishing but still
positive. Listed **priority 1** (proven math, infrastructure exists).

### A.2 Better / deeper MTP heads (the training frontier)

The geometric-decay problem is the headline architectural limit. Two training paths:
- **Fine-tune the single head** on the main model's own hidden states (distillation, no
  labels) → 79% → 85–90% accept. Cheap (~15 tensors / ~800M params, freeze main).
- **Multi-layer / per-position / cascade heads** → sustain accuracy at depth, enabling N>1.
  Full designs: [`per-position-heads-design.md`](per-position-heads-design.md),
  [`cascade-mtp-training.md`](cascade-mtp-training.md),
  [`training-mtp-heads.md`](training-mtp-heads.md).

### A.3 Reduce per-step overhead & Python-loop overhead (algorithmic)

From the rollback TECHNIQUE.md future-work, architecture/algorithm side:
- **Compile the MTP head into the main model's graph** — eliminate the ~3ms separate
  dispatch. Fuse the GDN split into existing kernels (eliminate ~1ms). Target: 34.5 ms/step
  → 52 tok/s. (Kernel-level execution is Agent C; the *idea* — fold the drafter into one
  graph — is general.)
- **Move accept/reject to the GPU** (argmax + compare as lazy ops); **batch multiple steps
  into one eval**. Target: save 1–2 ms/step.
- **Adaptive splitting** — track accept rate at runtime; below **26.5% acceptance**, switch
  from split-recurrence to standard T=2 checkpoint/redo (cheaper there). See
  [`recurrent-rollback-technique.md`](recurrent-rollback-technique.md) §6.

### A.4 Speculative state prediction (open, likely intractable)

Instead of rolling back and resuming from a checkpoint, **predict what the recurrent state
would have been** if only the accepted tokens were processed. Likely intractable for
nonlinear recurrences (DeltaNet) but plausibly tractable for **linear** ones (Mamba with a
fixed `A`). An open theoretical question tied directly to the irreversibility argument in
[`speculative-decoding-on-hybrid-models.md`](speculative-decoding-on-hybrid-models.md) §2.

### A.5 Cross-architecture validation of split-recurrence rollback

The rollback technique is *theoretically* applicable to Mamba, RWKV, Griffin, Jamba (see
[`recurrent-rollback-technique.md`](recurrent-rollback-technique.md) §5). Reference
implementations and benchmarks on those architectures would validate the generalization
claims — an open empirical direction.

### A.6 Compiler-level integration

Teach `mx.compile` (or an equivalent compiler) to **auto-insert state-capture points** when
it detects a split-recurrence pattern, eliminating the manual protocol and letting the
compiler optimize dispatch scheduling. General compiler direction; MLX specifics are Agent C.

---

## B. Hardware / platform frontiers (lossless summary; detail → Agent C `../03-MLX/`)

These are the M4-Max / Apple-Silicon-specific directions. Captured here for completeness;
platform measurements and kernel work live in Agent C's domain.

### B.1 Entropy-coded weights — and the debunk

**Original claim:** 4-bit quantized weights have Shannon entropy ~1.1–1.5 bits, so rANS
per-row encoding could shrink 13.7 GB → ~5.1 GB (bandwidth floor 25.1ms → 9.3ms) losslessly;
ECQ prototype reported **3.27× throughput (42.9 → 140 tok/s) on M3 Pro**; the missing piece
being a fused rANS-decode + GEMV Metal kernel. References: drxddy/ecq, MLX #3043, EntroLLM
(arXiv:2505.02380), DFloat11 (arXiv:2504.11651).

**CORRECTION (2026-04-05) — debunked empirically** on 1.4 billion 4-bit values from
Qwen3.5-27B-4bit:

```
Shannon entropy: 3.72 bits (NOT 1.1 bits)
Compression ratio: 1.08x (NOT 3.3x)
Savings: 7% (13.7 GB → 12.7 GB)
```

The distribution is a bell curve centered on 8 (mid-range of 0–15); standard affine 4-bit
quantization produces near-uniform distributions. The 1.1-bit figure came from a *different*
quantization format with learned codebooks that deliberately cluster values. **rANS on
standard 4-bit saves only 7% — not worth a custom Metal kernel.** Real entropy gains would
require replacing the whole quantization stack with a compression-aware quantizer (ECQ,
EntQuant) — a different project.

### B.2 MTP head on the ANE — CONFIRMED

The MTP head is pure attention + MLP (no DeltaNet), and the ANE has **its own memory bus**
(zero GPU bandwidth interference). **CONFIRMED (2026-04-05):** CoreML conversion succeeded —
424.7M params, 849.5 MB mlpackage, **2.24 ms median latency** on ANE, exact output match
with MLX, runs entirely within the 34ms GPU forward window. (Earlier estimate cited 0.037ms
compute / 10–22ms weight load.) Enables a **voting scheme**: two ANE MTP predictions
(~4ms total) for free while the GPU runs the main forward; when both agree (64% of the time)
skip the 38ms verification. Listed **priority 2**. Files: `~/models/mtp-head-ane.mlpackage`,
`~/optimizations/qwen-mtp-inference/ane-mtp/`.

### B.3 CPU SME offload

M4 replaced AMX with ARM SME (12 P-cores × ~2 TFLOPS = ~24 TFLOPS FP32). Run MTP on the CPU
stream (`mx.stream(mx.cpu)`, one line in MLX) while the GPU does the main forward — but the
CPU shares the 546 GB/s bus, so 265 MB of MTP weights steals ~0.5ms of GPU bandwidth; net
savings ~1.6ms. **Verdict:** quick win (+7%) but ANE is better (zero bandwidth interference).

### B.4 Die topology — no win

GPU is one contiguous block (~50% of die); 8 LPDDR5X channels striped across two die edges;
weights interleaved across all 8 channels (no locality optimization); SLC 48 MB distributed
near memory controllers; no NUMA on single-die M4 Max; DRAM→GPU wire 3–20mm, propagation
~0.2ns (irrelevant vs 30–40ns access). **Verdict:** no physical proximity optimization
available.

### B.5 SLC caching — no win for autoregressive

SLC is 48–96 MB; model is 13.7 GB (only 0.7% fits). Depth-first scheduling would maximize
SLC reuse but autoregressive generation prevents it (token N depends on N−1 through all
layers). **Batched inference** is the only weight-reuse path: at batch=4, effective
25.1/4 = 6.3 ms/token. Listed **priority 5** (multiplicative with everything else).

### B.6 Megakernel (single persistent dispatch)

One Metal dispatch for the entire forward, software-managed weight streaming, overlapped
load/compute — eliminates the ~500+ per-token dispatches. Challenge: reimplement the forward
in Metal outside MLX. Estimated gain **10–15%** (partly the 2.4ms 33.9−31.4 gap). Reference:
Jia et al. "Compiling LLMs into a Megakernel".

### B.7 The 20% bandwidth-gap breakdown (hardware physics)

Why 80% and not 100% of 546 GB/s:

| Source | Estimated loss |
|---|---|
| DRAM per-bank refresh cycles | 3–4% |
| Memory controller protocol (activate/precharge/CAS) | 4–5% |
| SLC contention (CPU/display/IO) | 2–4% |
| Scale/bias cache-line waste (128B line, 2–4B used) | 1–2% |
| Kernel dispatch overhead | ~1% |
| **Total** | **~11–16%** |

The rest is instruction scheduling / occupancy. **Verdict:** this gap is hardware physics
(DRAM refresh, protocol timing, cache-line granularity) — not software-fixable. Apple's
`qmv_fast` is already optimal (no threadgroup memory, register-based SIMD, bank-parallel).

---

## C. Combined projections

| Stack | Effective BPW | BW floor | With MTP n=2 | tok/s |
|---|---|---|---|---|
| Current (4-bit + MTP n=1) | 4.0 | 25.1 ms | 1.79 tok / 41.9 ms | 42.7 |
| + Parallel MTP n=2 | 4.0 | 25.1 ms | 2.41 tok / 45.9 ms | 52.6 |
| + Mixed 3/4-bit | 3.5 | 22.0 ms | 2.41 tok / 42.0 ms | 57.4 |
| + Entropy coding | ~1.5 | 9.3 ms | 2.41 tok / 29.3 ms | 82.3 |
| + Entropy + n=3 parallel | ~1.5 | 9.3 ms | 2.91 tok / 33.3 ms | 87.4 |

> **Caveat:** the entropy-coding rows assume the ~1.5 BPW figure that §B.1 **debunked** to
> 3.72 bits / 1.08× on standard 4-bit. The transformative 80–90 tok/s "prize" therefore
> depends on a compression-aware quantizer that does not yet exist for this stack. The
> robust near-term gains are **parallel MTP** and **ANE offload**.

**Priority order (as stated in the source):**
1. Parallel MTP n=2 — proven math, 22% break-even, infrastructure exists.
2. MTP on ANE — zero-cost drafting, enables deeper speculation.
3. Entropy-coded weights — transformative *if* a compression-aware quantizer is used
   (standard-4-bit version debunked).
4. Mixed 3/4-bit requantization — immediate, supported by MLX today.
5. Batched inference — multiplicative with everything above.

---

## D. Confirmed dead ends (don't re-explore)

From the matmul/optimization teams (kept so nobody burns time here again):

- **Kernel fusion (rms_norm into matmul)** — synthetic bench showed 10.4ms saved; real model
  +0.76ms *slower*. `mx.compile` already handles the dispatch pipeline. (But see the
  rollback dispatch-barrier note: fusing *between matmuls* did save a measured 2.0ms — the
  conflict is about *where* the barrier sits. Detail: Agent C.)
- **qmv_fast parameter tuning** (r=8, 4 simdgroups) — both slower (occupancy).
- **Group size 128** — 2.3× quantization error for 2.8% speed. Bad trade.
- **GPU-resident loop** — `mx.compile` is a dispatch scheduler, not a kernel fuser. No gain.
- **CPU draft model** — 3716 ms/tok, no Metal acceleration on CPU.
- **DeltaLLM weight sharing** — 10–15% accuracy drop for 12–25% compression. Bad trade.
- **SLC-aware tiling for M=1** — weights read once, no reuse; tiling changes nothing.
- **Morton/Z-order weight layout** — harmful; breaks sequential DRAM burst reads.
- **FP16 accumulation in qmv** — 0% gain (memory-bound, not compute-bound).
- **Additional SIMD shuffle tricks** — already optimal.
- **Weight layout interleaving** — ~1% possible from interleaved scale/bias; not worth it.
- **CPU SLC pre-warming** — pseudo-random SLC eviction makes retention unreliable.

---

## E. M5 timeline (context)

M5 base shipping now/imminent (April 2026); M5 Pro/Max expected Q4 2026. Key ML feature: GPU
Neural Accelerators (40 matrix-multiply units in GPU cores, Metal 4 TensorOps). Expected
bandwidth ~600–650 GB/s (LPDDR5X-9600). **Does NOT fundamentally change the story** — the
bandwidth wall remains, just shifted 10–15%.

---

## Related

- [`speculative-decoding-on-hybrid-models.md`](speculative-decoding-on-hybrid-models.md) §8 — the same O(1)/token ceiling argument in the core theory.
- [`recurrent-rollback-technique.md`](recurrent-rollback-technique.md) — the technique behind the 42.7 tok/s current state (and §6 adaptive-splitting referenced in §A.3).
- [`per-position-heads-design.md`](per-position-heads-design.md) + [`cascade-mtp-training.md`](cascade-mtp-training.md) + [`training-mtp-heads.md`](training-mtp-heads.md) — the head-training directions (§A.2).
- [`../03-MLX/bandwidth-and-dispatch.md`](../03-MLX/bandwidth-and-dispatch.md) — the M4 Max bandwidth-wall and dispatch-barrier measurements behind §B.
- [`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md) — the measured cross-platform numbers; [`../00-OVERVIEW/glossary.md`](../00-OVERVIEW/glossary.md) for terms.

## Provenance

- `qwen-ops/research/findings/RESEARCH_FRONTIERS.md` = `qwen-inference-lab/docs/RESEARCH_FRONTIERS.md`
  (byte-identical, **collapsed** — this is the canonical home for the architecture/training/
  general directions per the agent split).
- O(1)-per-token ceiling & 213 tok/s @N=8: also `recurrent-rollback/docs/TECHNIQUE.md` §7.
- Geometric accept decay (87%→16%): `qwen-ops/research/designs/CASCADE-MTP-TRAINING.md`.
- Hardware/MLX-specific frontier detail (entropy kernel, ANE, megakernel, dispatch barriers):
  cross-referenced to Agent C [`../03-MLX/`](../03-MLX/).
