# Bandwidth & dispatch — where the time actually goes

The central finding of the whole MLX track: **decode is memory-bandwidth-bound.** Every
token reads all 13.7 GB of weights once; at 546 GB/s that is a hard 25.1 ms floor. This
doc merges the bandwidth analysis and the dispatch-barrier profiling into one place, plus
the relevant research frontiers (and which of them were debunked).

Hardware: Apple M4 Max, 16-core (40 GPU cores), 128 GB unified, 546 GB/s, 48 MB SLC.
Model: Qwen3.5-27B-4bit.

---

## Weight budget (13.7 GB)

| Component | Size |
|---|---|
| Model weights (4-bit quantized) | 12.2 GB |
| Scales + biases (group_size=64) | 1.52 GB |
| **Total per-token weight read** | **13.7 GB** |

The metadata (scales/biases) is **11.1%** of total — 12.4% of the weight data itself. At
group_size=64, every 64 weights share one scale and one bias. group_size=128 would drop
metadata to ~0.76 GB (saving ~5.5% bandwidth) but increases quantization error **2.3×** —
not worth it. group_size=64 is the right trade-off.

---

## Theoretical limits

| Metric | Value |
|---|---|
| M4 Max peak bandwidth | 546 GB/s |
| Weight reads per token | 13.7 GB |
| **Theoretical minimum latency** | **25.1 ms/tok** |
| **Theoretical maximum throughput** | **39.8 tok/s** (100% BW) |
| At 85% sustained BW | 34.6 tok/s |
| At 80% sustained BW | 31.9 tok/s |

The max assumes reading every weight exactly once with perfect bandwidth utilization. In
practice you also read activations, KV-cache entries, and intermediate buffers.

---

## Actual end-to-end measurements

| Configuration | ms/tok | tok/s | BW utilization |
|---|---|---|---|
| Stock `mlx_lm` (async) | 33.8 | 29.5 | 74% |
| V5 monolithic (async) | 33.3 | 30.0 | 75% |
| Stock `mlx_lm` (sync) | 36.2 | 27.6 | 69% |
| V5 monolithic (sync) | 34.6 | 28.9 | 72% |

### Per-matmul bandwidth (via `mx.metal` timing)

| Matmul | Bandwidth |
|---|---|
| Fused input projection (DeltaNet) | 219 GB/s |
| Out projection | 228 GB/s |
| MLP gate+up (fused) | 234 GB/s |
| MLP down | 231 GB/s |
| QKV projection (attention) | 222 GB/s |

Individual matmuls hit 40–43% of peak — expected, each is a short kernel that can't
saturate the memory controller. **Pipelined** (consecutive matmuls dispatched without
sync), effective bandwidth reaches **~439 GB/s (80% of peak)** because the next kernel
starts before the previous one drains.

### Sync vs async, and CPU graph build

| Configuration | Graph build time |
|---|---|
| Stock `mlx_lm` | 1.71 ms |
| V5 monolithic compile | 0.64 ms (−63%) |

The GPU takes ~34 ms/token; the CPU takes ~1.7 ms (stock) or ~0.6 ms (V5) to build the
graph. GPU ≫ CPU, so the CPU is never the bottleneck. `async_eval` hides the CPU time
behind GPU execution — but there's almost nothing to hide. The 2.4 ms sync–async gap
(36.2 vs 33.8 ms) is exactly that overlapped CPU graph-build + Python dispatch time.

---

## Gap breakdown (the 8.7 ms over theoretical, two views)

### Coarse view (BANDWIDTH_ANALYSIS): 33.8 ms − 25.1 ms = 8.7 ms

| Source | Est. cost | Notes |
|---|---|---|
| Kernel dispatch overhead | 3–5 ms | ~930 traced ops/token (fewer real dispatches after `mx.compile`) |
| Non-matmul compute | 2–3 ms | RMS norms, activations, transposes, concat, argmax |
| Memory controller inefficiency | 1–2 ms | bank conflicts, TLB misses, non-sequential access |
| KV-cache reads | 0.5–1 ms | 16 attention layers, grows with context |
| Python overhead (sync only) | 1.5–2 ms | eliminated by `async_eval` / V5 compile |

### Fine view (DISPATCH_BARRIER_PROFILING, 2026-04-05)
Method: batched 200 reps per op in a single `mx.eval` (eliminates ~90 µs eval-barrier).

```
Theoretical bandwidth:     25.1 ms (13.7 GB at 546 GB/s)
Matmul chain (pipelined):  22.4 ms (weight reads only — some weights cached, below theoretical)
Matmul + norm chain:       31.0 ms (+8.6 ms from norm dispatch barriers)
Full model:                36.1 ms (+5.1 ms from GDN, activations, reshapes)
```

#### The 8.6 ms norm-barrier overhead
- Individual norm dispatch: ~5 µs in isolation.
- Pipelined between matmuls: ~33 µs per barrier (L2 coherency stall between norm output
  and matmul input).
- 256 norm+matmul pairs × 33 µs = **8.6 ms** (a 170× amplification over isolated cost).
- Fused `rms_norm_qmv` kernel saves **2.0 ms** (measured). Full metallib integration
  expected to save more. Details: [`kernel-fusion.md`](kernel-fusion.md).

#### The 5.1 ms non-weight, non-norm overhead

**DeltaNet layers (48) — 2.86 ms:**

| Op | Per-layer | ×48 | Notes |
|---|---|---|---|
| `fused_gdn_step` | 21.95 µs | 1.054 ms | already custom kernel; memory-bound on state `[1,28,128,64]`=229K elts |
| `fused_conv1d_silu` | 9.49 µs | 0.456 ms | could merge INTO gdn_step |
| `mlp_silu_gate` | 6.68 µs | 0.320 ms | 3 dispatches (slice+silu+mul) → fuse to 1 |
| `projection_slices` | 5.68 µs | 0.272 ms | extract q,z,b,a; "should be free" pointer ops |
| `splits_reshapes` | 5.54 µs | 0.266 ms | q/k/v from conv output; "should be free" |
| `silu×rms_norm` (gate) | 5.33 µs | 0.256 ms | 3 dispatches → fuse to 1 |
| `residual_add` (×2) | 2.11 µs | 0.203 ms | minimal |
| `reshape_out` | 0.70 µs | 0.033 ms | negligible |

**Attention layers (16) — 0.82 ms:**

| Op | Per-layer | ×16 | Notes |
|---|---|---|---|
| `kv_cache_update` | 10.83 µs | 0.173 ms | `put_along_axis` |
| `sdpa` | 7.68 µs | 0.123 ms | `mx.fast.scaled_dot_product_attention` |
| `mlp_silu_gate` | 6.43 µs | 0.103 ms | 3 dispatches → 1 |
| `rope_q_k` | 5.21 µs | 0.083 ms | `mx.fast.rope` |
| `mask_creation` | 5.16 µs | 0.083 ms | arange+compare, recomputed every step |
| `attn_proj_slices` | 5.18 µs | 0.083 ms | q/k/v/gate extraction |
| `residual_add` (×2) | 2.25 µs | 0.072 ms | minimal |
| `sigmoid×output` | 3.29 µs | 0.053 ms | gated attention output |
| `pre_transpose_qkv` | 1.82 µs | 0.029 ms | negligible |
| `transpose_reshape` | 1.30 µs | 0.021 ms | negligible |

**Summary of the 5.1 ms:**
```
Measured (isolated ops):  3.68 ms
Pipeline stall overhead:  1.42 ms (dependency chain across 64 layers)
Total:                    5.10 ms
```

The 1.42 ms is fundamental MLX dispatch scheduling: ~640 graph nodes (≈10/layer × 64) at
~2.2 µs/node. Cannot be eliminated without changing MLX's runtime or cutting node count.

**Key profiling insights:**
1. Slice/reshape ops cost 5–6 µs each despite being "free" pointer arithmetic — MLX
   schedules them as graph nodes with dispatch overhead (~0.54 ms total for DeltaNet
   slices).
2. `mx.compile` partially fuses elementwise chains but cannot fuse across kernel types
   (norm+matmul, activation+matmul). That barrier is the main target.
3. The GDN recurrence at 22 µs/layer is already efficient (state ≈ 458 KB read+write,
   ~0.9 µs theoretical at L2 bandwidth; the rest is compute + dispatch).
4. Mask creation (5 µs × 16 = 0.08 ms) is recomputed every step — could be cached for
   causal attention at known positions.

---

## Implications (why MTP, not kernels)

1. **Kernel optimization has diminishing returns.** At 80% pipelined bandwidth you're
   within 20% of the hardware limit. Getting to 90% needs MLX-runtime changes.
2. **Speculative decoding is the right lever.** Don't make each forward faster — get more
   tokens per forward. Even at 79% MTP acceptance you amortize the 34 ms over 1.79 tokens.
3. **The metadata tax (11.1%) is real but tolerable.** group_size=64 stays.
4. **Bandwidth is the fundamental constraint.** 13.7 GB / 546 GB/s = 25.1 ms minimum.
   No software beats this without reading less data (smaller model, harder quant, pruning).

If all profiled fusions land (~5.1 ms recoverable → forward 36.1 → ~31.0 ms), at 1.8
tokens/step that projects to **58.1 tok/s (1.97× baseline)**.

---

## The 20% bandwidth gap — why 80% not 100%

| Source | Est. loss |
|---|---|
| DRAM per-bank refresh cycles | 3–4% |
| Memory controller protocol (activate/precharge/CAS) | 4–5% |
| SLC contention (CPU/display/IO) | 2–4% |
| Scale/bias cache-line waste (128 B line, 2–4 B used) | 1–2% |
| Kernel dispatch overhead | ~1% |
| **Total** | **~11–16%** |

The rest is in-kernel instruction scheduling/occupancy. Industry LPDDR5X streaming peaks
~93%. **Verdict: this gap is hardware physics** — DRAM refresh, protocol timing, cache-line
granularity are not software-fixable. Apple's `qmv_fast` is already optimal (no
threadgroup memory, pure register SIMD, bank-parallel streaming).

---

## Research frontiers (MLX-hardware-specific)

> **General spec-decode math lives in Theory.** The platform-independent
> derivations — the parallel-MTP probability math (`P(both correct)=0.79²`…), the 22%
> break-even, the O(1)/token ceiling reframing (→ 213 tok/s @N=8), and batched-inference
> weight reuse — are the canonical content of
> [`../05-THEORY-AND-DESIGNS/research-frontiers.md`](../05-THEORY-AND-DESIGNS/research-frontiers.md) §A.
> This section keeps only the **M4-Max-specific** frontiers and the **M4-Max numeric
> instantiation** of those general results: ANE 2.24 ms, CPU SME offload, die topology /
> SLC, the 20% bandwidth-gap physics breakdown, the entropy-coding measurement on this
> exact Qwen3.5-27B-4bit model, the megakernel estimate, the projection table, and the M5
> outlook.

### Parallel MTP n=2 — M4-Max instantiation (highest-priority, MLX-feasible)
> General math (the `0.79²` probabilities, `E[tokens/step]=2.414`, the 22% break-even) is
> in Theory §A.1. The M4-Max numbers below are the instantiation those projections feed.

Two independent MTP drafts verified in one T=3 pass; on M4 Max a T=3 verify ≈ **45.9 ms**,
which at the theory-derived `E[tokens/step]=2.414` gives **52.6 tok/s (+23% over n=1)**.
n=3 parallel projects to **58.3 tok/s (2× baseline)**. Priority-1 frontier: the math is
proven (Theory §A.1) and the infrastructure already exists.

### MTP head on ANE — **CONFIRMED** (2026-04-05) — *MLX-hardware-specific, keep here*
The MTP head is pure attention + MLP (no DeltaNet) and the Neural Engine has its **own
memory bus** (zero GPU-bandwidth interference). CoreML conversion succeeded:
- 424.7M params, 849.5 MB mlpackage
- **2.24 ms median latency on ANE**, exact output match with MLX
- Runs entirely within the 34 ms GPU forward window

Earlier feasibility figures (pre-confirmation projection): MTP compute is **0.037 ms** at
the ANE's ~11 TOPS; the MTP weight load on ANE is **10–22 ms** (still fits within the
38 ms GPU step); bandwidth interference is **zero**. The enabling proof was an earlier
CoreML conversion of just embed+norm+lm_head at **1.46 ms** — the MTP head is the same op
types, so it should (and did) convert.

This enables a voting scheme: two ANE MTP predictions (~4 ms total) free while the GPU
runs the main forward; when both agree (~64% of the time), skip the 38 ms verification.
Files: `~/models/mtp-head-ane.mlpackage`, `~/optimizations/qwen-mtp-inference/ane-mtp/`
(author's M4 Max paths; external to this corpus). Priority-2 frontier (enables deeper speculation at zero cost).

### CPU SME offload (smaller win)
M4 replaced AMX with ARM SME (12 P-cores × ~2 TFLOPS ≈ 24 TFLOPS FP32). `mx.stream(mx.cpu)`
runs MTP on CPU while GPU does the main forward (one-line change) — but CPU shares the
546 GB/s bus, so 265 MB of MTP weights steals ~0.5 ms of GPU bandwidth. Net ~1.6 ms
(+7%). ANE is strictly better (zero interference).

### Die topology / SLC — no win (MLX-hardware-specific)
GPU is one contiguous block (~50% of die); 8 LPDDR5X channels striped across two die
edges; weights interleaved across all 8 (no locality optimization possible). SLC 48 MB
(distributed near the memory controllers) = 0.7% of the 13.7 GB model. Wire distance
DRAM→GPU is 3–20 mm, propagation ~0.2 ns (irrelevant vs 30–40 ns access). Depth-first
scheduling (process all tokens through one layer before the next) would maximize SLC reuse
but autoregressive generation prevents it (token N depends on token N−1 through all
layers). No NUMA on single-die M4 Max. **Only batched inference gives weight reuse** (the
general idea — applies to any bandwidth-bound decoder — is Theory §B.5); on M4 Max its
instantiation is B sequences × 1 weight read per layer, so **batch=4 → effective
25.1/4 = 6.3 ms per token.**

### Megakernel
One Metal dispatch for the whole forward (vs **~500+ kernel dispatches per token** today,
each with argument-buffer setup + GPU scheduler overhead) could save **10–15%** from
eliminating inter-kernel overhead — software-managed weight streaming with overlapped
load/compute. Requires reimplementing the whole forward in Metal outside MLX. The
estimated gain partly accounts for the **2.4 ms gap (33.9 − 31.4 ms)**. (Ref: Jia et al.,
"Compiling LLMs into a Megakernel.")

### Entropy coding — first framed as "the big prize," then **DEBUNKED** (2026-04-05)
Initially framed as the single highest-impact finding across all research teams: 4-bit
quantized weights nominally store 4 bits but were claimed to hold only **~1.1–1.5 bits**
of Shannon entropy (per ECQ/EntroLLM), so per-row rANS (asymmetric numeral system) coding
could shrink weights ~3.3× losslessly → **9.3 ms BW floor → 80–90 tok/s, 3× baseline**.
The enabling kernel would be a **fused rANS-decode + GEMV** that decompresses weights
directly into registers during the dot product (per-row encoding = O(rows) parallelism),
never materializing the full weight matrix. The pre-debunk projection cited an **ECQ
prototype measured at 3.27× throughput (42.9 → 140 tok/s) on an M3 Pro**.

Pre-debunk combined-stack projection (with MTP n=2 at 2.41 tok/step):

| Configuration | Effective BPW | Weight data | BW floor | With MTP n=2 |
|---|---|---|---|---|
| Current 4-bit + MTP n=1 | 4.0 | 13.7 GB | 25.1 ms | 42.7 tok/s (n=1) |
| + Parallel MTP n=2 | 4.0 | 13.7 GB | 25.1 ms | 52.6 tok/s |
| + Mixed 3/4-bit | 3.5 | 12.0 GB | 22.0 ms | 57.4 tok/s |
| + Entropy-coded 4-bit | ~1.5 | 5.1 GB | 9.3 ms | 82.3 tok/s |
| + Entropy + n=3 parallel | ~1.5 | 5.1 GB | 9.3 ms | 87.4 tok/s |
| + Entropy + mixed 3/4 | ~1.2 | 4.3 GB | 7.9 ms | ~95 tok/s |

**Then empirically measured on 1.4 billion 4-bit values from Qwen3.5-27B-4bit:**
```
Shannon entropy:    3.72 bits (NOT 1.1)
Compression ratio:  1.08× (NOT 3.3×)
Savings:            7% (13.7 GB → 12.7 GB)
```
Standard affine 4-bit quant (scale·int+bias) produces a near-uniform bell curve centered
on 8 (mid-range 0–15). The 1.1-bit entropy was measured on a *different* format with
learned codebooks that deliberately cluster values. rANS on standard 4-bit saves only 7%
— **not worth a custom Metal kernel.** Real entropy gains would need a compression-aware
quantizer (ECQ/EntQuant) — a different project. So the Mixed-3/4-bit and entropy rows of
the table above are **superseded** for this model. Mixed 3/4-bit requantization (→3.5 BPW,
12.0 GB, 22.0 ms floor) remains immediate and MLX-supported today; entropy coding does not
pay off here.

> The entropy-coding *measurement* (3.72 bits / 1.08× on this exact Qwen3.5-27B-4bit
> model) and the projection table above are the **M4-Max numeric instantiation** and stay
> here. The general bandwidth-floor derivation (BPW → GB → ms → tok/s) is platform-general
> theory — canonical in
> [`../05-THEORY-AND-DESIGNS/research-frontiers.md`](../05-THEORY-AND-DESIGNS/research-frontiers.md) §A/§C.
> References (entropy-coding literature): drxddy/ecq, MLX #3043, EntroLLM
> (arXiv:2505.02380), DFloat11 (arXiv:2504.11651), Float8@2bits.

### Other confirmed dead ends (matmul team)
SLC-aware tiling for M=1 (weights read once, no reuse), Morton/Z-order layout (breaks
DRAM bursts), FP16 accumulation in qmv (0% — memory-bound), extra SIMD shuffles (already
optimal), CPU SLC pre-warming (pseudo-random eviction). DeltaLLM weight sharing: 10–15%
accuracy drop for 12–25% compression — bad trade.

### Priority order (from `RESEARCH_FRONTIERS.md`; general math → Theory §A/§C)
1. **Parallel MTP n=2** — proven math, 22% break-even (Theory §A.1), infrastructure exists.
   M4-Max instantiation → 52.6 tok/s.
2. **MTP on ANE** — zero-cost drafting (2.24 ms), enables deeper speculation.
3. **Entropy-coded weights** — *was* ranked the transformative change, since debunked for
   this model (7% only); needs a compression-aware quantizer to matter.
4. **Mixed 3/4-bit requantization** — immediate, MLX-supported today (→57.4 tok/s w/ n=2).
5. **Batched inference** — multiplicative with everything above (general theory, Theory §B.5).

### M5 outlook
M5 base shipping ~April 2026; Pro/Max ~Q4 2026. Adds GPU Neural Accelerators (40
matrix-mul units, Metal 4 TensorOps), ~600–650 GB/s (LPDDR5X-9600). Does **not** change
the story — the bandwidth wall just shifts 10–15%.

---

## Related

- [`kernel-fusion.md`](kernel-fusion.md) — the `fused_rms_norm_qmv` kernel that attacks the 8.6 ms norm barrier; the actionable-fusions table.
- [`mtp-self-speculative-mlx.md`](mtp-self-speculative-mlx.md) — why speculative decoding (not faster kernels) is the lever; the O(1) ceiling.
- [`the-journey.md`](the-journey.md) — the dead ends (qmv_fast tuning, group_size=128, CoreML/ANE) that this profiling explains.
- [`README.md`](README.md) — the 03-MLX entry page + results table.
- Cross-domain: [`../05-THEORY-AND-DESIGNS/research-frontiers.md`](../05-THEORY-AND-DESIGNS/research-frontiers.md) (canonical for the general spec-decode / O(1)-ceiling math; this doc holds the M4-Max instantiation), [`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md).

---

*Sources merged here: `qwen-inference-lab/docs/BANDWIDTH_ANALYSIS.md` (which itself
appends a Dispatch-Barrier section), `qwen-inference-lab/docs/DISPATCH_BARRIER_PROFILING.md`,
`qwen-inference-lab/docs/RESEARCH_FRONTIERS.md`. The `qwen-ops/research/findings/` copies
of all three are byte-identical and collapsed.*
