# 03 — MLX / Apple Silicon track

Optimizing **Qwen3.5-27B-4bit** inference on a single **Apple M4 Max** (16-core GPU,
128 GB unified memory, 546 GB/s bandwidth), starting from stock `mlx_lm` and pushing to
the bandwidth wall.

**Headline: 29.5 tok/s stock → 51.1 tok/s (1.73×).**

This is the first working **MTP (Multi-Token Prediction) inference** for Qwen3.5 in
Python. Every other framework — MLX's own converter, HuggingFace transformers, vLLM —
silently strips the MTP head weights on load. We reverse-engineered the architecture
from the raw HF weight shapes and built a self-speculative decoder around it.

---

## Reader's map

| If you want… | Read |
|---|---|
| The whole journey, every approach in order + the dead ends | [`the-journey.md`](the-journey.md) |
| The Metal kernels (fused conv/GDN/T=2, rms_norm+qmv) and why custom kernels often *lost* | [`kernel-fusion.md`](kernel-fusion.md) |
| Where the time goes: 546 GB/s, 13.7 GB, 25.1 ms floor, dispatch barriers | [`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md) |
| The MTP self-speculative decode loop, 79% acceptance, split-recurrence, ceiling | [`mtp-self-speculative-mlx.md`](mtp-self-speculative-mlx.md) |
| The uncensored Huihui variant (conversion, 3% cost, ThreadLocalStream fix) | [`huihui-abliterated.md`](huihui-abliterated.md) |
| The code | [`src/`](src/) (+ [`src/README.md`](src/README.md) for file origins) |

Cross-domain:
- The general **split-recurrence rollback** *technique* is owned by Theory:
  [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md).
  This track covers the MLX-specific implementation only.
- The Qwen3.5 architecture + MTP head tensor details are owned by Architecture:
  [`../01-ARCHITECTURE/`](../01-ARCHITECTURE/).
- The llama.cpp 1.99× recipe is owned by llama.cpp:
  [`../02-LLAMACPP/the-recipe.md`](../02-LLAMACPP/the-recipe.md).

---

## Results (exact)

The canonical results table. Numbers are measured, not rounded.

| Configuration | tok/s | vs. baseline | Source |
|---|---|---|---|
| Stock `mlx_lm` (async eval, greedy) | 29.5 | 1.00× | baseline |
| V5 monolithic `mx.compile` (all 64 layers) | 30.0 | 1.02× | +1.7% async (+4.7% sync); costs +16 GB |
| Stock + spec decode (0.8B draft model) | 37.6 | 1.27× | requires separate draft model |
| MTP head (self-speculative) | 36.9 | 1.25× | checkpoint/restore rollback |
| MTP + split-recurrence rollback | 42.7 | 1.45× | zero-cost rollback |
| Adaptive MTP chain — Huihui abliterated (2026-04-08) | 49.5 | 1.68× | threshold=0.8, max_chain=2 |
| **Adaptive MTP chain — vanilla (revalidated 2026-04-08)** | **51.1** | **1.73×** | threshold=0.8, max_chain=2 |

Historical memory-anchor for the vanilla adaptive chain was **52.0 tok/s** (MLX 0.30.6);
revalidation on current MLX gives 51.1 (≈2% version drift, noise-level).

> **Note on the two "best" figures.** `mlx-qwen-mtp/README.md` reports a separate
> projection of **~45 tok/s (~1.52×)** for "MTP + fused rms_norm into matmul" — that is
> the *shipped repo's* stack (split-recurrence + the partial rms_norm fusion). The
> **51.1 tok/s** best comes from the **adaptive MTP confidence chain + batch verify**
> driver in the sibling `parallel-mtp-voting` project (only its logs are in this corpus).
> Both are real; 51.1 is the corpus headline.

### Per-matmul / detail numbers (M4 Max, Qwen3.5-27B-4bit)

| Metric | Value |
|---|---|
| Peak memory bandwidth | 546 GB/s |
| Total weight reads per token | 13.7 GB (12.2 GB weights + 1.52 GB scales/biases @ group_size=64) |
| Theoretical minimum latency | 25.1 ms/tok |
| Theoretical maximum throughput | 39.8 tok/s (100% BW) |
| Stock measured | 33.8 ms/tok = 29.5 tok/s = 74% BW utilization |
| Pipelined matmul bandwidth | ~439 GB/s (80% of peak) |
| MTP acceptance rate (single head, n=1) | 79% |

---

## The 4-phase journey (one-paragraph each)

**Phase 1 — Kernel fusion (V2–V6). Net gain: +1.7%.** Weeks fusing DeltaNet
projections (4 matmuls → 1), writing custom Metal kernels (`fused_conv1d_silu`,
`fused_gdn_step`), compiling whole layer stacks with `mx.compile`. The GPU was already
the bottleneck and `mx.compile` was already fusing elementwise ops. Custom fusion
kernels (V6: `fused_add_rms_norm`, `silu_mul_rms_norm`, `dual_rms_norm`) *broke* the
compile graph and made things **slower** (−1.5%). V5 monolithic compile got +1.7%
(async) at the cost of 16 GB extra memory. **Lesson: don't hand-fuse what `mx.compile`
already fuses; profile real GPU dispatches, not traced graph node counts.** →
[`kernel-fusion.md`](kernel-fusion.md)

**Phase 2 — Speculative decoding. 29.5 → 37.6 tok/s (1.27×).** A broken benchmark
(`stream_generate`, with per-token Python overhead) first showed 26.5 tok/s and we
nearly declared spec-decode dead for DeltaNet. Re-measured correctly with
`mlx_lm.generate` and got 37.6 with a 0.8B draft model. **Lesson: measure correctly
before declaring something dead.** → [`the-journey.md`](the-journey.md)

**Phase 3 — MTP self-speculative. 36.9 tok/s.** Discovered Qwen3.5 ships with MTP
weights that MLX and transformers strip on convert. Reverse-engineered a single
gated-attention transformer-layer head from weight shapes. The model drafts its own next
token (~3 ms overhead vs. ~34 ms forward pass), **79% acceptance, no draft model
needed**. The critical bug was concat order (`[embed, hidden]`, not `[hidden, embed]` —
canonical: [`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md)).
→ [`mtp-self-speculative-mlx.md`](mtp-self-speculative-mlx.md)

**Phase 4 — Split-recurrence rollback. 36.9 → 42.7 tok/s (1.45×); then adaptive chain →
51.1 tok/s (1.73×).** DeltaNet layers carry recurrent state that must roll back on draft
rejection. Naive checkpoint/restore-and-redo cost ~7 ms/step (34 ms × 21% reject). Two
insights made rollback **zero-cost**: (1) MLX arrays are immutable, so a "checkpoint" is
just saving Python references; (2) split the GDN recurrence into per-token calls while
keeping matmuls batched — the rejected path simply never runs the second recurrence
step. Stacking an *adaptive confidence chain* (recurrently re-applying the single head
K times, gated by `max(softmax) ≥ threshold`) + batch verify reaches 51.1. →
[`mtp-self-speculative-mlx.md`](mtp-self-speculative-mlx.md),
[`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md)

---

## Where the time goes (the central finding)

Decode is **memory-bandwidth bound**, not compute bound. Every token reads all 13.7 GB
of weights once; at 546 GB/s that is a hard **25.1 ms floor**. The consequence: **you
cannot make a forward pass meaningfully faster** without reading less data — the only
lever is getting **more tokens per weight read** (speculative decoding). That is why
Phases 2–4 dominate the gains and Phase 1 (kernel fusion) does not.

```
T=1 baseline:         34 ms → 1 token   → 29.5 tok/s
T=2 split-recurrence: 38 ms → 1.8 tokens → 42.7 tok/s (1.45×)
  (38 ms = 34 weight reads + 3 MTP head + 1 split-recurrence overhead)
```

The full M4-Max numbers — the 13.7 GB weight budget, the 25.1 ms floor derivation, the
8.6 ms norm-barrier discovery, and the 20%-bandwidth-gap physics — are canonical in
[`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md). The general O(1)-per-token
ceiling framing (spec decode is *not* capped at 2×) is in
[`../05-THEORY-AND-DESIGNS/research-frontiers.md`](../05-THEORY-AND-DESIGNS/research-frontiers.md).

---

## Quick start (shipped 42.7 tok/s path)

```python
import mlx_lm
from src import patch_model, mtp_generate, load_mtp

# 1. Extract MTP weights (one-time) — they're in the HF checkpoint but stripped on load
from src.extract_weights import extract_mtp_weights
extract_mtp_weights(output_path="src/mtp_weights.safetensors")  # ~265 MB

# 2. Load base model, patch DeltaNet with fused kernels, load MTP head
model, tokenizer = mlx_lm.load("mlx-community/Qwen3.5-27B-4bit")
patch_model(model)                       # fuses 4→1 DeltaNet projections + Metal kernels
mtp_head = load_mtp(model, weights_path="src/mtp_weights.safetensors")

# 3. Generate with split-recurrence speculative decoding
print(mtp_generate(model, tokenizer,
                   prompt="Explain quantum computing in simple terms.",
                   max_tokens=256, mtp_head=mtp_head))
```

Requirements: Python ≥ 3.10, `mlx >= 0.30`, `mlx-lm >= 0.20`, Apple Silicon (M1+). License Apache-2.0.

---

## Related

- [`the-journey.md`](the-journey.md) — every approach in order, including the dead ends.
- [`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md) — the 25.1 ms floor, dispatch barriers, research frontiers (canonical M4-Max numbers).
- [`mtp-self-speculative-mlx.md`](mtp-self-speculative-mlx.md) — the MTP head, generation loop, 79% acceptance, the O(1) ceiling.
- [`kernel-fusion.md`](kernel-fusion.md) — the Metal kernels that survived vs. lost.
- [`huihui-abliterated.md`](huihui-abliterated.md) — the uncensored variant (~3% cost) + the `ThreadLocalStream` server fix.
- Cross-domain: [`../01-ARCHITECTURE/`](../01-ARCHITECTURE/) (Qwen3.5 arch + MTP head), [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) (the general technique), [`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md) (all-platform results).

---

*Sources distilled into this README: `mlx-qwen-mtp/README.md`, `qwen-inference-lab/README.md`,
`qwen-inference-lab/docs/TIMELINE.md`, `BANDWIDTH_ANALYSIS.md`, `HUIHUI_ABLITERATED.md`,
`logs/adaptive_mtp_{vanilla,huihui}.log`. See [`../99-LEDGER/inventory-03-mlx.md`](../99-LEDGER/inventory-03-mlx.md).*
