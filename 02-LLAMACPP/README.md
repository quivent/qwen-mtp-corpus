# 02 — llama.cpp track: MTP speculative decoding for Qwen3.5-27B

> End-to-end port of the Qwen3.5 Multi-Token Prediction (MTP) head into
> [llama.cpp](https://github.com/ggerganov/llama.cpp): HuggingFace → GGUF →
> loader → graph builder → speculative-decoding loop, on a **hybrid attention +
> DeltaNet** architecture. The journey from "the model won't load" → correct
> spec output → **1.99× over K=1 vanilla**.

This folder owns the llama.cpp implementation domain. Map:

| Doc | What it covers |
|---|---|
| [`infrastructure-patches.md`](infrastructure-patches.md) | The 16 ordered infra patches (5 base + 11 series). What each does, why. Patch 11 = the unblock. |
| [`optimization-variants.md`](optimization-variants.md) | The 9 optimization variants. Status table: coherent? speedup? winner / broken / negative. |
| [`the-bug.md`](the-bug.md) | The cache-bookkeeping bug story. `id_last = corr` double-write. Why 6 agents missed it. 3 lessons. |
| [`the-recipe.md`](the-recipe.md) | **Canonical home** of the 1.99× adaptive-chained-MTP recipe. Env vars, 5-prompt table, cost breakdown, remaining levers. (MLX track links here.) |
| [`benchmarks.md`](benchmarks.md) | **Server-mode** numbers from the two raw `llama-server` log captures: MTP+0.8B-draft vs no-draft baseline. Confirms server-mode spec is throughput-neutral (~20 tok/s). |
| [`integration-plan.md`](integration-plan.md) | The 6-phase per-position-heads integration plan with spend gates. |
| [`patches/`](patches/) | Copied patch series: `infrastructure/` (16) + `optimizations/` (9). |
| [`source/`](source/) | Copied MTP-specific fork source files (6). See `source/README.md`. |
| [`logs/`](logs/) | Raw `llama-server` log captures (copied verbatim): `llama_cpp_server.log` (MTP+draft), `llama_cpp_server_nomtp.log` (baseline). See `logs/README.md`. |

---

## The journey, in one narrative

**Qwen3.5-27B is a hybrid model.** It interleaves **48 DeltaNet** (linear-attention,
recurrent) layers with **16 full-attention** layers, plus **one MTP head as layer 64**.
That hybrid shape is what makes everything hard: speculative decoding's standard
trick — cheaply roll back a wrong-path decode — assumes you can recover the model
state at any intermediate position. DeltaNet is **irreversible**: there is no
"state at position k", only the state after processing all tokens in a batch.

### Stage 1 — "the model won't load"

llama.cpp had never been exercised against a real converted Qwen3.5 GGUF with an
MTP layer. Every layer of the stack was broken in its own way:

- the converter **silently strips** MTP tensors (block_count never bumped past 64);
- the loader doesn't read `nextn_predict_layers`;
- the tensor classifier files `NEXTN_*` tensors as global-output instead of
  per-layer-repeating, so they fail the layer-index sanity check;
- the `QWEN35` graph builder has **no draft path** at all;
- `post_attention_layernorm` is emitted as `blk.N.ffn_norm.weight` but the loader
  looked for `attn_post_norm` (a latent upstream bug, never hit before);
- the recurrent memory module has **no snapshot/restore** primitive.

Patches **0000–04** fix each of these. By patch 03 the MTP head runs end-to-end and
produces *plausible* drafts; the model loads and the draft graph dispatches.

### Stage 2 — "the spec path produces garbage"

With drafting working, the **verify/rollback loop** produced degraded text — digit
loops like "1. 2. 3. ... 1000000" within ~10 tokens of any rejection. The team
spent most of the session chasing **numerical** suspects: chunking-vs-AR DeltaNet
divergence in fp16 (patch 09), RoPE positions, a recurrent state leak (patch 06's
private scheduler), hidden-state pipe corruption, the `kq_mask` scheduler bug
(patch 04). All real, all red herrings.

The actual culprit was a **one-line host-side cache-bookkeeping bug** in
`mtp-speculative.cpp`: after a batched rollback re-decode of `[id_last, drafts…, corr]`,
the code set `id_last = corr` — but `corr` was already the last slot written to the
cache. The next verify wrote `corr` a *second* time, shifting every subsequent
position by one slot. The model saw garbage context. See [`the-bug.md`](the-bug.md).

**Patch 11** is the unblock: read the tail logits of the re-decoded batch and use
`argmax` as the new `id_last`, mirroring the accept-all branch. One line. After it,
K=1 spec output matches plain decode structurally on all benchmark prompts.

### Stage 3 — "the recipe was there the whole time"

The cache bug had been **masking every measurement** for the entire session. The
chained-recurrent threading (`next_prev_hidden = scratch_out_hidden.data()`) and
the confidence gate were both committed *early* — but every measurement of those
env vars produced "speedups" on broken text, so they were dismissed as ineffective.

After patch 11, re-running the previously-dismissed `MTP_CHAIN_KMAX=2
MTP_CHAIN_THRESH=0.85` combo delivered **1.99× over K=1 vanilla** with coherent
output. The session's biggest win was a **re-validation**, not a new feature. See
[`the-recipe.md`](the-recipe.md).

---

## Honest performance numbers

Qwen3.5-27B **Q4_K_M**, **M4 Max** (16-core GPU, 128 GB unified, 546 GB/s),
post-bug-fix, n=64 tokens/prompt. **All numbers verified against plain-decode
output coherence** (the hard lesson of the bug):

| Path | tok/s | vs plain | vs K=1 vanilla | Output |
|---|---|---|---|---|
| **Plain decode** (`llama-bench` tg32) | **17.90** | 1.00× | — | ✓ ground truth |
| K=1 MTP spec (single head) | **7.64** | **0.43×** | 1.00× | ✓ correct |
| **K=2 adaptive chained MTP** (the recipe) | **13.98** | **0.78×** | **1.99×** | ✓ coherent |

The headline arc:

- **Single-head MTP spec is *slower* than plain decode** (0.43×). On this hybrid
  model the MTP draft pass costs roughly as much as a main forward pass, so K=1
  spec cannot win — the fixed per-pass overhead (graph alloc, snapshot/restore, KV
  bookkeeping) dominates, not the head's own FLOPs.

  **Server-mode evidence** (raw `llama-server` logs, [`benchmarks.md`](benchmarks.md)):
  the MTP head accepts only **~2% (2/98)** of its drafts; even adding a **Qwen3.5-0.8B-Q8_0
  companion draft model** (which itself accepts ~49%, 47/95) leaves decode at **18.5–21.8
  tok/s** — identical to the no-draft baseline's **20.81 tok/s**. Server-mode spec is
  throughput-neutral; the companion draft's ~264–417 ms forward cost cancels its accepted
  tokens. The CLI win comes *only* from confidence-gated chaining, not from naive drafting.
- **Chained MTP + confidence gating** triples K=1 to **1.99×**, the closest any
  Qwen3.5-27B speculative path has come in llama.cpp — but still **0.78× of plain
  decode**, i.e. *not yet over the line*. The remaining gap is per-forward-pass
  overhead in llama.cpp, **not algorithm** (MLX runs the same recipe at 1.73× of
  *its* baseline because it uses cache slicing instead of snapshot/restore).

### The hybrid-model spec-decode challenge (why this is hard)

1. **DeltaNet is irreversible.** No intermediate-position state → rollback needs
   snapshot+restore of the recurrent half *and* an attn-only KV trim, then an AR
   re-decode to make recurrent numerics match plain decode.
2. **Chunking vs AR DeltaNet kernels diverge in fp16.** T≥2 verify batches route
   through `build_delta_net_chunking`; T=1 through `build_delta_net_autoregressive`.
   Mathematically equivalent, numerically a few ulps apart. Hypothesized as the
   garbage-output cause; proven a red herring (patch 09). Bounded by `n_commit`.
3. **Cross-stream `seq_cp` is alias-only on recurrent memory.** Tree/multi-seq
   variants can't get real per-branch recurrent state without a deep memory rewrite;
   they fall back to per-branch `id_last` duplication (`kv_unified=true` + bumped
   `n_parallel`).
4. **Single MTP head ≠ multi-step drafter.** Qwen3.5 ships *one* MTP head trained
   to predict +1 from the main hidden. Chaining it through its own output to draft
   +2, +3 is out-of-distribution: +1 ≈ 80% accurate, +2 ≈ 60%, +3 ≈ 40%. That's why
   K=2 (not K=3) is the local optimum, and why DeepSeek-V3-style per-position heads
   are the alternative path ([`integration-plan.md`](integration-plan.md)).

---

## How to apply + reproduce

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout 9c7911f4f          # parent commit recorded in patches/infrastructure/00-base.txt
git am path/to/patches/infrastructure/*.patch     # 0000..04 base, then 01..11
git am path/to/patches/optimizations/01-feat-mtp-adaptive-chain*.patch   # the winner
cmake -B build && cmake --build build -j 12 --target llama-mtp-speculative

MODEL=path/to/qwen3.5-27b-q4km.gguf
# Plain decode ground truth
./build/bin/llama-bench -m $MODEL -p 0 -n 32 -ngl 99
# THE RECIPE — chained MTP with confidence gating (1.99x over K=1)
MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85 \
  ./build/bin/llama-mtp-speculative -m $MODEL \
  -p "Explain photosynthesis." -n 64 -ngl 99 -c 2048
```

Output must be byte-coherent with plain decode. If it isn't, you don't have
patch 11 applied.

---
*Provenance: distilled from `qwen-mtp-llamacpp/README.md`,
`qwen-mtp-optimizations/README.md`, `qwen-mtp-research/README.md`, and the patch
commit messages. Hardware/arch facts cross-checked against `MTP/_meta/CONVENTIONS.md`.*
