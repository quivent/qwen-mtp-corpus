# Results — the single authoritative benchmark table

> **The one place to find "which number is real."** Every number on this page is
> copied **exactly** from a source doc and cited in the right-hand column — no
> rounding, no averaging, no invented values. Where two docs report different
> values for what looks like the same thing, **both are shown** and the
> disambiguation is in [§4](#4-numbers-that-look-like-contradictions-but-arent),
> pointing to the open-questions registry (`../99-LEDGER/open-questions.md`).
>
> Layered for two readers: the **headline table** (§1) is the best result per
> platform; the **per-platform detail tables** (§2) list every configuration
> measured; §3 is the hardware reference; §4 disambiguates the famous
> apparent-contradictions.

---

## 1. Headline — best result per platform

> **Platform roles.** Production/highest-throughput work is the **datacenter-GPU (vLLM)**
> rows; the **M4 Max rows are a reference measurement platform**, not a target — see
> [`platform-scope.md`](platform-scope.md). Numbers are anchored to the box each was
> measured on and are not comparable across rows.

| Platform | Role | Hardware | Model + quant | Best tok/s | Speedup | Method | Source |
|---|---|---|---|---:|---|---|---|
| **vLLM** | production | GH200 480GB | Qwen3.5-27B W4A16 | **186** (batch=1) / **1030** (batch=8) | **5.54× batch** (+454% aggregate); 186 is baseline | Stock MTP spec=7 | `04-VLLM-GPU/README.md` (`vllm-patches-benchmarks.md`) |
| **vLLM** | single-box | RTX 5090 (32 GB) | Qwen3.5-27B GPTQ W4A16 (Huihui abliterated) | **151** (single 256 tok) | — (51% MTP accept; 347 batch=4) | GPTQ W4A16 + MTP=5 | `04-VLLM-GPU/README.md`, `04-VLLM-GPU/quantization.md` |
| **llama.cpp** | portable | reference: M4 Max (546 GB/s) | Qwen3.5-27B Q4_K_M | **13.98** | **1.99× over K=1 vanilla** (= 0.78× of plain decode 17.90) | Chained recurrent MTP + confidence gating (`MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`) | `02-LLAMACPP/the-recipe.md`, `02-LLAMACPP/README.md` |
| **MLX** | reference | M4 Max (546 GB/s) | Qwen3.5-27B-4bit (13.7 GB) | **51.1** | **1.73× over stock 29.5** | Adaptive MTP confidence chain + batch verify (threshold=0.8, max_chain=2) | `03-MLX/README.md`, `03-MLX/the-journey.md` |

**Reading the cross-platform speedups.** Each platform's "speedup" is **relative
to that platform's own baseline** — they are not comparable across rows. The
llama.cpp 1.99× is over its broken-then-fixed K=1 spec path (and is still
*below* plain decode at 0.78×); the MLX 1.73× is over stock `mlx_lm`; the vLLM
5.54× is batch-vs-single concurrency scaling, not a single-request speedup.
See §4 for why these baselines differ.

---

## 2. Per-platform detail — every configuration measured

### 2.1 llama.cpp (M4 Max, Qwen3.5-27B Q4_K_M, post-bug-fix, n=64 tok/prompt)

**Honest performance table** (all verified against plain-decode output coherence):

| Path | tok/s | vs plain | vs K=1 vanilla | Output | Source |
|---|---:|---|---|---|---|
| Plain decode (`llama-bench` tg32) | **17.90** | 1.00× | — | ✓ ground truth | `02-LLAMACPP/README.md` |
| K=1 MTP spec (single head) | **7.64** | 0.43× | 1.00× | ✓ correct | `02-LLAMACPP/README.md` |
| K=2 adaptive chained MTP (the recipe) | **13.98** | 0.78× | **1.99×** | ✓ coherent | `02-LLAMACPP/README.md`, `the-recipe.md` |

**5-prompt benchmark breakdown** (Q4_K_M, M4 Max, n=64; the mean is the 13.98 headline):

| Prompt | K=1 vanilla | K=2 thresh=0.85 | Speedup | Source |
|---|---:|---:|---|---|
| Write a haiku about spring. | 4.6 | 13.0 | 2.83× | `02-LLAMACPP/the-recipe.md` |
| Explain photosynthesis in one paragraph. | 7.1 | 14.7 | 2.07× | `02-LLAMACPP/the-recipe.md` |
| Write a Python function to compute Fibonacci. | 6.6 | 14.0 | 2.12× | `02-LLAMACPP/the-recipe.md` |
| List the planets of the solar system. | 8.3 | 13.8 | 1.66× | `02-LLAMACPP/the-recipe.md` |
| Translate hello world to French. | 8.5 | 14.4 | 1.69× | `02-LLAMACPP/the-recipe.md` |
| **Mean** | **7.02** | **13.98** | **1.99×** | `02-LLAMACPP/the-recipe.md` |

> Note: the recipe's run table also lists a K=1 mean of **7.02** while the
> headline performance table lists the single-head K=1 figure as **7.64**. Both
> are reported verbatim; 7.02 is the 5-prompt-mean baseline for the 1.99×, 7.64
> is the K=1 figure in the "Honest performance numbers" table.

**K-sweep — why K=2, not K=3** (single-prompt sweet-spot figures):

| K | thresh | tok/s | Note | Source |
|---|---|---:|---|---|
| 2 | 0.85 | **15.0** | Sweet spot (single-prompt) | `02-LLAMACPP/the-recipe.md` |
| 3 | 0.85 | 12.8 | 3rd step marginal accept doesn't pay back | `02-LLAMACPP/the-recipe.md` |
| 3 | 0.90 | 13.2 | Tighter threshold helps but K=2 still wins | `02-LLAMACPP/the-recipe.md` |

Per-position MTP-head accuracy on this single-head chain: **+1 ≈ 80%, +2 ≈ 60%, +3 ≈ 40%** (`02-LLAMACPP/the-recipe.md`, `README.md`).
Theoretical-ceiling cost breakdown projects **15.2 tok/s** (matches observed
14.8 on a quiet GPU) (`02-LLAMACPP/the-recipe.md`).

**Server-mode (`llama-server`) — neutral, distinct from the CLI recipe** (raw log captures):

| Run | Draft model | Decode tok/s (eval) | prompt eval tok/s | MTP-head accept | Companion accept | Source |
|---|---|---|---:|---|---|---|
| No-MTP baseline (Log B) | none | **20.81** (300 tok) | 79.74 | n/a (0 drafts) | n/a | `02-LLAMACPP/benchmarks.md` |
| MTP + 0.8B draft (Log A), per-req range | Qwen3.5-0.8B-Q8_0 | **18.5–21.8** | 95–98 | ~2% (2/98) | ~49% (47/95) | `02-LLAMACPP/benchmarks.md` |

Server-mode per-request eval throughput (Log A, the 6 requests): **20.71 / 20.75
/ 20.79 / 18.54 / 21.75 / 21.70 tok/s** (`02-LLAMACPP/benchmarks.md`).
Conclusion: server-mode spec is throughput-neutral (**~20 tok/s**); the CLI win
comes *only* from the confidence-gated chain, which these server logs were not
run with.

Per-pass theoretical levers (projections, not measured): graph caching →
~16.5 tok/s; MTP fused into main decode → **19.8 tok/s** (crosses plain);
per-position trained heads → ceiling ~2.23× of plain (`02-LLAMACPP/the-recipe.md`).

### 2.2 MLX (M4 Max, Qwen3.5-27B-4bit, 13.7 GB weights)

**The full journey ladder** (the canonical MLX results table — measured, not rounded):

| Configuration | tok/s | vs baseline | Note | Source |
|---|---:|---|---|---|
| Stock `mlx_lm` (async eval, greedy) | **29.5** | 1.00× | baseline | `03-MLX/README.md`, `the-journey.md` |
| V5 monolithic `mx.compile` (all 64 layers) | **30.0** | 1.02× | +1.7% async (+4.7% sync); costs +16 GB | `03-MLX/README.md`, `the-journey.md` |
| Stock + spec decode (0.8B draft model) | **37.6** | 1.27× | requires separate draft model | `03-MLX/README.md`, `the-journey.md` |
| MTP head (self-speculative) | **36.9** | 1.25× | checkpoint/restore rollback | `03-MLX/README.md`, `the-journey.md` |
| MTP + split-recurrence rollback | **42.7** | 1.45× | zero-cost rollback; the shipped win | `03-MLX/README.md`, `the-journey.md` |
| Adaptive MTP chain — Huihui abliterated (2026-04-08) | **49.5** | 1.68× | threshold=0.8, max_chain=2 | `03-MLX/README.md`, `huihui-abliterated.md` |
| **Adaptive MTP chain — vanilla (revalidated 2026-04-08)** | **51.1** | **1.73×** | threshold=0.8, max_chain=2 | `03-MLX/README.md`, `the-journey.md` |

> The ladder the task tracks (29.5 → 30.0 → 37.6 → 36.9 → 42.7 → 49.5 → 51.1) is
> exactly the rows above, in order. The 49.5 (Huihui) and 51.1 (vanilla) are the
> *same revalidation run pair*; 51.1 is the corpus headline.

**Adaptive-chain run statistics** (revalidated 2026-04-08, threshold=0.8, max_chain=2, max_tokens=128, mean of 3):

| Model | tok/s | Tokens/step | Avg chain | Rollbacks | Source |
|---|---:|---:|---:|---|---|
| Memory anchor (vanilla, MLX 0.30.6, historical) | 52.0 | 2.51 | — | 0 | `03-MLX/huihui-abliterated.md` |
| **Vanilla (current MLX)** | **51.1** | 2.29 | 1.27 | 0 / 56 | `03-MLX/huihui-abliterated.md`, `the-journey.md` |
| **Huihui abliterated (current MLX)** | **49.5** | 2.17 | 1.17 | 1 / 59 | `03-MLX/huihui-abliterated.md` |

Per-run (rock steady): vanilla 50.7 / 51.1 / 51.2 / 51.1; Huihui 49.4 / 49.6 /
49.7 / 49.5 (`03-MLX/huihui-abliterated.md`).

**MLX dead ends** (preserved — they prove the GPU is bandwidth-bound):

| Approach | tok/s | Change | Verdict | Source |
|---|---:|---|---|---|
| V6 custom Metal fusion kernels | 28.9 (28.92) | −1.5% | dead end — broke `mx.compile` fusion | `03-MLX/the-journey.md` |
| qmv_fast `results_per_simdgroup=8` | 28.3 | −4.2% | dead end — register pressure | `03-MLX/the-journey.md` |
| qmv_fast `num_simdgroups=4` | — | −1.5% | dead end — register pressure | `03-MLX/the-journey.md` |
| group_size=128 quantization | 30.3 | +2.8% | dead end — 2.3× quant error | `03-MLX/the-journey.md` |
| GPU-resident loop (V7) | 29.5 | +0% | dead end — `mx.compile` ≠ kernel fuser | `03-MLX/the-journey.md` |
| CPU draft model | 3716 ms/tok | — | dead end — no Metal on CPU | `03-MLX/the-journey.md` |
| Sanity check (Llama-3.3-70B spec, not Qwen) | 11.9 → 20.1 | 1.69× | confirms spec-decode harness works | `03-MLX/the-journey.md` |
| Broken-benchmark spec (`stream_generate`) | 26.5 | false negative | re-measured correctly as 37.6 | `03-MLX/the-journey.md` |

**MLX bandwidth / detail numbers** (M4 Max, Qwen3.5-27B-4bit):

| Metric | Value | Source |
|---|---|---|
| Peak memory bandwidth | 546 GB/s | `03-MLX/bandwidth-and-dispatch.md` |
| Total weight read per token | 13.7 GB (12.2 weights + 1.52 scales/biases @ group=64) | `03-MLX/bandwidth-and-dispatch.md` |
| Theoretical min latency | 25.1 ms/tok | `03-MLX/bandwidth-and-dispatch.md` |
| Theoretical max throughput | 39.8 tok/s (100% BW) | `03-MLX/bandwidth-and-dispatch.md` |
| Stock measured | 33.8 ms/tok = 29.5 tok/s = 74% BW | `03-MLX/bandwidth-and-dispatch.md` |
| Pipelined matmul bandwidth | ~439 GB/s (80% of peak) | `03-MLX/bandwidth-and-dispatch.md` |
| MTP acceptance (single head, n=1) | 79% → 1.79 tokens/step | `03-MLX/README.md`, `mtp-self-speculative-mlx.md` |
| MTP head on ANE (CoreML) | 2.24 ms median latency, exact match, 424.7M params | `03-MLX/bandwidth-and-dispatch.md` |

**MLX projections (NOT measured — flagged as such):** parallel MTP n=2 →
**52.6 tok/s**; n=3 → 58.3 tok/s; mixed-3/4-bit + n=2 → 57.4; if all fusions
land → 58.1 tok/s (1.97×); O(1)-per-token ceiling at N=8 → 213 tok/s. Entropy-
coding projection (80–90 tok/s) was **debunked** — measured 3.72 bits / 1.08×
on this exact model, not the claimed 1.1 bits / 3.3× (`03-MLX/bandwidth-and-dispatch.md`).

### 2.3 vLLM — GH200 480GB (W4A16, vLLM 0.19.0)

| Config | tok/s | vs baseline | Source |
|---|---:|---|---|
| Stock MTP spec=7, **batch=1** | **186** | baseline | `04-VLLM-GPU/README.md` (`vllm-patches-benchmarks.md`) |
| Stock MTP spec=7, **batch=8** | **1,030** | **+454% agg / 5.54× batch** | `04-VLLM-GPU/README.md` (`vllm-patches-benchmarks.md`) |
| MTP spec=5 (deploy.sh / fresh-install), batch=1 | **~192** (range 185–202) | — | `04-VLLM-GPU/deploy/gh200/07-FRESH-INSTALL.md` |
| Full model BF16, no spec, batch=1 | **125** | — | `04-VLLM-GPU/modal-self-speculative.md` (modal-mtp README) |
| Full model BF16 + MTP5 | **193** | 1.88× | `04-VLLM-GPU/modal-self-speculative.md` |
| **W4A16 abliterated + MTP5, single** | **176.6** | — | `04-VLLM-GPU/modal-self-speculative.md` |
| **W4A16 abliterated + MTP5, 32 concurrent** | **1800+** | — | `04-VLLM-GPU/modal-self-speculative.md` |

**Strategy sweep — none beat stock MTP** (GH200; baseline 186):

| Strategy / module | tok/s | vs baseline | Why | Source |
|---|---:|---|---|---|
| `adaptive_mtp.py` (EMA-gated chain length) | **186** | **0%** | all positions profitable; nothing to cut | `04-VLLM-GPU/vllm-patches-and-strategies.md`, `README.md` |
| `deltanet_transplant.py` (DeltaNet weight transplant) | **174** | **−6%** (within noise) | dedicated DeltaNet draft weights, no win | `04-VLLM-GPU/vllm-patches-and-strategies.md`, `gpu-designs.md` |
| `native_multi_head.py` (sibling MTP heads, weight swap) | **139** | **−25%** | swap overhead | `04-VLLM-GPU/vllm-patches-and-strategies.md` |
| `cascade_mtp_corrective.py` (depth-trained cascade) | **47** | **−75%** | training-data mismatch | `04-VLLM-GPU/vllm-patches-and-strategies.md`, `README.md` |
| Tree speculation (9 spec tokens) | **27** | **−85%** | drafter overhead | `04-VLLM-GPU/README.md` |
| `modal_mtp.py` (DeltaNet self-spec, no CUDA graphs) | **3.2** | **−98%** | draft path runs eager (no CUDA graphs) | `04-VLLM-GPU/vllm-patches-and-strategies.md`, `README.md`, `modal-self-speculative.md` |
| Standalone DeltaNet draft model | **5** | **−97%** (0% accept) | useless drafter | `04-VLLM-GPU/README.md` |
| `partial_layer_verify.py` (PLV, layer 60) | — | **−3 to −7%** | not worth it; early exit needs 87–94% of layers | `04-VLLM-GPU/vllm-patches-and-strategies.md`, `gpu-designs.md` |

**GH200 MTP acceptance per draft position** (spec=7; overall **55.5%**):

| Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 | Source |
|---|---|---|---|---|---|---|---|
| 87% | 68% | 54% | 39% | 28% | 21% | 16% | `04-VLLM-GPU/README.md` (`vllm-patches-benchmarks.md`) |

Monotonic decay → spec=5 is the sweet spot (positions 6–7 add little). The
deploy/fresh-install doc rounds these slightly: **85 / 68 / 54 / 40 / 31** (5
positions, spec=5) (`04-VLLM-GPU/deploy/gh200/07-FRESH-INSTALL.md`) — see §4.

**GH200 batch-scaling (the GPU-designs roofline doc — a different model arch; see §4):**

| Path | batch=1 | batch=2 | batch=4 | batch=8 | Source |
|---|---:|---:|---:|---:|---|
| Baseline (no spec), aggregate tok/s | 90.6 | 106.0 | 263.4 | **592.8** | `04-VLLM-GPU/gpu-designs.md` |
| Spec decode, aggregate tok/s | 36.7 | 26.2 | 87.0 | **225.5** | `04-VLLM-GPU/gpu-designs.md` |

Single-request on that arch: **baseline 94 / MTP-tree (9 tokens) 36 tok/s
(−62%)** (`04-VLLM-GPU/gpu-designs.md`). **This is NOT the 27B model — see §4.**

### 2.4 vLLM — RTX 5090 (32 GB, NixOS "captain", vLLM 0.19.0, MTP=5)

| Config | tok/s | MTP accept | Batch=4 agg | Model size | Source |
|---|---:|---:|---:|---:|---|
| **GPTQ W4A16 (Huihui abliterated), single 256 tok** | **151** | **51%** | 347 | 19.5 GB | `04-VLLM-GPU/README.md`, `quantization.md` (`autoawq-benchmarks.md`) |
| AWQ W4A16, single 256 tok | **77** | **31%** | 313 | 18.6 GB | `04-VLLM-GPU/README.md`, `quantization.md` |
| cyankiwi AWQ textonly (server preset) | **~140–149** | ~50–53% | ~403–450 | 19.1 GB | `04-VLLM-GPU/README.md` (`01-SERVER-STATUS.md`, `vllm-serve.sh`) |
| BF16 abliterated (no quant) | **125** | — | — | — | `04-VLLM-GPU/README.md` (modal-mtp README) |

**AWQ vs GPTQ head-to-head** (RTX 5090, MTP=5):

| Metric | GPTQ W4A16 | AWQ W4A16 | Source |
|---|---:|---:|---|
| Single 256 tok | **151** | 77 | `04-VLLM-GPU/quantization.md` |
| MTP acceptance | **51%** | 31% | `04-VLLM-GPU/quantization.md` |
| Batch=4 aggregate | **347** | 313 | `04-VLLM-GPU/quantization.md` |
| Model size | 19.5 GB | 18.6 GB | `04-VLLM-GPU/quantization.md` |

> GPTQ wins single-request + MTP acceptance (Hessian-optimal rounding preserves
> MTP-head quality); AWQ wins batch throughput in the `vllm-serve.sh` preset
> (AWQ 403 vs GPTQ 347 at batch=4) — but that preset compares *different base
> models*, so read it as "both ~quant-equivalent for batch" (`04-VLLM-GPU/quantization.md`).

**Expected throughput by GPU** (deploy doc projection table, not all measured):
RTX 5090 ~140 / batch=4 ~450; RTX 4090 ~80–100 / ~250; A6000 ~100–120 / ~350;
GH200 ~190 / ~500 (`04-VLLM-GPU/deploy/gh200/07-FRESH-INSTALL.md`).

---

## 3. Hardware reference

| Hardware | Spec | Model + quant | Source |
|---|---|---|---|
| **Apple M4 Max** | 16-core GPU, 128 GB unified, **546 GB/s** bandwidth, 48 MB SLC. **13.7 GB** weight read/tok (12.2 weights + 1.52 scales/biases @ group=64). Theoretical floor **25.1 ms/tok = 39.8 tok/s** (100% BW). Pipelined matmul ~439 GB/s (80%). | Qwen3.5-27B-4bit (15.65 GiB / 4.92 BPW as Q4_K_M in llama.cpp) | `_meta/CONVENTIONS.md`, `03-MLX/bandwidth-and-dispatch.md`, `02-LLAMACPP/benchmarks.md` |
| **NVIDIA GH200 480GB** | H100 die, 132 SMs, 96 GB HBM3e, spec 4.8 TB/s (~4.0 TB/s achievable), max SM clock 1980 MHz, ~1070 FP16 TFLOPS dense, NVLink-C2C 900 GB/s. vLLM 0.19.0. | Qwen3.5-27B BF16 and W4A16 (`j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-textonly`, ~19.1 GB) | `_meta/CONVENTIONS.md`, `04-VLLM-GPU/README.md` |
| **RTX 5090** | NixOS host "captain", 32 GB VRAM. vLLM 0.19.0. | Qwen3.5-27B GPTQ W4A16 (19.5 GB) + AWQ W4A16 (18.6 GB), MTP=5 | `_meta/CONVENTIONS.md`, `04-VLLM-GPU/README.md` |

Model architecture (the real Qwen3.5-27B): 64+1 layers (48 DeltaNet + 16 full
attention + 1 MTP head), hidden 5120, 24 heads / 4 KV, head_dim 256,
intermediate 17408, vocab 248320, 27.32 B params (`02-LLAMACPP/benchmarks.md`,
`04-VLLM-GPU/gpu-designs.md`).

---

## 4. Numbers that look like contradictions but aren't

Disambiguation of the famous ones. For the full registry and resolution status,
see **`../99-LEDGER/open-questions.md`**.

**GH200 186 vs 94 tok/s — the deepest one (two confounds stacked).** The benchmark
docs and deploy runbooks report **186 (batch=1) / 1030 (batch=8)** for the real
**27B** model; the GPU-designs roofline doc reports **94 (single-request) / 36
(spec)**. These are **not the same measurement and not the same model.** (1) The
94/36 doc's own architecture table is **hidden 3584, 28 heads, intermediate
18944, vocab 151936, ~16B "effective"** — a **mislabeled ~16B-class model**, not
Qwen3.5-27B (hidden 5120, vocab 248320; its FLOP math, 30.92 GFLOPs/tok, only
reproduces from the 3584 profile — the real 27B is 46.16). (2) Even setting the
model aside, 186/1030 are **aggregate/batched** production numbers while 94/36
are **single-request**. Both kept; neither rounded away. → `04-VLLM-GPU/gpu-designs.md` §1, `../99-LEDGER/open-questions.md`.

**13.98 mean vs 15.0 sweet-spot (llama.cpp).** Same recipe, two figures: **13.98**
is the **5-prompt mean** (the headline 1.99× over the 7.02 mean baseline);
**15.0** is the **single-prompt K=2/thresh=0.85 sweet-spot** from the K-sweep
table. Both verbatim from `02-LLAMACPP/the-recipe.md`. (Also: K=1 baseline shows
as 7.02 in the recipe's 5-prompt mean vs 7.64 in the README "honest performance"
table — mean-of-5 vs the single K=1-spec figure.) → `02-LLAMACPP/the-recipe.md`, `../99-LEDGER/open-questions.md`.

**MLX ~45 vs 51.1 (different stacks, both real).** `mlx-qwen-mtp/README.md`
projects **~45 tok/s (~1.52×)** for the **shipped repo's own stack**
(split-recurrence + partial rms_norm fusion). The **51.1** corpus headline comes
from the sibling **`parallel-mtp-voting`** project's adaptive MTP confidence chain
+ batch verify driver (`adaptive_mtp.py` / `stacked_v2.py`; only its logs are in
this corpus). Not a contradiction — different code paths. 51.1 is the corpus
headline. → `03-MLX/README.md`, `99-LEDGER/inventory-03-mlx.md`, `../99-LEDGER/open-questions.md`.

**MLX 52.0 vs 51.1 (version drift).** 52.0 is the historical memory-anchor on
**MLX 0.30.6**; 51.1 is the revalidation on current MLX — ~2% drift, noise-level.
51.1 is the current measured value. → `03-MLX/huihui-abliterated.md`.

**Per-position acceptance rounding variants (GH200).** The benchmark doc gives 7
positions **87/68/54/39/28/21/16** (spec=7); the deploy/fresh-install doc rounds
to 5 positions **85/68/54/40/31** (spec=5). Same monotonic-decay phenomenon,
different spec depth + rounding. Overall acceptance **55.5%** in both. →
`04-VLLM-GPU/README.md`, `04-VLLM-GPU/deploy/gh200/07-FRESH-INSTALL.md`.

**Modal MTP 176.6 (potential) vs 3.2 (measured).** The 125 / 193 / 176.6 / 1800+
modal-mtp table is the **projected potential** once draft-mode CUDA graphs are
captured; the **3.2 tok/s** is what was **actually measured** — catastrophically
slow because the draft path runs eager (no CUDA graphs). Not a contradiction:
unrealized design target vs measured-with-missing-piece. → `04-VLLM-GPU/modal-self-speculative.md`, `vllm-patches-and-strategies.md`.

**AWQ 31% vs GPTQ 51% acceptance (real, not noise).** GPTQ's Hessian-optimal
rounding preserves MTP-head draft quality; AWQ's activation-aware approach
degrades it more → lower acceptance → lower spec throughput. Both measured on
RTX 5090, MTP=5. → `04-VLLM-GPU/quantization.md`.

---

## Provenance & traceability note

Every figure above is traced to a `MTP/`-corpus doc cited inline (which in turn
footnotes its upstream source repo). No number on this page is invented,
averaged, or rounded by this synthesis. Cross-platform speedups are
each-relative-to-own-baseline and intentionally not unified.

**Numbers I could not trace to a corpus source doc:** none. Every value in the
task brief and in §1–§3 was located verbatim in a source doc and cited. The §4
disambiguations are cross-referenced to the cross-domain contradiction registry
[`../99-LEDGER/open-questions.md`](../99-LEDGER/open-questions.md) (13 resolved +
3 open) and backed by the per-domain docs and the `99-LEDGER/inventory-*`
contradiction sections.

---

## Related docs

- [`the-big-picture.md`](the-big-picture.md) — the whole arc; §9 reads these numbers per platform.
- [`glossary.md`](glossary.md) — the vocabulary (acceptance rate, K, bandwidth wall, …).
- [`corpus-map.md`](corpus-map.md) — where everything came from; "where do I find X?".
- [`../99-LEDGER/open-questions.md`](../99-LEDGER/open-questions.md) — the contradictions registry behind §4.
- [`../README.md`](../README.md) — the corpus front door.

---
*Sources read in full to build this table: `_meta/CONVENTIONS.md`;
`02-LLAMACPP/{README,the-recipe,benchmarks,optimization-variants,logs/README}.md`;
`03-MLX/{README,the-journey,bandwidth-and-dispatch,mtp-self-speculative-mlx,huihui-abliterated}.md`;
`04-VLLM-GPU/{README,gpu-designs,vllm-patches-and-strategies,quantization,precision-divergence,modal-self-speculative}.md`,
`04-VLLM-GPU/deploy/gh200/07-FRESH-INSTALL.md`; `99-LEDGER/inventory-0{3,4}-*.md`.*
