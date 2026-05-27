# llama.cpp server-mode benchmarks (raw log numbers)

> Every concrete number extracted from the two `llama-server` captures in
> [`logs/`](logs/). These are **server-mode** runs (HTTP `/completion`), distinct from
> the `llama-mtp-speculative` CLI runs that produced the 5-prompt recipe table in
> [`the-recipe.md`](the-recipe.md). They confirm the same headline: on this hybrid
> model the MTP draft path does **not** beat plain decode in throughput, because the
> single MTP head's accept rate is near-zero and a 0.8B companion only lifts it to ~49%.

## Shared run configuration (identical in both logs)

| Field | Value |
|---|---|
| Build | `8468 (19fdba56b)`, AppleClang 17.0.0.17000603, Darwin arm64 |
| Device | Apple M4 Max, MTL0 GPU (family Apple9), 128 GB unified, `recommendedMaxWorkingSetSize = 103079.22 MB` |
| Threads | `n_threads = 12`, `n_threads_batch = 12`, total 16 |
| Target model | `Qwen3.5-27B-Q4_K_M.gguf` — arch `qwen35`, 27.32 B params, **15.65 GiB (4.92 BPW)**, GGUF V3 |
| Tensor types | f32: 360, q4_K: 439, q6_K: 67 (866 tensors total) |
| Target arch hparams | `block_count=65`, `n_embd=5120`, `n_layer=65`, `n_head=24`, `n_head_kv=4`, `n_embd_head_k/v=256`, `n_ff=17408`, `nextn_predict_layers=1`, `full_attention_interval=4` |
| SSM (DeltaNet) hparams | `ssm.conv_kernel=4`, `state_size=128`, `group_count=16`, `time_step_rank=48`, `inner_size=6144` |
| Context | `n_ctx = 4096` (`n_ctx_train = 262144`, not fully used), `n_batch = 2048`, `n_ubatch = 512`, `flash_attn = auto`, `kv_unified = true` |
| Parallelism | `n_parallel = 4` (auto), `n_seq_max = 4` for the target context |
| Offload | `offloaded 66/66 layers to GPU`; `MTL0_Mapped model buffer = 16021.46 MiB`, `CPU_Mapped = 682.03 MiB` |
| Target KV cache | `272.00 MiB` (4096 cells, **17 layers**, 4/1 seqs), K (f16) 136.00 MiB + V (f16) 136.00 MiB — note: only the 17 full-attention layers carry KV; the 48 DeltaNet layers use recurrent state |
| Sampling defaults (from GGUF) | `top_k=20`, `top_p=0.95`, `temp=0.6` |
| Server | `http://127.0.0.1:8090`, 15 HTTP threads, `--cache-ram` prompt cache on |

Both runs print at load:
`[MTP-SEQRM] … NO checkpoint found — seq_rm FAILED` →
`common_speculative_is_compat: seq_rm not supported, but MTP model detected — using
checkpoint/restore for rollback` →
`srv load_model: model has 1 MTP layer(s) — auto-enabling MTP speculative decoding`.
i.e. the server **auto-enables** MTP spec whenever the GGUF carries a NextN layer, and
falls back to the snapshot/restore rollback primitive (infra patches 0000 + 07) because
plain `seq_rm` cannot trim the recurrent half.

---

## Log A — `llama_cpp_server.log` (MTP head **+ 0.8B companion draft**)

A second model is loaded as a draft model:

| Field | Value |
|---|---|
| Draft model | `Qwen3.5-0.8B-Q8_0.gguf` — arch `qwen35`, **752.39 M** params, **763.78 MiB (8.52 BPW)**, GGUF V3 |
| Draft arch | `block_count=24`, `n_layer=24`, `n_embd=1024`; offloaded 25/25 layers to GPU |
| Draft KV cache (per slot) | `48.00 MiB` (4096 cells, **6 layers**, 1/1 seqs), K 24 + V 24 MiB |
| Spec contexts | 4 slots, each "speculative decoding context initialized", `n_seq_max=1`, `n_batch=4096` |
| End-of-run memory | `MTL0: 98304 = 77418 free + (17903 self = 16021 model + 870 ctx + 1012 compute) + 2981 unaccounted`; Host 710 MiB |

### Per-request timing blocks (server `slot print_timing`)

| Req (task) | prompt eval | eval (decode) | total | n_tokens |
|---|---|---|---|---|
| 0   | 367.69 ms / 35 tok → **95.19 tok/s** | 3862.34 ms / 80 tok → **20.71 tok/s** (48.28 ms/tok) | 4230.04 ms / 115 | 114 |
| 82  | 358.45 ms / 35 tok → **97.64 tok/s** | 4530.13 ms / 94 tok → **20.75 tok/s** (48.19 ms/tok) | 4888.58 ms / 129 | 128 |
| 178 | 358.81 ms / 35 tok → **97.54 tok/s** | 3896.83 ms / 81 tok → **20.79 tok/s** (48.11 ms/tok) | 4255.64 ms / 116 | 115 |
| 261 | 359.11 ms / 35 tok → **97.46 tok/s** | 2211.53 ms / 41 tok → **18.54 tok/s** (53.94 ms/tok) | 2570.64 ms / 76 | 75 |
| 292 | 359.11 ms / 35 tok → **97.46 tok/s** | 1885.47 ms / 41 tok → **21.75 tok/s** (45.99 ms/tok) | 2244.58 ms / 76 | 75 |
| 322 | 359.18 ms / 35 tok → **97.44 tok/s** | 3134.24 ms / 68 tok → **21.70 tok/s** (46.09 ms/tok) | 3493.42 ms / 103 | 102 |

Decode throughput sits at **18.5–21.8 tok/s** regardless of MTP/draft — i.e. server-mode
MTP spec does **not** beat plain decode here (matches the CLI finding that K=1 MTP spec
is *slower* than plain decode; the companion draft barely moves the eval-time number).

### Drafter accept statistics (cumulative, `statistics … #calls(b,g,a)`)

`#calls(b,g,a)` = (batch-decode calls, generate calls, accept-bearing calls);
then `#gen drafts / #acc drafts / #gen tokens / #acc tokens / dur(b,g,a) ms`.

**MTP head** (`statistics mtp`) — drafts almost never accepted:

| Cumulative call | gen drafts | acc drafts | gen tokens | acc tokens | gen dur (ms) |
|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 0.000 |
| 2 | 0 | 0 | 0 | 0 | 0.000 |
| 3 | 0 | 0 | 0 | 0 | 0.000 |
| 4 | 28 | 0 | 28 | 0 | 1.075 |
| 5 | 55 | 0 | 55 | 0 | 2.112 |
| 6 | **98** | **2** | 98 | 2 | 3.839 |

→ MTP head accept ≈ **2/98 ≈ 2%** cumulative. The head's top-1 is structurally wrong on
this hybrid model (the same "head saturated, ~90% wrong" finding behind negative
optimization variant 09).

**0.8B companion draft** (`statistics draft`) — the real source of accepts:

| Cumulative call | gen drafts | acc drafts | gen tokens | acc tokens | gen dur (ms) |
|---|---|---|---|---|---|
| 1–3 | 0 | 0 | 0 | 0 | 0.000 |
| 4 | 28 | 12 | 28 | 12 | 264.541 |
| 5 | 55 | 25 | 55 | 25 | 272.777 |
| 6 | 95 | **47** | 95 | 47 | 416.974 |

→ Companion draft accept ≈ **47/95 ≈ 49%** cumulative. The 0.8B model is the effective
drafter; the MTP head contributes ~nothing. But its `gen dur` is large (264–417 ms
cumulative) — the draft-model forward cost roughly cancels the accepted-token savings,
so net throughput stays at plain-decode level.

### Per-token verify decisions (`[MTP-VERIFY]`)

- **58** `ACCEPTED` lines, **66** `REJECTED` lines (≈ 47% accept on verified tokens).
- ACCEPTED lines show `sampled == draft` (e.g. `pos=0: sampled=5010, draft=5010, ACCEPTED`);
  REJECTED lines show a mismatch (e.g. `pos=0: sampled=40718, draft=15250, REJECTED`).
- `bonus pos=1: sampled=…` lines are the free verify-tail token committed on accept.

### Final per-slot perf prints (shutdown, `llama_perf_context_print`)

Eight streams (target + draft × 4 slots). Representative target-context numbers:

| load time (ms) | prompt eval | eval (decode) | total | graphs reused |
|---|---|---|---|---|
| 89320.24 | 116.40 ms / 58 tok → 498.28 tok/s | 141.14 ms / 31 runs → **219.64 tok/s** (4.55 ms/tok) | 1275522.23 ms / 89 | 31 |
| 77836.62 | 125.80 ms / 60 tok → 476.97 tok/s | 141.77 ms / 31 runs → **218.66 tok/s** (4.57 ms/tok) | 1275524.04 ms / 91 | 31 |
| 56216.36 | 26.25 ms / 38 tok → 1447.67 tok/s | 13.72 ms / 3 runs → **218.63 tok/s** (4.57 ms/tok) | 325406.37 ms / 41 | 3 |
| 47689.56 | 442.78 ms / 96 tok → 216.81 tok/s | 63.76 ms / 14 runs → **219.57 tok/s** (4.55 ms/tok) | 1275527.71 ms / 110 | 14 |

These `llama_perf_context_print` "runs" are the **draft model's** isolated forward passes
(0.8B, ~4.55 ms/tok ≈ 219 tok/s) and small batched verifies — not the user-visible
decode rate. The user-visible rate is the `slot print_timing` eval line above (~20 tok/s).

---

## Log B — `llama_cpp_server_nomtp.log` (no draft model — pure baseline)

- **No draft model loaded** (no `loading draft model` line, no `statistics draft`).
- MTP head present and auto-enabled, but for the single request it generated **0 drafts**:
  `statistics mtp: #calls(b,g,a) = 1 0 0, #gen drafts = 0, #acc drafts = 0`. → this is the
  pure greedy-decode ground truth.
- Same context config as Log A (4 slots, n_ctx 4096, KV 272 MiB / 17 layers).

### The single request

| | prompt eval | eval (decode) | total |
|---|---|---|---|
| task 0 | 476.56 ms / 38 tok → **79.74 tok/s** | 14413.02 ms / 300 tok → **20.81 tok/s** (48.04 ms/tok) | 14889.57 ms / 338 tok |

End-of-run memory: `MTL0: 98304 = 80399 free + (17903 self = 16021 model + 870 ctx +
1012 compute) + 0 unaccounted`; Host 710 MiB.

---

## Side-by-side: does the draft path help in server mode?

| Run | Draft model | Decode tok/s (eval) | prompt eval tok/s | MTP-head accept | Companion accept |
|---|---|---|---|---|---|
| **No-MTP baseline** (Log B) | none | **20.81** (300 tok) | 79.74 | n/a (0 drafts) | n/a |
| **MTP + 0.8B draft** (Log A) | Qwen3.5-0.8B-Q8_0 | **18.5–21.8** (per req) | 95–98 | ~2% (2/98) | ~49% (47/95) |

**Conclusion (consistent with [`the-recipe.md`](the-recipe.md) and the CLI numbers):**
in server mode the MTP+draft path lands at the *same* ~20 tok/s as the no-draft baseline.
The MTP head's accept rate is ~2% (structurally wrong top-1); the 0.8B companion accepts
~49% of its drafts but its forward cost (264–417 ms cumulative `gen dur`) eats the
savings. Server-mode spec decoding on this hybrid model is throughput-neutral — the win
in the CLI path comes only from the **confidence-gated chained MTP** recipe
(`MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`), which these server logs were not run with.

> Note on prompt-eval: the MTP+draft run shows faster prompt eval (95–98 tok/s vs 79.74)
> because its prompts are shorter (35 tok vs 38 tok) and warm-cached across the 6 requests;
> not an MTP effect.

---

## Related

- [`the-recipe.md`](the-recipe.md) — the CLI path (confidence-gated chained MTP) these server logs were *not* run with.
- [`logs/`](logs/) — the two raw `llama-server` captures these numbers come from.
- [`optimization-variants.md`](optimization-variants.md) — the ~2% MTP-head accept here is the same "head saturated" finding as variant 09.
- [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) — the arch hparams logged at model load (the 17 attention-layer KV note).
- [`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md) — the headline numbers across all platforms.

---
*Provenance: numbers extracted verbatim from `logs/llama_cpp_server.log` and
`logs/llama_cpp_server_nomtp.log` (copied from `qwen-inference-lab/logs/`). Hardware/arch
facts cross-checked against `MTP/_meta/CONVENTIONS.md`.*
