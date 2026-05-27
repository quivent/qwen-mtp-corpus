# 04 — vLLM / Datacenter-GPU Track

The vLLM side of the Qwen3.5-27B MTP speculative-decoding effort: production
deployment on **NVIDIA GH200 480GB** and **RTX 5090 (NixOS)**, a pile of vLLM
patches and experimental speculative-decode strategies, the **"modal"
self-speculative** drafting idea (one model, two speeds), the
**precision-divergence saga**, and **AWQ/GPTQ W4A16 quantization** for the
hybrid DeltaNet+attention architecture.

## TL;DR for a reader

- **Stock MTP speculative decoding is the only thing that beat baseline.** On
  GH200, `--speculative-config '{"method":"mtp","num_speculative_tokens":5..7}'`
  on the W4A16 model is the production config. None of the ~14 experimental
  strategies (modal self-spec, tree spec, sibling heads, cascade, partial-layer
  verify, DeltaNet transplant…) outperformed it.
- **The architecture is a hybrid:** 64 layers, every 4th is full attention
  (16 total), the other 48 are DeltaNet/GDN linear-attention with rolling
  recurrent state — a strict **3:1 DeltaNet:attention pattern**.
- **The "modal" insight:** that 3:1 pattern *is* a speculative schedule. Skip
  the 16 attention layers → an O(1) draft model that *is* the target model.
  Proven 100% accurate on CPU; diverges on GPU due to a **fused-kernel
  numerical difference, not a precision problem** (see `precision-divergence.md`).
- **GPTQ > AWQ for MTP serving:** GPTQ W4A16 keeps the MTP head's draft quality
  higher (51% vs 31% acceptance), so it is the recommended quant on RTX 5090.

## Results table (exact measured numbers)

### GH200 480GB (H100 die, 96 GB HBM3e, vLLM 0.19.0)

| Config | tok/s | vs baseline | Source |
|---|---:|---|---|
| Stock MTP spec=7, **batch=1** | **186** | baseline | `vllm-patches-benchmarks.md` |
| Stock MTP spec=7, **batch=8** | **1,030** | **+454% agg / 5.54× batch** | `vllm-patches-benchmarks.md` |
| MTP spec=5 (deploy.sh / fresh-install), batch=1 | ~192 (185–202) | — | `07-FRESH-INSTALL.md` |
| Full model BF16, no spec, batch=1 | 125 | — | modal-mtp README |
| Full model BF16 + MTP5 | 193 (1.88×) | — | modal-mtp README |
| **W4A16 abliterated + MTP5, single** | **176.6** | — | modal-mtp README |
| **W4A16 abliterated + MTP5, 32 concurrent** | **1800+** | — | modal-mtp README |
| Tree speculation (9 spec tokens) | 27 | -85% | `vllm-patches-benchmarks.md` |
| DeltaNet self-spec (modal_mtp, no CUDA graphs) | 3.2 | -98% | `vllm-patches-benchmarks.md` |
| Standalone DeltaNet draft model | 5 | -97% (0% accept) | `vllm-patches-benchmarks.md` |
| Sibling MTP heads (weight swap) | 139 | -25% (swap overhead) | `vllm-patches-benchmarks.md` |
| Adaptive MTP chain length | 186 | 0% (all positions profitable) | `vllm-patches-benchmarks.md` |
| DeltaNet weight transplant | 174 | -6% (within noise) | `vllm-patches-benchmarks.md` |
| Partial-layer verification (layer 60) | — | -3 to -7% (not worth it) | `vllm-patches-benchmarks.md` |
| Cascade MTP (depth-trained) | 47 | -75% (training mismatch) | `vllm-patches-benchmarks.md` |

> **One optimization, none beat baseline.** The 11 patches fix real bugs in
> experimental vLLM features; the strategies explore the design space; but
> **stock MTP wins** on this model + hardware. See `gpu-designs.md` for *why*
> (the GH200 single-request workload is overhead-bound, not compute- or
> bandwidth-bound).

### RTX 5090 (32 GB, NixOS "captain", vLLM 0.19.0, MTP=5)

| Config | tok/s | MTP accept | Batch=4 agg | Model size | Source |
|---|---:|---:|---:|---:|---|
| **GPTQ W4A16 (Huihui abliterated), single 256 tok** | **151** | **51%** | 347 | 19.5 GB | `autoawq-benchmarks.md` |
| AWQ W4A16, single 256 tok | 77 | 31% | 313 | 18.6 GB | `autoawq-benchmarks.md` |
| cyankiwi AWQ textonly (server preset) | ~140–149 | ~50–53% | ~403–450 | 19.1 GB | `01-SERVER-STATUS.md`, `vllm-serve.sh` |
| BF16 abliterated (no quant) | 125 | — | — | — | modal-mtp README |

> The RTX 5090 server preset values in `vllm-serve.sh` quote **GPTQ 151 tok/s
> single / 347 batch=4** and **AWQ 149 single / 403 batch=4** — note AWQ wins
> batch throughput while GPTQ wins single-request + MTP acceptance.

### MTP acceptance rate per draft position (GH200, spec=7)

| Pos 1 | Pos 2 | Pos 3 | Pos 4 | Pos 5 | Pos 6 | Pos 7 |
|---|---|---|---|---|---|---|
| 87% | 68% | 54% | 39% | 28% | 21% | 16% |

Overall MTP acceptance on the known-good GH200 config: **55.5%**. This monotonic
decay is *why* `num_speculative_tokens=5` is the sweet spot — positions 6–7 add
little. (Per-position figures in `07-FRESH-INSTALL.md` round slightly:
85/68/54/40/31.)

## Documents in this track

| Doc | What it covers |
|---|---|
| [`modal-self-speculative.md`](modal-self-speculative.md) | "Same model, two speeds" — skip the 16 attention layers to draft. The 3:1 pattern *is* the spec schedule. Snapshot/restore DeltaNet state. Done-vs-to-build status. |
| [`precision-divergence.md`](precision-divergence.md) | **The saga, resolved.** FP32-CPU 100% → FP16-GPU ~15% → the reversal: CPU FP16 = 100% too, so it's **CUDA-kernel numerics, not precision**. Full test matrix. |
| [`quantization.md`](quantization.md) | AWQ vs GPTQ W4A16 for the hybrid arch, dual layer-type handling, post-quant MTP weight injection, the conv1d llmcompressor fix, RTX 5090 results. |
| [`deployment.md`](deployment.md) | GH200 install/deploy runbook + RTX 5090/NixOS runbook. Exact commands, env vars, gotchas. |
| [`vllm-patches-and-strategies.md`](vllm-patches-and-strategies.md) | The 16 patches — 13 vLLM + 3 modal-mtp (what each fixes) + the optimization strategies as explained tables. |
| [`gpu-designs.md`](gpu-designs.md) | The 9 GPU/vLLM design docs distilled: compute/bandwidth roofline, CUDA-graph weight swap, two-graph dispatch, propose_tree analysis, MTP-tree config path, PLV boundaries, DeltaNet transplant, EAGLE PR descriptions. |

## Copied artifacts

| Folder | Contents |
|---|---|
| [`patches/`](patches/) | 13 vLLM patches + `apply.sh` + 3 modal-mtp patches (`modal-0{1,2,3}-*.patch`). |
| [`optimizations/`](optimizations/) | 14 vLLM strategy modules + `microgreens/` (sibling MTP heads) + `scripts/` + modal-mtp validation/diagnose Python + the validation log. |
| [`quantization/`](quantization/) | autoawq-qwen35 model class, MTP injector, and the 4 `.patch` files for AutoAWQ. |
| [`deploy/`](deploy/) | gh200 (deploy.sh, install docs) + rtx5090 (serve/watchdog scripts, NixOS config + guide). |

## Hardware reference (exact strings)

- **GH200**: NVIDIA GH200 480GB. H100 die, 132 SMs, 96 GB HBM3e, spec 4.8 TB/s
  (~4.0 TB/s achievable), max SM clock 1980 MHz, ~1070 FP16 TFLOPS dense,
  NVLink-C2C 900 GB/s. vLLM 0.19.0. BF16 and W4A16.
- **RTX 5090**: NixOS host "captain" (32 GB VRAM). GPTQ W4A16 + AWQ W4A16. vLLM 0.19.0.

## Related

- [`modal-self-speculative.md`](modal-self-speculative.md) · [`precision-divergence.md`](precision-divergence.md) · [`quantization.md`](quantization.md) · [`deployment.md`](deployment.md) · [`gpu-designs.md`](gpu-designs.md) · [`vllm-patches-and-strategies.md`](vllm-patches-and-strategies.md) — the docs in this track.
- [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) — the 3:1 hybrid layer table (canonical).
- [`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md) — unified cross-platform benchmark table.
- [`../00-OVERVIEW/glossary.md`](../00-OVERVIEW/glossary.md) — MTP, DeltaNet/GDN, W4A16 terms.
- [`../99-LEDGER/open-questions.md`](../99-LEDGER/open-questions.md) — the precision saga + the GH200 roofline-arch mislabel.

---
*Provenance: synthesized from `modal-mtp/`, `qwen-ops/vllm/`,
`qwen-ops/quantization/`, `qwen-ops/deploy/`, and
`qwen-ops/research/{benchmarks,findings,designs}/`. See
`../99-LEDGER/inventory-04-vllm-gpu.md` for the full source→destination map.*
