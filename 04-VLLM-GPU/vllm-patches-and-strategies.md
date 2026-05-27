# vLLM Patches & Speculative-Decode Strategies

Two catalogs: (1) the **patches** that modify stock vLLM 0.19.0 to fix bugs or
add hooks, and (2) the **strategy modules** that explore ways to beat stock MTP.
All files are copied under [`patches/`](patches/) and
[`optimizations/`](optimizations/).

> **Bottom line up front:** none of the strategies beat stock MTP on GH200 (see
> the results table in [`README.md`](README.md)). The patches fix real bugs in
> experimental vLLM features and enable the experiments; the experiments mapped
> the design space and mostly confirmed stock MTP is already near-optimal here.

---

## Part 1 — The patches (`patches/`)

`apply.sh` manages 7 named patches against the installed vLLM. It backs up files
to `.bak`, applies with `patch -p1/-p0 --forward`, and — crucially — **reverts by
extracting clean files from the pip wheel** (not from `.bak`, which may itself be
patched), then clears `~/.cache/vllm/torch_compile_cache/`.

```bash
./apply.sh list           # show patches
./apply.sh check          # report PATCHED/stock per file
./apply.sh all            # safe set: eagle + qwen3_next
./apply.sh <name>         # apply one
./apply.sh revert         # restore ALL from wheel
./apply.sh revert <name>  # restore one from wheel
```

### Patch catalog

| Patch file | apply.sh name | Target file | What it fixes / adds |
|---|---|---|---|
| `eagle.patch` | `eagle` | `v1/spec_decode/eagle.py` | 5 fixes: (1–3) MRoPE/tree-attention crashes in `propose_tree` (position accessors, `model_returns_tuple()` unpacking, 1D position arithmetic — see `gpu-designs.md` EAGLE PRs); (4) **adaptive chained MTP** env-var hooks (`MTP_CHAIN_KMAX`, `MTP_CHAIN_THRESH=0.85`) with confidence-gated re-runs of the MTP forward; (5) warmup pass for the chain path so AOT/CUDA-graph captures the inputs_embeds variant. |
| `qwen3_next.patch` | `qwen3_next` | `models/qwen3_next.py` | The `_skip_attention` identity pass-through on full-attention layers (the core modal-MTP draft hook). Also fixes the compiled-forward tensor shape for modal draft mode. |
| `qwen3_5-shadow-state.patch` | `qwen3_5` | `models/qwen3_5.py` | Adds `set_draft_mode()`, `snapshot_deltanet_state()` (persistent reused buffers), `restore_deltanet_state()` for modal_mtp. |
| `gdn-shadow-state.patch` | `gdn` | `mamba/gdn_linear_attn.py` | DeltaNet uses a `_draft_kv_cache` shadow during draft mode so draft forwards don't corrupt the real recurrent state. |
| `speculative-dual-mode.patch` | `speculative` | `config/speculative.py` | Adds class flag `_skip_mtp_conversion`: only convert `qwen3_5 → qwen3_5_mtp` for self-speculative; a **standalone draft model** keeps its own arch (prevents hijacking it into the MTP extraction path). |
| `speculative-mtp-tree-compat.patch` | `gpu_model_runner` | `v1/worker/gpu_model_runner.py` | **CUDA-graph segfault fix:** force `cudagraph_mode=NONE` when PIECEWISE + spec decode + `min_cg_support==NEVER` + tree (`uniform_decode_query_len>1`); replaying tree-shaped attention metadata with PIECEWISE crashes EngineCore. |
| `recurrent-rollback.patch` | `rollback` | `mamba/gdn_linear_attn.py` (+ `qwen3_5.py`) | O(1) GDN state rollback for MTP verification: `GDNRollbackCheckpoint` clones ssm_state + conv_state per draft position; restore in O(1). ~893 MB for MTP=5. Apply **last**. |
| `modal_mtp.patch` | (added file) | `v1/spec_decode/modal_mtp.py` | The full 478-line `ModalMTPProposer` (new file). See `modal-self-speculative.md`. |
| `speculative-draft-override.patch` | — | `config/speculative.py` | Earlier/alternate form of the dual-mode fix: only convert to MTP when `mtp_num_hidden_layers` is actually declared. (Superseded by `speculative-dual-mode.patch`'s class-flag approach.) |
| `gdn-inhibition-cycle.patch` | — | `models/qwen3_next.py` | One-time GDN "inhibition" recurrence cycle after prefill→decode transition (in-place state update, hidden untouched). Cost ~2.4 ms, **improves MTP acceptance ~3% (50%→53%)**. |
| `int8-embedding.patch` | — | `vocab_parallel_embedding.py` (+ `qwen3_5.py`) | `quantize_to_int8()` on the embedding (CPU-side, per-row scale) + INT8 forward path. **Saves ~1.27 GB VRAM.** Low-VRAM GPUs only (RTX 5090); skip on GH200. |
| `gh200-strip-torch-dep.patch` | — | `pyproject.toml` | Removes `torch>=2.0` from build requires so the pre-built ARM+CUDA system torch isn't replaced by a CPU-only PyPI wheel on GH200. |
| `llmcompressor-conv1d.patch` | — | `llmcompressor/.../module.py` | Import `Conv1D` from `transformers.pytorch_utils` (moved) — unblocks GDN conv1d quantization. See `quantization.md`. |
| `modal-01-skip-attention.patch` | — | `models/qwen3_next.py` | modal-mtp repo's version of the `_skip_attention` hook (≈ `qwen3_next.patch`). |
| `modal-02-draft-mode-and-state.patch` | — | `models/qwen3_5.py` | modal-mtp repo's `set_draft_mode` + snapshot/restore (simpler than `qwen3_5-shadow-state.patch`). |
| `modal-03-model-runner-draft-loop.patch` | — | `v1/spec_decode/modal_mtp.py` | modal-mtp repo's 280-line proposer + the 5 model-runner integration points (as comments). |

> The `apply.sh` `revert_all` also deletes non-stock files it knows about:
> `modal_mtp.py`, `native_multi_head.py`, `sibling_sequential.py`,
> `adaptive_mtp.py`, `enhanced_mtp_proposer.py`, `deltanet_adjuster.py`.

---

## Part 2 — Strategy modules (`optimizations/`)

| Module | Idea | Result / status |
|---|---|---|
| `adaptive_mtp.py` (879L) | EMA-gated chain length + immediate rollback: track per-position acceptance, stop drafting when EMA[K] drops below threshold; drop-in wrapper around EagleProposer, zero vLLM changes. | **186 tok/s = 0% vs baseline** — "all positions profitable", adaptivity found nothing to cut. |
| `modal_mtp.py` (377L) | Self-speculative: run target in draft mode (skip 16 attn layers). | **3.2 tok/s (-98%)** — no CUDA graphs for draft path. See `modal-self-speculative.md`. |
| `partial_layer_verify.py` (597L) | PLV: verify draft tokens with only first N layers; if early-exit logits agree, skip N+1..63. Needs `--enforce-eager`. Env defaults: `PLV_EXIT_LAYER=32`, `PLV_FULL_VERIFY_INTERVAL=8`, `PLV_AGREEMENT_THRESHOLD=0.90`. | **-3 to -7%** (layer 60). Not worth deploying — see `gpu-designs.md` PLV boundaries. |
| `plv_bench.py`, `plv_layer60_bench.py` | PLV benchmarks: p_agree at full-attention boundaries; layer-60 early-exit timing with fallback. | Layer 56 ≈ 30% top-1 / 57% top-5; layer 60 ≈ 53% / 83%. Aggressive skip ruled out (need 87–94% of layers). |
| `early_verify_probe.py` (371L) | Early-exit probe: raw PLV vs trained regression adapter (h_L → h_63) vs cosine alignment, two-phase (cache hiddens, then train adapters). Constants: `HIDDEN_DIM=5120`, `VOCAB_SIZE=248320`, `PROBE_LAYERS=[3,7,11,15,31,47]`. | Adapters lift p_agree (1.6%→25% @L3) but plateau far below usable. |
| `deltanet_transplant.py` (489L) | Copy dedicated DeltaNet-draft-model weights into the main model's 48 DeltaNet slots (in-place or merged checkpoint). Mapping `main = draft + draft//3`. | **174 tok/s (-6%, within noise).** See `gpu-designs.md` transplant. |
| `deltanet_transplant_w4a16.py` (169L) | Same transplant but W4A16→W4A16 (qweight/qzeros/scales). Skips `in_proj_a/b` (format mismatch). | Tooling — enables transplant on quantized models. |
| `cascade_mtp_corrective.py` (331L) | Train depth-specific MTP heads with CE + **MSE-to-ideal-hidden** loss so each head reduces drift for the next. | **47 tok/s (-75%)** — training-data mismatch. |
| `deltanet_adjuster.py` (449L) | Between MTP chain steps, run a few **attention-only** DeltaNet layers (skip MLP, ~92% of FLOPs) to correct hidden-state drift at deep draft positions. | 4 attn-only DeltaNet layers ≈ 33% of one MTP step. Refinement primitive. |
| `enhanced_mtp_proposer.py` (478L) | Wraps EagleProposer/AdaptiveMTPProposer to inject `deltanet_adjuster` between chain steps. | Integration of the adjuster idea. |
| `native_multi_head.py` (570L) | K sibling MTP heads via **in-place weight-swap** through the same compiled/CUDA-graphed path; route to the best head by tracked acceptance. Pre-fuses qkv/gate_up at bank load so swap is one `copy_()`. Env: `SIBLING_MTP_HEADS_DIR`, `NUM_SIBLING_HEADS=0`, `SIBLING_HEAD_SELECTION=round_robin`, `SIBLING_EMA_ALPHA=0.1`. Swap cost ~1 ms/head (849 MB @ 900 GB/s) vs ~5 ms verify. | **139 tok/s (-25%)** — swap overhead. (Weight-swap correctness verified — see `gpu-designs.md`.) |
| `sibling_sequential.py` (812L) | K=3 sibling heads run **sequentially** via cheap `chain_forward` (~2%/step, no KV cache); submit best head's chain; switch active head by EMA. Avoids TREE_ATTN entirely. | Drop-in EagleProposer replacement. |
| `selective_state_snapshot.py` (311L) | Snapshot only the **active batch** DeltaNet slots for modal_mtp (O(batch), ~598 MB for 8 slots) instead of all 256 (~18.7 GB). | The fix for modal_mtp's disabled snapshot. State geometry doc. |
| `microgreens/mtp_clone.py` (256L) | Clone the single trained MTP head into K noisy siblings (Gaussian σ); only the 15 unique per-head tensors cloned (embed/lm_head shared). | Step 1 of sibling-heads pipeline. |
| `microgreens/mtp_diversity_train.py` (721L) | Fine-tune K siblings jointly: CE per draft position + **diversity penalty** (minimize pairwise cosine sim of logits). | Step 2 — trains diverse siblings. |
| `microgreens/sibling_mtp_proposer.py` (728L) | Run K siblings **in parallel** → K candidates/position → **tree-structured** verification via vLLM tree attention. Each head ~200 MB int4, shares embed+lm_head. | vLLM integration for tree-spec drafting. |

### Scripts (`optimizations/scripts/`)

| Script | Purpose |
|---|---|
| `vllm-tree-spec.sh` | Launch vLLM with a 3-2 branching tree (`[(0,),(1,),(2,),(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]`), `--attention-backend TREE_ATTN --enforce-eager`. 9-node tree, depth 2, 12 draft tokens verified in one pass. |
| `bench-tok-s.py` | Throughput benchmark over 5 fixed prompts against the OpenAI endpoint; reports per-request avg + aggregate tok/s. |
| `quantize_deltanet.py` | CPU RTN GPTQ W4A16 quantizer for the DeltaNet draft model (see `quantization.md`). |

### Modal validation/diagnostic Python (`optimizations/`)

| File | Purpose |
|---|---|
| `modal_mtp_proposer_reference.py` | Reference copy of `modal-mtp/modal_mtp.py` (the ModalMTPProposer, with the 5 integration points). |
| `diagnose_divergence.py` | Layer-by-layer hidden-state divergence diagnostic (normal vs skip-attention). See `precision-divergence.md`. |
| `validate_draft_accuracy.py` | FP32/CPU 5-prompt × 50-token draft-accuracy check (the 100% result). |
| `validate_extended.py` | 12-prompt × 100-token extended validation (CPU FP16 → 100%). |
| `validate_extended_output.log` | The decisive log: **1146/1146 tokens, 100% CPU FP16**. |
| `test_partial_skip.py` | Sweep partial attention-skip configs (how many of 16 layers can be skipped). |

## Related

- [`gpu-designs.md`](gpu-designs.md) — the design docs behind these patches (EAGLE PRs, weight-swap, two-graph dispatch, PLV, transplant).
- [`modal-self-speculative.md`](modal-self-speculative.md) — the `modal_mtp` mechanism the `qwen3_next`/`qwen3_5-shadow-state`/`modal_mtp` patches implement.
- [`quantization.md`](quantization.md) — the `llmcompressor-conv1d` patch and `quantize_deltanet.py` in context.
- [`deployment.md`](deployment.md) — how `apply.sh` and these patches fit the install runbooks.
- [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) — the technique behind `recurrent-rollback.patch`.
- [`README.md`](README.md) — the results table (why none of the strategies beat stock MTP).

---
*Provenance: `qwen-ops/vllm/patches/*` (14 incl apply.sh),
`qwen-ops/vllm/optimizations/*` (14), `qwen-ops/vllm/microgreens/*`,
`qwen-ops/vllm/scripts/*`, `qwen-ops/research/benchmarks/vllm-patches-benchmarks.md`,
`modal-mtp/*.py` + `modal-mtp/patches/*`.*
