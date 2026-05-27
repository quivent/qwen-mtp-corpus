# MLX track — source artifacts

Self-contained copy of the MLX (Apple Silicon) implementation files. Each file's
origin is noted below. The consolidated implementation (`mlx-qwen-mtp/src/`) and the
earlier consolidation (`qwen-ops/mlx/`) are **byte-identical**; this copy is from the
canonical `mlx-qwen-mtp/src/`.

## The shipped implementation (the 42.7 tok/s working version)

| File | Origin | What it is |
|---|---|---|
| `mtp_head.py` | `mlx-qwen-mtp/src/mtp_head.py` ≡ `qwen-ops/mlx/mtp_head.py` | `MTPHead` nn.Module (one full gated-attention transformer layer + norms + eh-projection fc) and `load_mtp()`. The reverse-engineered Qwen3.5 MTP head. |
| `generate.py` | `mlx-qwen-mtp/src/generate.py` ≡ `qwen-ops/mlx/generate.py` | `fused_conv1d_silu`, `fused_gdn_step` Metal kernels, `fused_gdn_call_v2` (patched GatedDeltaNet forward, 4→1 fused projection), `patch_model`, and `mtp_generate` — the split-recurrence rollback generation loop. |
| `fused_kernels_t2.py` | `mlx-qwen-mtp/src/fused_kernels_t2.py` ≡ `qwen-ops/mlx/fused_kernels_t2.py` | The T=2 verify kernels: `fused_conv1d_silu_t2`, `fused_gdn_step_with_intermediate` (captures intermediate state after token 0 for zero-cost rollback), and `fused_rms_norm_qmv` (the rms_norm+quantized_matmul fusion). |
| `extract_weights.py` | `mlx-qwen-mtp/src/extract_weights.py` ≡ `qwen-ops/mlx/extract_weights.py` | Extracts the 15 MTP tensors from the HF checkpoint, applies the `+1.0` norm shift, quantizes 2D matrices to 4-bit (group_size=64), saves standalone safetensors. |
| `__init__.py` | `mlx-qwen-mtp/src/__init__.py` | Package exports. |

## The exploration / journey code

| File | Origin | What it is |
|---|---|---|
| `fused_gdn.py` | `qwen-inference-lab/kernels/fused_gdn.py` | The full V2–V7 exploration: V2 fused kernels, V3 per-layer `mx.compile`, V4 compiled attention, V5 **monolithic compile** (`_build_monolithic_decode`, one `mx.compile` for all 64 layers), V7 **GPU-resident loop** (`_build_gpu_loop`, `gpu_generate`), plus an in-file `MTPHead`/`load_mtp`. 1666 lines. Most of this is *dead ends* (see `../the-journey.md`); the winning kernels were extracted into `generate.py`/`fused_kernels_t2.py`. |
| `bench_v7.py` | `qwen-inference-lab/benchmarks/bench_v7.py` | Speculative-decode benchmark harness (stock vs V5 vs draft-model spec decode, sweeps n=2..6). Draft model `mlx-community/Qwen3.5-0.8B-MLX-4bit`. |
| `extract_mtp_huihui.py` | `qwen-inference-lab/benchmarks/extract_mtp_huihui.py` | Parametrized MTP-head extractor (works against any Qwen3.5-27B fp16 HF repo, vanilla or abliterated). Uses `mx.load` to handle bf16 shards. Produces the 265 MB drop-in MTP safetensors. |

## Logs

| File | Origin | What it is |
|---|---|---|
| `logs/adaptive_mtp_vanilla.log` | `qwen-inference-lab/logs/adaptive_mtp_vanilla.log` | 2026-04-08 revalidation, vanilla model: **51.1 tok/s** (threshold=0.8, max_chain=2). |
| `logs/adaptive_mtp_huihui.log` | `qwen-inference-lab/logs/adaptive_mtp_huihui.log` | 2026-04-08 revalidation, Huihui abliterated: **49.5 tok/s**. |

> The `adaptive_mtp.py` / `stacked_v2.py` driver that produces the 51.1 tok/s adaptive
> chain lives in the sibling `parallel-mtp-voting` project, which is **not** in this
> corpus. The logs above are the only captured artifacts of that run; their numbers are
> distilled in `../README.md` and `../the-journey.md`.

> `qwen-inference-lab/logs/llama_cpp_server{,_nomtp}.log` are llama.cpp server logs
> (Agent B's domain) and are not copied here. See `../../02-LLAMACPP/`.
