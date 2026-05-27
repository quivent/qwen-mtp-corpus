# Huihui abliterated variant — MLX conversion + benchmark

Running the **uncensored** Qwen3.5-27B through the full MLX optimization stack (MTP head
+ adaptive confidence chain) and quantifying the cost of abliteration.

- **Date:** 2026-04-08
- **Model:** `huihui-ai/Huihui-Qwen3.5-27B-abliterated` (uncensored fine-tune of
  Qwen3.5-27B via refusal-direction abliteration)
- **Hardware:** M4 Max, 128 GB, 546 GB/s
- **MLX:** 0.31.1

**Bottom line: abliteration costs ~3% throughput (51.1 → 49.5 tok/s).** The MTP head is
preserved intact by abliteration; the small cost is from a slightly noisier MTP-confidence
distribution on the shifted residual stream.

---

## Workflow (conversion)

1. **Download fp16 source** (~52 GB, 11 safetensors shards) from HuggingFace via `hf` CLI.
2. **Convert base to MLX 4-bit** (~14 GB):
   `mlx_lm.convert --hf-path ... -q --q-bits 4`.
   Stock `mlx_lm.convert` **silently drops the 15 MTP head tensors** during the
   load-and-resave step because the Python model class has no `mtp` field — unknown keys
   are discarded without warning.
3. **Extract the MTP head separately** with `extract_mtp_huihui.py` (in `src/`): loads the
   4 fp16 shards containing MTP tensors via `mx.load` (handles bf16 natively, unlike the
   numpy-framework safetensors path), applies the `+1.0` norm shift, quantizes 2D matrices
   to 4-bit (group_size=64), keeps small tensors/norms as bf16. Output: **265 MB**
   standalone safetensors, drop-in for `load_mtp()`.
4. **Symlink** so existing code picks up the abliterated head without edits:
   `ln -sf mtp_weights_huihui.safetensors mtp_weights.safetensors`.

### Is the MTP head preserved by abliteration? — Yes.
The Huihui fp16 repo contains all 15 MTP tensors intact:
```
mtp.fc.weight                                 (5120, 10240)
mtp.layers.0.self_attn.q_proj.weight          (12288, 5120)
mtp.layers.0.self_attn.k_proj.weight          (1024,  5120)
mtp.layers.0.self_attn.v_proj.weight          (1024,  5120)
mtp.layers.0.self_attn.o_proj.weight          (5120,  6144)
mtp.layers.0.self_attn.q_norm.weight          (256,)
mtp.layers.0.self_attn.k_norm.weight          (256,)
mtp.layers.0.mlp.{gate,up,down}_proj.weight   (17408, 5120)
mtp.layers.0.input_layernorm.weight           (5120,)
mtp.layers.0.post_attention_layernorm.weight  (5120,)
mtp.norm.weight                               (5120,)
mtp.pre_fc_norm_{embedding,hidden}.weight     (5120,)
```
Abliteration targets the **main model's refusal direction**; the MTP subnetwork rides
along with whatever distribution shift the residual stream undergoes, but its own weights
are not surgically modified.

---

## Benchmark

Driver: `parallel-mtp-voting/adaptive_mtp.py` (sibling project; only its logs are in this
corpus — see [`src/logs/`](src/logs/)). Config: `threshold=0.8, max_chain=2,
max_tokens=128`. 1 warmup + 3 timed runs. Prompt: the standard transformer-explanation
prompt from `revalidate_adaptive.py` (36 prompt tokens).

| Model | tok/s (mean of 3) | Tokens/step | Avg chain | Rollbacks |
|---|---|---|---|---|
| Memory anchor (vanilla, MLX 0.30.6, historical) | 52.0 | 2.51 | — | 0 |
| **Vanilla (today, current MLX)** | **51.1** | 2.29 | 1.27 | 0 / 56 |
| **Huihui abliterated (today, current MLX)** | **49.5** | 2.17 | 1.17 | 1 / 59 |

Per-run (rock steady): Huihui 49.4 / 49.6 / 49.7 / 49.5; vanilla 50.7 / 51.1 / 51.2 / 51.1.
Both runs: prefill 36 tokens ~286 ms, generate 128 tokens ~2500–2590 ms. Draft
accepted 43, bonus 56 (vanilla) / 59 (Huihui).

### Interpretation
- **MLX version drift: 52.0 → 51.1**, ~2% on vanilla — noise-level. The 52.0 anchor from
  the memory index is verified.
- **Abliteration cost: 51.1 → 49.5**, ~3% on the same script, same MLX, same config. The
  mechanism is visible in the stats: **avg chain length drops 1.27 → 1.17** because the
  MTP head's peak-confidence distribution is slightly noisier on the shifted residual
  stream. With threshold=0.8, more chains terminate after 1 draft instead of 2, so
  tokens-per-step falls 2.29 → 2.17. One rollback appeared in Huihui vs zero in vanilla.
- A threshold sweep on Huihui (0.7, 0.75) may recover most of the gap — the MTP still
  produces useful drafts, it just falls below 0.8 confidence more often. Not done in this
  log.

---

## Side finding — `ThreadLocalStream` fix for `mlx_lm.server`

Setting up a daemon against the fork's `mlx_lm.server` (`~/mlx-fork/mlx-lm/`), the first
completion crashed:
```
RuntimeError: There is no Stream(gpu, 0) in current thread.
  File "mlx-fork/mlx-lm/mlx_lm/generate.py", line 1090, in _process_prompts
    mx.eval([c.state for c in prompt_cache])
```
Cause: the fork's `generate.py` creates `generation_stream = mx.new_stream(mx.default_device())`
at **module-import time on the main thread**. The HTTP server runs generation on a worker
thread, and streams from `mx.new_stream()` are bound to the creating thread. `mx.stream(...)`
+ `mx.eval(...)` in the worker thread fails because that thread has no device-0 default
stream.

**Fix (1 line):**
```python
# before
generation_stream = mx.new_stream(mx.default_device())
# after
generation_stream = mx.ThreadLocalStream(mx.default_device())
```
`mx.ThreadLocalStream` is documented as "unique per thread" and lazily materializes
per-thread stream state on first access. After the fix, `mlx_lm.server` from the fork
handles chat completions cleanly.

---

## Files touched (outside the source repos)

*The original work used these paths on the author's M4 Max — illustrative of the setup, not a layout you must reproduce; the portable artifacts are in [`src/`](src/).*

- `~/models/Huihui-Qwen3.5-27B-abliterated-mlx-4bit/` — converted base (14 GB)
- `~/mlx-fork/mtp_weights_huihui.safetensors` — extracted MTP head (265 MB)
- `~/mlx-fork/mtp_weights.safetensors` → symlink, toggles vanilla/huihui
- `~/mlx-fork/mlx-lm/mlx_lm/generate.py` — ThreadLocalStream fix
- Model paths repointed in `~/mlx-fork/`: `profile_gap_v2.py`,
  `mtp_speculative_decode.py`, `mtp_v5_bench.py`, `bench_v7.py`, `mtp_batch.py`

Raw logs (copied into this corpus): [`src/logs/adaptive_mtp_vanilla.log`](src/logs/adaptive_mtp_vanilla.log)
(51.1 tok/s), [`src/logs/adaptive_mtp_huihui.log`](src/logs/adaptive_mtp_huihui.log)
(49.5 tok/s).

---

## Related

- [`mtp-self-speculative-mlx.md`](mtp-self-speculative-mlx.md) — the MTP head + extraction (`extract_mtp_huihui.py` mirrors `extract_weights.py`) this benchmark runs.
- [`README.md`](README.md) — the 03-MLX entry page; the 51.1 (vanilla) / 49.5 (Huihui) headline rows.
- [`the-journey.md`](the-journey.md) — Phase 7, the adaptive-chain win that both variants use.
- [`bandwidth-and-dispatch.md`](bandwidth-and-dispatch.md) — why throughput is bandwidth-bound (abliteration changes the residual stream, not the weight budget).
- Cross-domain: [`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md) (the 15 MTP tensors preserved by abliteration), [`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md).

---

*Source: `qwen-inference-lab/docs/HUIHUI_ABLITERATED.md` (canonical; the
`qwen-ops/research/findings/HUIHUI_ABLITERATED.md` copy is byte-identical and collapsed),
plus `benchmarks/extract_mtp_huihui.py` and the two adaptive logs.*
