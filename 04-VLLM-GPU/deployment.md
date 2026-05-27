# Deployment Runbook — GH200 and RTX 5090 (NixOS)

This runbook gets Qwen3.5-27B serving with MTP speculative decoding on two very
different targets. **Part 1 (GH200 480GB)** is the datacenter path — a rented
Lambda/Hopper box with 96 GB HBM3e where the goal is maximum batch throughput
(~190 tok/s single, ~500 batch=4, up to 1030 at batch=8); you start fresh each
session, so the steps are a clean venv install. **Part 2 (RTX 5090 on NixOS)**
is a single persistent production box ("captain") where the goal is a
self-healing, always-on OpenAI-compatible endpoint behind a Cloudflare tunnel;
the steps lean on systemd services and NixOS-specific wiring. Follow whichever
part matches your hardware — by the end you have a live `qwen3.5-27b` endpoint
on port 8001 with verified MTP acceptance.

Distilled install/serve runbooks. Exact commands, env vars, and gotchas are
preserved. Full original scripts are copied under [`deploy/`](deploy/).

---

## Part 1 — GH200 480GB

vLLM 0.19.0 in a venv at `/opt/vllm-env`, Qwen3.5-27B AWQ W4A16 (vision
stripped), MTP spec decode. **Known-good result (Apr 2026):** single-request
**~192 tok/s** (185–202), TTFT p90 ~80 ms, MTP acceptance **55.5%** overall
(per-position 85/68/54/40/31%).

### Fastest path: `deploy/gh200/deploy.sh`

```bash
./deploy.sh all                       # env + pull + prep (~8 min)
nohup ./deploy.sh launch > vllm.log 2>&1 &   # first launch ~3-5 min JIT, cached after
./deploy.sh smoke                     # curl sanity + MTP acceptance check
```

Env overrides (defaults): `VENV=/opt/vllm-env`,
`MODEL_DIR=/opt/models/Qwen3.5-27B-AWQ`,
`HF_REPO=j-a-a-a-y/Qwen3.5-27B-AWQ-4bit-textonly`, `PORT=8001`,
`NUM_SPEC_TOKENS=5`, `HF_TOKEN`.

The launch command it runs:
```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name qwen3.5-27b \
  --host 0.0.0.0 --port 8001 \
  --dtype float16 --max-model-len 4096 \
  --max-num-seqs 4 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.95 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' \
  --performance-mode interactivity --enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
```
(`PYTORCH_ALLOC_CONF=expandable_segments:True`, `HF_HUB_ENABLE_HF_TRANSFER=1`,
and `PATH=$VENV/bin:$PATH` so flashinfer finds `ninja`.)

### The GH200 gotchas (these cause most 3× misses)

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | `pip install vllm` pulls `torch-2.10.0+cpu`; `torch.cuda.is_available()==False` | aarch64 (Grace CPU) resolves CPU-only torch | `pip install --force-reinstall torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128` |
| 2 | Engine exits during `profile_cudagraph_memory`: `FileNotFoundError: 'ninja'` | flashinfer JIT needs the `ninja` **binary**, not just the pip package | `apt-get install -y ninja-build` (or venv bin on PATH) |
| 3 | Silent 3–5 min on first launch after "torch.compile took X s" | flashinfer JIT-compiles ~32 GDN prefill kernels | Expected, one-time, cached in `~/.cache/flashinfer/`. Skip with `--gdn-prefill-backend triton`. |
| 4 | `RuntimeError: Cannot find any model weights` on `-textonly` repos | Repo ships single `model.safetensors` but stale 4-shard `index.json` | Rewrite the index (snippet below) |
| 5 | `/metrics` acceptance ≈ 0% → tok/s ~1/3 expected | On `...-retrained-mtp` model — abandoned draft head, broken weight map | Use `...-AWQ-4bit-textonly` instead |
| 6 | Vision-strip script erases the freshly-written index | Copy-loop didn't exclude `model.safetensors.index.json` | Keep `.index.json` in the skip list |
| 7 | Tempted to apply INT8 embedding patch on GH200 | It's only for low-VRAM GPUs | Skip it — 96 GB has headroom |

**Rewrite stale index** (gotcha #4):
```bash
python3 -c "
import json, os
from safetensors import safe_open
p = '/opt/models/Qwen3.5-27B-AWQ/model.safetensors'
keys = list(safe_open(p, framework='pt').keys())
idx = {'metadata': {'total_size': os.path.getsize(p)}, 'weight_map': {k: 'model.safetensors' for k in keys}}
json.dump(idx, open('/opt/models/Qwen3.5-27B-AWQ/model.safetensors.index.json', 'w'), indent=2)
"
```

**Disable thinking** in `chat_template.jinja`:
```bash
# prepend if not present:
{%- if enable_thinking is not defined %}{%- set enable_thinking = false %}{%- endif %}
```

### Manual install (when not using deploy.sh)

`deploy/gh200/07-FRESH-INSTALL.md` and `08-GH200-AGENT-INSTALL.md` give the
copy-paste 9-step version. Key steps: venv + `pip install vllm==0.19.0` →
`snapshot_download` model → strip vision (saves 0.92 GB) → disable thinking →
apply patches (`./apply.sh eagle qwen3_next rollback`) → optional INT8 embedding
patch (low-VRAM only) → launch → verify → systemd service.

`08-GH200-AGENT-INSTALL.md` uses `cyankiwi/Qwen3.5-27B-AWQ-4bit` and applies the
**recurrent-rollback** patch (O(1) GDN state restore on MTP rejection;
`max-model-len 8192`, `max-num-seqs 8`). Apply `rollback` **last** — it patches
two files (`gdn_linear_attn.py` + `qwen3_5.py`) that other patches also touch.

### Recurrent rollback (what it buys you on GH200)

MTP proposes 5 draft tokens, verifies in one batched forward. Qwen3.5's 48 GDN
layers have a **non-invertible** recurrent update
`S_{t+1} = g_t·S_t + beta_t·k_t·(v_t − k_t^T·S_t)`. On rejection at position K,
the GDN state is corrupted by tokens K+1..4. The patch saves O(1) `clone()`
checkpoints at each draft position during verify; on rejection it restores all
48 layers' conv_state + ssm_state in **~0.15 ms** (one `copy_` per layer) instead
of rerunning the forward. Memory: **~893 MB for MTP=5** (48 layers × 6 positions
× 3.1 MB) — negligible on 96 GB.

### GH200 vs consumer-GPU settings

| Setting | RTX 5090 (32 GB) | GH200 (96 GB) |
|---|---|---|
| `gpu-memory-utilization` | 0.97–0.98 | 0.95 |
| `max-model-len` | 1024–4096 | 8192+ |
| `max-num-seqs` | 4 | 8 |
| Rollback memory (893 MB) | tight | negligible |
| Expected tok/s (single / batch=4) | ~140 / ~450 | ~190 / ~500 |

Per-GPU expectations (`07-FRESH-INSTALL.md`): RTX 5090 ~140 single / ~450 batch4
/ ~33K KV; RTX 4090 ~80–100 / ~250 / ~15K; A6000 ~100–120 / ~350 / ~80K; GH200
~190 / ~500 / ~200K.

---

## Part 2 — RTX 5090 on NixOS (host "captain")

A persistent production box. SSH `ssh -p 2227 root@185.193.125.244`, local API
`http://localhost:8001`, model name `qwen3.5-27b`, OpenAI-compatible. Public URL
changes on reboot (Cloudflare tunnel) — find it via
`journalctl -u cloudflared-tunnel | grep trycloudflare.com | tail -1`.

### Three persistent systemd services

1. **vllm.service** — the model server (starts on boot, restarts on crash).
2. **vllm-watchdog.service** — health-checks every 30 s; if vLLM is unresponsive
   for 3 min (6 fails), kills GPU procs and `systemctl restart vllm`
   (`deploy/rtx5090/vllm-watchdog.sh`: 30 s interval, 6-fail threshold, 180 s
   cooldown, 120 s startup grace).
3. **cloudflared-tunnel.service** — public internet access.

### Current config (`01-SERVER-STATUS.md`)

| Setting | Value |
|---|---|
| Model | `/opt/models/Qwen3.5-27B-AWQ-textonly` (cyankiwi AWQ, 19.1 GB) |
| Quant | compressed-tensors (auto, Marlin kernel) |
| MTP spec tokens | 5 |
| Perf mode | interactivity |
| Prefix caching | enabled |
| Max context | 4096 |
| Max concurrent | 4 |
| GPU mem util | 97% |
| KV cache | 26,112 tokens (INT8 embeddings save ~1.27 GB) |
| Thinking | disabled |

Performance: ~140 tok/s @256, ~143 @512, ~450 batch=4, MTP acceptance ~50–53%.

### Serve script (`deploy/rtx5090/vllm-serve.sh`)

Presets (switch with one command):
```bash
./vllm-serve.sh gptq      # Huihui abliterated GPTQ W4A16 — best single (151 tok/s, gptq_marlin)
./vllm-serve.sh awq       # cyankiwi AWQ textonly — best batch (403 tok/s)
./vllm-serve.sh --status  # service + GPU + health
./vllm-serve.sh --stop
```
Flags: `--model --quant --mtp N --no-mtp --port --ctx --seqs --gpu-mem --dtype
--perf --eager --bench --extra`. Defaults: ctx=1024, seqs=4, gpu-mem=0.97,
dtype=float16, MTP=5, perf=interactivity. It exports the NixOS env (see below),
stops any running instance, launches, polls `/health` for up to 90×5 s.

### NixOS-specific gotchas (`05-NIXOS-GUIDE.md`)

All config lives in `/etc/nixos/configuration.nix`; after edits run
`nixos-rebuild switch`. The vLLM service is `systemd.services.vllm`
(`nixos-captain-configuration.nix`).

1. **`/etc` is read-only** — systemd services must be defined in
   `configuration.nix`, not created directly.
2. **`/usr/bin`, `/usr/local/bin` don't exist** — binaries in `/nix/store/` and
   `/run/current-system/sw/bin/`.
3. **Dynamic linking is broken for pip-installed binaries** — `ptxas`, `ninja`
   need wrappers:
   - `.../triton/backends/nvidia/bin/ptxas` (wrapper → `.real`)
   - `.../triton/backends/nvidia/bin/ptxas-blackwell` (wrapper → `.real`)
   - `/opt/vllm-env/bin/ninja` (wrapper → `.real`)
4. **Python is 3.13** (NixOS) — some packages expect 3.10–3.12.
5. **`/bin/bash` doesn't exist** — use `#!/usr/bin/env bash` or
   `#!/run/current-system/sw/bin/bash`.

Required env vars for the vLLM service (set in `configuration.nix`):
```
LD_LIBRARY_PATH=/run/opengl-driver/lib:/nix/store/...-gcc-14.3.0-lib/lib:/nix/store/...-cuda-merged-12.8/lib
CC=/run/current-system/sw/bin/gcc
CPATH=/nix/store/...-python3-3.13.12/include/python3.13:/nix/store/...-cuda-merged-12.8/include
PATH=/opt/vllm-env/bin:/run/current-system/sw/bin:/nix/store/...-cuda-merged-12.8/bin
```

Patches re-applied after each `pip install vllm==0.19.0 --force-reinstall`:
**INT8 embeddings** (saves 1.27 GB) and **eagle** (tree-spec MRoPE crash fix).
The `.patch` files are reference diffs and may not apply cleanly across Python
versions — apply manually if needed.

The NixOS box also keeps a `qwen3.5-27b-q4km.gguf` (16 GB) for llama.cpp and the
GPTQ abliterated model (19.5 GB) on disk; switch via the `vllm-serve.sh` presets.

---

## Related

- [`README.md`](README.md) — the GH200 + RTX 5090 results tables and per-position acceptance.
- [`quantization.md`](quantization.md) — which W4A16 model to serve (GPTQ vs AWQ) + vision-strip / stale-index fixes.
- [`vllm-patches-and-strategies.md`](vllm-patches-and-strategies.md) — the patches `apply.sh` installs (eagle, qwen3_next, rollback, INT8 embedding).
- [`deploy/`](deploy/) — the full original scripts (`deploy.sh`, `vllm-serve.sh`, `vllm-watchdog.sh`, NixOS config) and copy-paste install docs.
- [`../06-TOOLING/README.md`](../06-TOOLING/README.md) — the `qwen-ops` CLI that automates download/patch/serve on GH200.
- [`../99-LEDGER/open-questions.md`](../99-LEDGER/open-questions.md) — the GH200 roofline-arch mislabel behind the 186-vs-94 tok/s gap.

---
*Provenance: `qwen-ops/deploy/gh200/{deploy.sh,01-SERVER-STATUS.md,07-FRESH-INSTALL.md,08-GH200-AGENT-INSTALL.md}`,
`qwen-ops/deploy/rtx5090/{05-NIXOS-GUIDE.md,nixos-captain-configuration.nix,vllm-serve.sh,vllm-watchdog.sh}`.
Note: `01-SERVER-STATUS.md` describes the RTX 5090 box but lives under `gh200/`
in the source tree — preserved as-is.*
