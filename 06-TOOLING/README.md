# 06 — Tooling: the `qwen-ops` Go CLI

The operational umbrella that ties the whole corpus together. `qwen-ops` is a
Cobra-based Go CLI (module `github.com/joshkornreich/qwen-ops`) that **downloads
the model, applies the MTP patches, launches optimized inference, benchmarks it,
and reports machine state** — the single command-line entry point for the entire
Qwen3.x-27B MTP speculative-decoding stack on a GH200.

The Go sources are copied verbatim into [`qwen-ops-cli/`](qwen-ops-cli/) so the
tool stays buildable; this README distills *what it does*. The underlying
artifacts it drives live in sibling domains:

- llama.cpp patches it applies → [`../02-LLAMACPP/patches/`](../02-LLAMACPP/patches/)
- vLLM patches it applies → [`../04-VLLM-GPU/patches/`](../04-VLLM-GPU/patches/)
- deploy scripts that overlap its job → [`../04-VLLM-GPU/deploy/`](../04-VLLM-GPU/deploy/)

---

## Narrative — what the tool is for

The corpus is a pile of patches, scripts, and findings spread across ten repos.
`qwen-ops` is the layer that makes them *operational on one box*. The intended
workflow on a fresh GH200 is:

```
qwen-ops download           # 1. pull Qwen3.6-27B Q4_K_M GGUF from HuggingFace
qwen-ops mtp apply          # 2. apply the 16 infra patches to a llama.cpp checkout
qwen-ops mtp apply --optimize   #    (optionally also the 9 optimization patches)
qwen-ops serve --spec       # 3. launch llama-server with MTP speculative decoding
qwen-ops mtp bench          # 4. measure throughput
qwen-ops status             # 5. inspect what's built / running / on the GPU
```

It is deliberately **opinionated**: `serve` bakes in the "proven GH200 optimal
flags" rather than exposing every llama-server knob, and `download` defaults to a
specific repo + quant. It is also **GH200-first** — the defaults (34 threads,
`-ngl 99`, `--mlock --no-mmap`, `q8_0`/`q4_0` KV cache, 8 parallel slots) assume a
big NVIDIA box with the whole model resident in GPU memory, but `status` still
degrades gracefully to Apple-Silicon detection so it is usable on an M4 Max too.

### Two patch managers, one tool

There are *two* patch subsystems because the corpus targets two engines:

- **`qwen-ops mtp apply`** drives the **llama.cpp** patch series
  (`~/qwen-ops/llamacpp/infrastructure/` then optionally
  `~/qwen-ops/llamacpp/optimizations/`), applied in sorted order via `git apply`.
- **`qwen-ops patches apply`** drives the **vLLM** patch set
  (`~/qwen-ops/vllm/patches/` + `~/qwen-ops/vllm/optimizations/`).

Both share the same apply strategy: `git apply --check` first, fall back to
`git apply --3way` on conflict, otherwise plain `git apply`. `qwen-ops patches
list` enumerates all four directories without applying anything.

### What it does NOT do

It is a *driver*, not a library: it shells out to `git`, `llama-server`,
`llama-bench`/`llama-cli`, `huggingface-cli`/`hf`, and `nvidia-smi`. It contains
no MTP inference logic of its own — all the intelligence is in the patches and
scripts it orchestrates (captured in domains 02 and 04). There is no `train`,
`convert`, or `quantize` subcommand; those workflows live as standalone scripts
in `../01-ARCHITECTURE/`, `../04-VLLM-GPU/quantization/`, and
`../05-THEORY-AND-DESIGNS/training/`.

---

## Path assumptions (important for reuse)

The CLI hard-codes a `~/qwen-ops/` layout for patch sources and `~/llama.cpp`,
`~/vllm`, `~/models` for targets. To drive the corpus copies, point these at the
MTP folders:

| CLI expects (default) | Corpus equivalent | Used by |
|---|---|---|
| `~/qwen-ops/llamacpp/infrastructure/` | `../02-LLAMACPP/patches/infrastructure/` | `mtp apply` |
| `~/qwen-ops/llamacpp/optimizations/` | `../02-LLAMACPP/patches/optimizations/` | `mtp apply --optimize` |
| `~/qwen-ops/vllm/patches/` | `../04-VLLM-GPU/patches/` | `patches apply`, `patches list` |
| `~/qwen-ops/vllm/optimizations/` | `../04-VLLM-GPU/optimizations/` (`.py`, not `.diff`) | `patches apply`, `patches list` |
| `~/llama.cpp` (with `CMakeLists.txt`) | a fresh upstream llama.cpp checkout | `mtp apply`, `serve`, `bench`, `status` |
| `~/vllm` (with `setup.py`/`pyproject.toml`) | a fresh vLLM checkout | `patches apply` |
| `~/models/Qwen3.6-27B-Q4_K_M.gguf` | downloaded model | `serve`, `bench`, `status` |

> Quirk: `patches apply` and `patches list` scan `~/qwen-ops/vllm/optimizations/`
> for `*.patch`/`*.diff`, but in the corpus that folder holds Python strategy
> files (`.py`), not diffs — so the vLLM "optimizations" there are applied as
> source files by the deploy/apply scripts, not by `git apply`. The CLI will
> simply find zero diffs there.

---

## Command / flag reference

### `qwen-ops serve` — launch optimized inference
Launches `llama-server` with the proven GH200-optimal flag set for Qwen3.6-27B.

**Fixed (always-on) flags:** `-ngl 99 -fa on --mlock --no-mmap -ctk q8_0 -ctv q4_0
--cont-batching`. With `--spec`, it additionally passes `--spec-type draft-mtp
--spec-draft-n-max <spec-n>`.

**Binary resolution order:** `--llama-server` path → `PATH` (`llama-server`) →
`~/llama.cpp/build/bin/llama-server` → `~/llama.cpp/llama-server` →
`/usr/local/bin/llama-server`.

| Flag | Default | Meaning |
|---|---|---|
| `-m, --model` | `~/models/Qwen3.6-27B-Q4_K_M.gguf` | GGUF model path (errors with "Run: qwen-ops download" if missing) |
| `--port` | `8080` | server port |
| `--host` | `0.0.0.0` | bind host |
| `-c, --ctx` | `8192` | context length |
| `-b, --batch` | `2048` | batch size |
| `--ubatch` | `2048` | micro-batch size |
| `-t, --threads` | `34` | CPU threads (GH200 Grace core count assumption) |
| `--parallel` | `8` | parallel decode slots (continuous batching) |
| `--spec` | `false` | enable MTP speculative decoding (`--spec-type draft-mtp`) |
| `--spec-n` | `3` | max speculative draft tokens (`--spec-draft-n-max`) |
| `--llama-server` | (auto) | explicit binary path |
| `--dry-run` | `false` | print the assembled command and exit (no launch) |

### `qwen-ops mtp` — llama.cpp patch + benchmark
Subcommands: `apply`, `bench`.

**`mtp apply`** — applies infrastructure patches from
`~/qwen-ops/llamacpp/infrastructure/` (sorted), then with `--optimize` also the
optimization patches. Requires `~/llama.cpp/CMakeLists.txt` to exist. Apply
strategy: `git apply --check` → fall back to `git apply --3way` on failure →
else plain `git apply`. Prints a rebuild hint:
`cmake -B build -DGGML_CUDA=ON && cmake --build build -j`.

| Flag | Default | Meaning |
|---|---|---|
| `--llama-cpp-dir` | `~/llama.cpp` | target checkout (must contain `CMakeLists.txt`) |
| `--optimize` | `false` | also apply the 9 optimization patches after infra |

**`mtp bench`** — runs `llama-bench` (`-ngl 99 -fa 1 -t 34`) if present; otherwise
falls back to `llama-cli` repeated `--repeat` times with the proven inference
flags (`-ngl 99 -fa on -t 34 --mlock --no-mmap -n <n-tokens> -p <prompt>`).

| Flag | Default | Meaning |
|---|---|---|
| `-m, --model` | `~/models/Qwen3.6-27B-Q4_K_M.gguf` | GGUF model |
| `-p, --prompt` | `"Explain quantum computing in simple terms:"` | benchmark prompt (CLI fallback only) |
| `-n, --n-tokens` | `256` | tokens to generate (CLI fallback only) |
| `--repeat` | `3` | benchmark repetitions (CLI fallback only) |

### `qwen-ops patches` — vLLM / llama.cpp patch management
Subcommands: `list`, `apply`.

**`patches list`** — enumerates patches in all four directories (llama.cpp
infrastructure + optimizations, vLLM patches + optimizations), counting `*.patch`
and `*.diff`. Read-only.

**`patches apply`** — applies vLLM patches (`~/qwen-ops/vllm/patches/` then
`~/qwen-ops/vllm/optimizations/`, sorted, concatenated) to a vLLM checkout.
Requires `setup.py` or `pyproject.toml` in the target. Same `--check`/`--3way`/
plain `git apply` strategy as `mtp apply`.

| Flag | Default | Meaning |
|---|---|---|
| `--vllm-dir` | `~/vllm` | target vLLM checkout |

### `qwen-ops status` — machine state report
Prints five sections, all read-only:

1. **[llama.cpp]** — `~/llama.cpp` presence; whether `llama-server`/`llama-cli`/
   `llama-bench` are built (`build/bin/`); `git log --oneline -1` version; count of
   `git diff --stat` modified files (i.e. how many MTP patches are applied).
2. **[Models]** — lists `*.gguf` in `~/models` with sizes in GB; hints
   `qwen-ops download` if none.
3. **[Server]** — `pgrep -la llama-server` for running PIDs; parses `--port` from
   `ps aux`.
4. **[GPU]** — `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version`;
   falls back to plain `nvidia-smi`; falls back to Apple-Silicon detection via
   `sysctl machdep.cpu.brand_string` + `hw.memsize` (→ unified-memory GB).
5. **[Patches]** — counts `.patch`/`.diff` in the four patch directories.

No flags.

### `qwen-ops download` — fetch the model
Downloads the GGUF via `huggingface-cli` (or `hf`):
`huggingface-cli download <repo> --include '*<quant>*' --local-dir <dir>`.
Skips (and reports size) if a matching `*<quant>*.gguf` already exists.

| Flag | Default | Meaning |
|---|---|---|
| `-d, --dir` | `~/models` | download directory (created if missing) |
| `-q, --quant` | `Q4_K_M` | quantization level → glob `*Q4_K_M*` |
| `--repo` | `unsloth/Qwen3.6-27B-GGUF` | HuggingFace repo |

---

## Notes on the source

- **Model naming drift:** the CLI defaults reference **`Qwen3.6-27B`**
  (`unsloth/Qwen3.6-27B-GGUF`, `Qwen3.6-27B-Q4_K_M.gguf`) and `serve` is described
  as "Qwen3.6". The rest of the corpus targets **Qwen3.5-27B**. This is a naming
  drift in the later `qwen-ops` tool, not a different model family — the patches,
  tensor maps, and architecture it drives are the Qwen3.5/Qwen3-Next MTP work
  documented in domains 01–05. Recorded, not "fixed".
- **`--spec-type draft-mtp` / `--spec-draft-n-max`** are llama-server flags
  introduced by the MTP infrastructure patch series (see
  [`../02-LLAMACPP/infrastructure-patches.md`](../02-LLAMACPP/infrastructure-patches.md)).
  They presuppose `qwen-ops mtp apply` has been run against the llama.cpp checkout.
- The proven `serve`/`bench` flag set is the GH200 production configuration; the
  M4 Max recipe (`MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`) lives in
  [`../02-LLAMACPP/the-recipe.md`](../02-LLAMACPP/the-recipe.md).

---

## Related

- [`qwen-ops-cli/`](qwen-ops-cli/) — the verbatim Go sources this README distills.
- [`../02-LLAMACPP/infrastructure-patches.md`](../02-LLAMACPP/infrastructure-patches.md) — the llama.cpp patch series `mtp apply` drives (`--spec-type draft-mtp`).
- [`../04-VLLM-GPU/vllm-patches-and-strategies.md`](../04-VLLM-GPU/vllm-patches-and-strategies.md) — the vLLM patches `patches apply` drives.
- [`../04-VLLM-GPU/deployment.md`](../04-VLLM-GPU/deployment.md) — the deploy scripts that overlap this tool's job (GH200 + RTX 5090).
- [`../02-LLAMACPP/the-recipe.md`](../02-LLAMACPP/the-recipe.md) — the M4 Max chained-MTP recipe.
- [`../00-OVERVIEW/glossary.md`](../00-OVERVIEW/glossary.md) — MTP, GGUF, Q4_K_M, speculative decoding terms.
