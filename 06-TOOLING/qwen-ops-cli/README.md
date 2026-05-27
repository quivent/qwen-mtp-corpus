# qwen-ops-cli — Go sources (verbatim copy)

These are the **unmodified** source files of the `qwen-ops` Cobra CLI, copied here
so the corpus is self-contained and the tool stays buildable.

## Origin

| Field | Value |
|---|---|
| Source repo | `/home/ubuntu/qwen27/qwen-ops/` (the earlier partial consolidation repo) |
| Go module | `github.com/joshkornreich/qwen-ops` |
| Go version | `go 1.25.0` |
| Only dependency | `github.com/spf13/cobra v1.9.1` (+ transitive `pflag`, `mousetrap`) |
| Files copied | `cmd/{main,serve,mtp,patches,status,download}.go`, `go.mod`, `go.sum` |

No source agent in iteration 1 consumed this Go CLI; it was the single biggest gap
in the corpus. It is the **operational umbrella** that drives the artifacts captured
in the other domains:

- the llama.cpp patches in `../../02-LLAMACPP/patches/`
- the vLLM patches in `../../04-VLLM-GPU/patches/`
- the GH200 / RTX 5090 deploy scripts in `../../04-VLLM-GPU/deploy/`

## Layout preserved

```
qwen-ops-cli/
├── go.mod
├── go.sum
└── cmd/
    ├── main.go       # root cobra command + subcommand registration
    ├── serve.go      # `qwen-ops serve`     — launch llama-server (GH200 optimal flags)
    ├── mtp.go        # `qwen-ops mtp`        — apply llama.cpp patches + benchmark
    ├── patches.go    # `qwen-ops patches`    — list/apply vLLM + llama.cpp patches
    ├── status.go     # `qwen-ops status`     — inspect build/model/server/GPU/patch state
    └── download.go   # `qwen-ops download`   — fetch the GGUF model from HuggingFace
```

## Build

```bash
cd qwen-ops-cli
go build -o qwen-ops ./cmd
./qwen-ops --help
```

> Note: at runtime the CLI hard-codes paths under `~/qwen-ops/` (e.g.
> `~/qwen-ops/llamacpp/infrastructure/`, `~/qwen-ops/vllm/patches/`) as the patch
> source directories. To use the patches captured in this corpus, either symlink
> those locations to the corpus folders or copy the patches under `~/qwen-ops/`.
> See `../README.md` for the full path-assumption map.

## NOT copied

- The compiled binary `qwen-ops/qwen-ops` (a 3.9 MB Mach-O arm64 executable — a
  build artifact, regenerable from these sources).
- The non-`cmd/` satellite trees inside the source `qwen-ops/` repo
  (`llamacpp/`, `vllm/`, `mlx/`, `deploy/`, `quantization/`, `research/`,
  `training/`, `validation/`) — those are duplicates of the satellite repos and
  are captured (de-duplicated) in domains `01`–`05`. See
  `../../99-LEDGER/coverage-report.md`.
