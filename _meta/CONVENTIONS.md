# MTP Corpus — Build Conventions (read this first)

This file is the shared contract for every agent building the `MTP/` corpus. Read it
fully before writing anything.

## What we are building

`MTP/` is the **fully reorganized, distilled, lossless, non-confusing** corpus
synthesized from 10 source repositories about **Multi-Token Prediction (MTP)
speculative decoding for Qwen3.5-27B** across llama.cpp, MLX, and vLLM.

Source repos live as siblings of `MTP/` under `/home/ubuntu/qwen27/`:

| Repo | Role |
|---|---|
| `llama-mtp/` | The actual llama.cpp **fork** (383M, mostly upstream). MTP-specific source only: `src/models/qwen35.cpp`, `src/models/qwen3next.cpp`, `common/speculative-mtp.{cpp,h}`, `examples/mtp-speculative/`, plus MTP hooks in `src/llama-arch.*`, `llama-model.cpp`, `llama-hparams.h`, `llama-graph.h`. |
| `qwen-mtp-llamacpp/` | llama.cpp MTP **infrastructure** patches (11, ordered) + README narrative |
| `qwen-mtp-optimizations/` | llama.cpp **optimization variants** (9 patches) + README |
| `qwen-mtp-tensors/` | HF→GGUF **tensor mapping** / converter work + deep-dive doc |
| `qwen-mtp-research/` | Research notes, the-bug story, the-recipe, per-position-heads design |
| `mlx-qwen-mtp/` | **MLX** (Apple Silicon) implementation + README |
| `qwen-inference-lab/` | M4 Max inference **optimization journey** (TIMELINE, bandwidth, dispatch) |
| `recurrent-rollback/` | The **split-recurrence rollback** technique (generalizable) + TECHNIQUE.md |
| `modal-mtp/` | **Self-speculative** DeltaNet-skip drafting + precision-divergence saga |
| `qwen-ops/` | An EARLIER partial consolidation of the above. Contains DUPLICATES + extra `research/designs/*`, `vllm/`, `quantization/`, `deploy/`, `training/`, `cmd/` (Go). Treat as a *source*, not the target. |

> **Known duplication:** 13 docs in `qwen-ops/research/*` are byte-identical copies
> of docs in the satellite repos (TIMELINE, BANDWIDTH_ANALYSIS, the-recipe,
> tensor-layout, per-position-heads, integration-plan, mlx-reference,
> DISPATCH_BARRIER_PROFILING, HUIHUI_ABLITERATED, RESEARCH_FRONTIERS,
> modal-mtp-design). Collapse each to ONE canonical home in `MTP/`.

## Target structure (domains → owners)

```
MTP/
├── README.md                 # master entry + navigation (orchestrator)
├── _meta/CONVENTIONS.md       # this file
├── 00-OVERVIEW/               # SYNTHESIS layer (built in later iterations)
│   ├── the-big-picture.md     #   the whole arc as one narrative
│   ├── results.md             #   unified benchmark table, all platforms
│   ├── glossary.md            #   MTP, DeltaNet, eh_proj, split-recurrence, …
│   └── corpus-map.md          #   provenance / where everything came from
├── 01-ARCHITECTURE/           # AGENT A  — Qwen3.5 arch + MTP head + tensors
├── 02-LLAMACPP/               # AGENT B  — infra patches, variants, the bug, the recipe, source/, patches/
├── 03-MLX/                    # AGENT C  — kernel fusion, split-recurrence, journey, src/
├── 04-VLLM-GPU/               # AGENT D  — modal self-spec, precision saga, quant, deploy, patches/
├── 05-THEORY-AND-DESIGNS/     # AGENT E  — spec-decode theory, rollback technique, per-position heads, designs/
└── 99-LEDGER/                 # provenance proof + open-questions (synthesis layer)
```

## Rules for every agent

1. **Lossless.** Every unique fact, number, recipe, env var, tensor name, dead-end,
   and code artifact in your domain must survive into `MTP/`. When in doubt, keep it.
   Losing a hard-won detail is the worst failure mode.
2. **Self-contained artifacts.** COPY (do not just reference) the unique small
   code/patches/scripts for your domain into your domain folder. Do NOT copy the
   383M upstream llama.cpp tree — reference it by path. DO copy the MTP-specific
   source files from the fork.
3. **De-duplicate ruthlessly.** If the same content lives in N places, write it ONCE
   in its canonical home and note the collapse. "Not confusing" = one source of truth.
4. **Layered for two readers.** Top of each doc = narrative/findings a researcher can
   read straight through. Lower sections = exact reproducible ops (commands, env vars,
   tensor maps, file refs) for an engineer rebuilding the result.
5. **Resolve, don't flatten, evolving threads.** Some understanding CHANGED over time
   (e.g. the FP16 precision-divergence story; "MLX uses trained per-position heads"
   later corrected to "single head, chained recurrence"). Present the *current/resolved*
   understanding clearly up top, then preserve the journey ("we first thought X,
   then found Y") so nothing is lost and nobody is misled.
6. **Cite provenance.** When you distill a doc, note its source path in a footer so the
   provenance ledger can be assembled. Use the per-domain inventory file (below).
7. **Numbers are sacred.** Copy benchmark numbers, hardware specs, and tensor shapes
   EXACTLY. Do not round, average, or "clean up" measured values.
8. **Write frequently.** Save partial work as you go; don't hold everything to the end.

## Per-domain inventory (lossless ledger input)

Each agent writes ONE inventory file: `MTP/99-LEDGER/inventory-<NN>-<domain>.md`
containing a table: every source file you consumed → which MTP/ file(s) absorbed it →
status (distilled / copied / merged-into-X / duplicate-of-Y / deferred). Also a short
"Contradictions & evolving threads" section listing anything that disagrees across
sources. The orchestrator merges these into `99-LEDGER/provenance.md`.

## Hardware reference (use exact strings)

- **M4 Max**: Apple M4 Max, 16-core GPU, 128 GB unified, 546 GB/s bandwidth. Model: Qwen3.5-27B-4bit (13.7 GB weights). Theoretical min 25.1 ms/tok (39.8 tok/s).
- **GH200**: NVIDIA GH200 480GB. vLLM. BF16 and W4A16.
- **RTX 5090**: NixOS, GPTQ W4A16.

## Headline results (verify against sources; do not invent)

| Platform | HW | Best tok/s | vs baseline | Method |
|---|---|---|---|---|
| llama.cpp | M4 Max | 13.98 | 1.99× over K=1 vanilla (0.78× of plain decode 17.90) | Chained MTP + confidence gating (`MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`) |
| MLX | M4 Max | 51.1 | 1.73× over stock 29.5 | Adaptive MTP chain + batch verify |
| vLLM | GH200 480GB | 1030 (batch=8) / 186 (batch=1) | 5.54× batch / baseline | Stock MTP spec=7 |
| vLLM | RTX 5090 | 151 | — | GPTQ W4A16 + MTP=5 |
