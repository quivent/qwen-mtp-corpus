# Per-position-heads stub patches (DESIGN ONLY — not the working series)

Three **design-stub** diffs for the *alternative* DeepSeek-V3-style **per-position
trained heads** path — NOT part of the working 16-patch infrastructure series. They
sketch the loader/graph/inference changes needed to support **multiple** MTP heads
(`head_0` predicts +1, `head_1` predicts +2, …) instead of the single chained head the
shipped port uses. Referenced by [`../../integration-plan.md`](../../integration-plan.md)
Phase 4. The deep design lives in `MTP/05-THEORY-AND-DESIGNS/per-position-heads-design.md`.

| File | Touches | What it stubs |
|---|---|---|
| `loader-stub.patch` | `src/llama-model.cpp` (~line 5523) | Load N independent NextN heads — each `i in [n_layer-N, n_layer)` holds an independent `head_k`; `nextn_predict_layers` was always 1, now ≥1. |
| `graph-stub.patch` | `src/models/qwen35.cpp` (~line 17) | MTP draft graph that runs only the NextN head over a single seed + previous hidden, generalized to select head `k`. |
| `inference-stub.patch` | `src/llama-context.cpp` (`llama_mtp_draft`, ~line 3841) | Drafting entry point that dispatches to head `k` per draft position instead of chaining the single head. |

These are **incomplete stubs** (no commit metadata, just diffs). Copied here for
losslessness because they are unique small llama.cpp artifacts cited by this domain's
integration plan; the working series is in [`../infrastructure/`](../infrastructure/).

---
*Provenance: copied verbatim from `qwen-mtp-research/patches/`.*
