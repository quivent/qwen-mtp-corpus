# 05 — Theory & Designs: spec-decode on hybrid models, rollback, MTP-head designs

> The platform-independent **"ideas"** domain. Why speculative decoding is hard on a
> hybrid attention/recurrent model, the generalizable rollback technique that makes it
> tractable, and the (mostly design-stage) MTP-head training directions. Where 01–04 own
> *what each platform did*, this folder owns the architecture-independent *why* — the
> principles every implementation had to obey, written once.

This domain is the **canonical home** for the general speculative-decoding math
(irreversibility of recurrent state, the O(1)/token ceiling, the four rollback strategies)
and for the **per-position-heads correction** — the resolved truth that the MLX 1.68×/1.73×
came from *chained* single-head application + confidence gating, **not** from trained
per-position heads (that design remains an unbuilt alternative).

## Reading path

1. [`speculative-decoding-on-hybrid-models.md`](speculative-decoding-on-hybrid-models.md) —
   **start here.** The core theory: why a 3:1 DeltaNet/attention hybrid breaks the "rollback
   is cheap" assumption, and the principles every platform obeyed.
2. [`recurrent-rollback-technique.md`](recurrent-rollback-technique.md) — the generalizable
   **split-recurrence rollback** (batch the matmuls, split only the recurrence; saved-ref
   restore, no redo). Delta-rule math, cost analysis, applicability across Mamba/RWKV/etc.
3. Then the **MTP-head design / training** docs (read in any order):
   - [`per-position-heads-design.md`](per-position-heads-design.md) — the DeepSeek-V3-style
     N-head design **and** the resolved truth about what actually shipped; the Phase-0
     kill-gate.
   - [`cascade-mtp-training.md`](cascade-mtp-training.md) — depth-specific cascade-corrective
     heads (CE + MSE) to stop chained-drafting acceptance from collapsing with depth.
   - [`training-mtp-heads.md`](training-mtp-heads.md) — general distill-from-logits head
     training (freeze main, no labels), cost estimates, the two reference scripts.
4. [`research-frontiers.md`](research-frontiers.md) — the open directions, the O(1)/token
   ceiling argument, and the confirmed dead-ends (don't re-explore).

## Subfolders

- [`recurrent-rollback-src/`](recurrent-rollback-src/) — the copied MLX reference
  implementation of split-recurrence rollback (`src/`, `examples/`). See its `README.md`.
- [`training/`](training/) — the two **design-only** training scripts
  (`build_training_data.py`, `train_per_position_heads.py`; both guard with a NOT-RUNNING
  exit). See its `README.md`.

## Cross-domain

The host-side mechanics are platform-specific: llama.cpp in
[`../02-LLAMACPP/`](../02-LLAMACPP/), MLX in [`../03-MLX/`](../03-MLX/), vLLM/modal
self-speculative in [`../04-VLLM-GPU/`](../04-VLLM-GPU/). Authoritative architecture and
tensor shapes are [`../01-ARCHITECTURE/`](../01-ARCHITECTURE/); measured numbers are in
[`../00-OVERVIEW/results.md`](../00-OVERVIEW/results.md) and terms in
[`../00-OVERVIEW/glossary.md`](../00-OVERVIEW/glossary.md).
