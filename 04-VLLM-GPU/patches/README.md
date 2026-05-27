# patches/ — vLLM 0.19.0 patches

Copied verbatim from `qwen-ops/vllm/patches/*` and `modal-mtp/patches/*`.

- `apply.sh` — apply/check/revert manager (reverts from the pip wheel, not .bak).
- `eagle.patch`, `qwen3_next.patch`, `qwen3_5-shadow-state.patch`,
  `gdn-shadow-state.patch`, `speculative-dual-mode.patch`,
  `speculative-mtp-tree-compat.patch`, `recurrent-rollback.patch`,
  `modal_mtp.patch`, `speculative-draft-override.patch`,
  `gdn-inhibition-cycle.patch`, `int8-embedding.patch`,
  `gh200-strip-torch-dep.patch`, `llmcompressor-conv1d.patch`
  — from `qwen-ops/vllm/patches/`.
- `modal-01-skip-attention.patch`, `modal-02-draft-mode-and-state.patch`,
  `modal-03-model-runner-draft-loop.patch` — from `modal-mtp/patches/`
  (renamed with a `modal-` prefix to distinguish from the qwen-ops versions of
  the same hooks).

See [`../vllm-patches-and-strategies.md`](../vllm-patches-and-strategies.md) for
what each patch does.
