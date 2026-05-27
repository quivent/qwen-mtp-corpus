# optimizations/ — speculative-decode strategy modules + validation

Origins:
- Strategy modules (`adaptive_mtp.py`, `modal_mtp.py`, `partial_layer_verify.py`,
  `plv_bench.py`, `plv_layer60_bench.py`, `early_verify_probe.py`,
  `deltanet_transplant.py`, `deltanet_transplant_w4a16.py`,
  `cascade_mtp_corrective.py`, `deltanet_adjuster.py`,
  `enhanced_mtp_proposer.py`, `native_multi_head.py`, `sibling_sequential.py`,
  `selective_state_snapshot.py`) — copied from `qwen-ops/vllm/optimizations/`.
- `microgreens/` (sibling MTP heads: `mtp_clone.py`, `mtp_diversity_train.py`,
  `sibling_mtp_proposer.py`, `__init__.py`) — from `qwen-ops/vllm/microgreens/`.
- `scripts/` (`vllm-tree-spec.sh`, `bench-tok-s.py`, `quantize_deltanet.py`) —
  from `qwen-ops/vllm/scripts/`.
- `modal_mtp_proposer_reference.py` — copy of `modal-mtp/modal_mtp.py`.
- `diagnose_divergence.py`, `validate_draft_accuracy.py`, `validate_extended.py`,
  `test_partial_skip.py`, `validate_extended_output.log` — from `modal-mtp/`.

See [`../vllm-patches-and-strategies.md`](../vllm-patches-and-strategies.md) for
the explained table, and [`../precision-divergence.md`](../precision-divergence.md)
for what the validation scripts proved.
