# recurrent-rollback reference implementation (origins)

MLX reference implementation of split-recurrence rollback, copied verbatim from the source
repo `recurrent-rollback/` (Josh Kornreich, MIT). The distilled technique writeup is in
[`../recurrent-rollback-technique.md`](../recurrent-rollback-technique.md).

| File (here) | Origin | What it is |
|---|---|---|
| `src/__init__.py` | `recurrent-rollback/src/__init__.py` | Package exports (`split_recurrence_forward`, `rollback_to`, `RollbackPoint`, `DeltaNetRollbackLayer`). |
| `src/split_recurrence.py` | `recurrent-rollback/src/split_recurrence.py` | Architecture-agnostic core: `RecurrentLayer`/`AttentionLayer` protocols, `RollbackPoint`, `SplitForwardOutput`, `split_recurrence_forward(...)`, `rollback_to(...)`, `_trim_kv_cache`. |
| `src/delta_net_rollback.py` | `recurrent-rollback/src/delta_net_rollback.py` | DeltaNet-specific: `DeltaNetState{conv_state, rnn_state}`, `DeltaNetRollbackLayer` (conv1d+SiLU + GDN delta rule in `recurrence_step`), `wrap_model_layers(model)`. |

> **State-shape correction (kept verbatim, flagged here).** The copied
> `delta_net_rollback.py` docstring (lines 23–24) cites an illustrative state shape of
> `14 heads × 256 × 256 = 1.75 MB/layer → 84 MB` total. That shape is **wrong** for the
> shipped Qwen3.5-27B. The adjudicated per-layer state is
> `ssm_state = (num_v_heads=48, head_v_dim=128, head_k_dim=128) = 786,432 elements` plus
> `conv_state = (3, 10240)`, totalling **75.5 MB ssm-only / ~78 MB with conv (BF16);
> 151 MB FP32** over 48 layers. The file is preserved unedited for source fidelity; the
> distilled doc [`../recurrent-rollback-technique.md`](../recurrent-rollback-technique.md)
> uses the corrected numbers. Authoritative source:
> [`../../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../../01-ARCHITECTURE/qwen35-hybrid-architecture.md).
| `examples/__init__.py` | `recurrent-rollback/examples/__init__.py` | (empty) |
| `examples/mtp_speculative_decode.py` | `recurrent-rollback/examples/mtp_speculative_decode.py` | Full MTP-drafted spec loop with split-recurrence rollback + stats (accept/reject, rollbacks, estimated redo saved). Qwen EOS token `151643`. |
| `examples/simple_example.py` | `recurrent-rollback/examples/simple_example.py` | Minimal correctness check: prefill, process 2 tokens, `rollback_to(rollback_points[0])`, verify restored state matches (diff < 1e-8). |

Targets `Qwen/Qwen3.5-27B-MLX-4bit`. Live MLX integration, kernel work, and the M4 Max
journey are Agent C's [`../../03-MLX/`](../../03-MLX/).
