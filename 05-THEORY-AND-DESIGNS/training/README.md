# Training scripts (origins)

Design-only MTP-head training scripts, copied verbatim from the source corpus. **Neither
was executed** — both guard with a `[DESIGN-ONLY] … NOT RUNNING` exit and require an
H100-class GPU. Distilled discussion is in
[`../training-mtp-heads.md`](../training-mtp-heads.md) and
[`../per-position-heads-design.md`](../per-position-heads-design.md).

| File | Origin | Purpose |
|---|---|---|
| `build_training_data.py` | `qwen-ops/training/build_training_data.py` | Corpus → `(h_t, targets, prev_tokens)` parquet shards for per-position head training. Storage math; recommends tokens-only + on-the-fly hidden recompute (36 GB vs ~4 TB). ~100 GPU-hours for 1B tokens. |
| `train_per_position_heads.py` | `qwen-ops/training/train_per_position_heads.py` | Freeze main Qwen3.5-27B, warm-start N=4 MTP blocks from `mtp.layers.0` + N(0,0.02) noise, summed per-head CE, teacher forcing. DeepSeek-V3 hyperparameters; expected loss curves & accept rates; ~23 GPU-hours / ~$80 (Phase 1+2 ≈ ~50 GPU-hours / ~$175). |

Related cascade scripts referenced in [`../cascade-mtp-training.md`](../cascade-mtp-training.md)
(`collect_hidden_states.py`, `cascade_mtp_corrective.py`) live in the vLLM/GH200 ops domain
(Agent D, [`../../04-VLLM-GPU/`](../../04-VLLM-GPU/)) where the W4A16 serving checkpoint runs.
