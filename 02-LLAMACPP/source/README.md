# llama.cpp fork — MTP source files (copied verbatim)

These are the **MTP-specific source files** lifted from the llama.cpp fork
(`llama-mtp/`, 383 MB, mostly upstream). Only the files that implement the
Qwen3.5-27B MTP speculative-decoding path are copied here; the rest of the
fork is referenced by path. These files represent the **post-fix state** (all
11+ infrastructure patches and the optimization variants applied on top of
upstream llama.cpp).

| File here | Original path in `llama-mtp/` | What it implements |
|---|---|---|
| `qwen35.cpp` | `src/models/qwen35.cpp` | The **dense Qwen3.5** graph builder. Contains `build_mtp_head` (the NextN head: enorm(embed) + hnorm(hidden) → embed-first concat → `eh_proj` → one gated full-attention block with q/k norms and MRoPE at pos 0 → dense SiLU MLP → `shared_head_norm` → `shared_head_head` to vocab). Has the `LLM_GRAPH_TYPE_MTP` early-return branch, the `t_mtp_out_hidden` export for K-step chaining, and the `MTP_VERIFY_FORCE_AR` in-graph AR loop (`build_delta_net_autoregressive` per token, gated `n_seq_tokens<=16`) alongside `build_delta_net_chunking`/`build_delta_net_recurrent`. |
| `qwen3next.cpp` | `src/models/qwen3next.cpp` | The **qwen3next (hybrid)** graph builder that the qwen35 path was mirrored from. `build_mtp_head` at line 509. Carries the `TODO(split-rec)` marker describing the split-recurrence work (split DeltaNet into T=1 steps while keeping input/output projections batched at T=N — the MLX `delta_net_rollback.py` reference). DeltaNet kernel dispatch (`build_delta_net_autoregressive` for T=1 vs `build_delta_net_chunking` for T>1) lives here too. |
| `speculative-mtp.cpp` | `common/speculative-mtp.cpp` | The single-model MTP speculative helper (`common_speculative_mtp_*`). Drafts from the MTP head on the *target* model itself (no separate draft model), verifies via one K+1-token forward pass, greedy-accepts, trims KV on rejection. Stats mirror the MLX reference output format. |
| `speculative-mtp.h` | `common/speculative-mtp.h` | Public API + the forward-declared core entry point `llama_mtp_draft(ctx, last_token, n_draft, out_tokens, out_logits)` and its contract. `common_speculative_mtp_stats` fields (exact names): `total_tokens, draft_tokens, accepted_tokens, rejected_tokens, forward_passes, rollbacks, draft_calls`. The `common_speculative_mtp_*` helpers: `init / free / draft / verify_greedy / accept / note_forward_pass / note_committed / get_stats / print_stats` (the last mirrors the MLX reference output format). |
| `mtp-speculative.cpp` | `examples/mtp-speculative/mtp-speculative.cpp` | The **driver example** (`llama-mtp-speculative` binary). Implements the verify/accept/rollback loop, the batched rollback re-decode, the famous `id_last` bookkeeping fix (patch 11), and reads every `MTP_*` env var (see below). |
| `mtp-speculative.CMakeLists.txt` | `examples/mtp-speculative/CMakeLists.txt` | Build target `llama-mtp-speculative` (links `common`, `llama`; `cxx_std_17`). |

## Other MTP hooks in the fork (NOT copied — referenced by path)

The MTP wiring also touches these upstream-shared files (changes are captured
in the patch series under `../patches/`, not copied as whole files):

- `llama-mtp/src/llama-arch.{cpp,h}` — `LLM_ARCH_QWEN35` tensor registration + 6 `NEXTN_*` enums, reclassified `LAYER_REPEATING`.
- `llama-mtp/src/llama-model.cpp` — loader: `nextn_predict_layers`, `layer.nextn.*` allocation, recurrent-layer mask fix, `has_mtp_head()`.
- `llama-mtp/src/llama-hparams.h` — `nextn_predict_layers` hparam.
- `llama-mtp/src/llama-graph.h` — `LLM_GRAPH_TYPE_MTP`, `t_mtp_prev_hidden_in`, `t_mtp_pos_zero_in`, `t_mtp_out_hidden` graph-result slots.
- `llama-mtp/src/llama-context.{cpp,h}` — `mtp_graph_compute` dispatcher, `mtp_prev_hidden` capture, `llama_mtp_draft` K-step loop, the chained-recurrent threading (`next_prev_hidden = scratch_out_hidden.data()` ~line 4180) and confidence gate (~line 4140), `MTP_CHAIN_*` / `MTP_STACK_*` / `MTP_PREDICTIVE_ALPHA` / `MTP_ENSEMBLE_*` env handling.
- `llama-mtp/src/llama-memory-snapshot.{cpp,h}` — recurrent-state snapshot/restore (patch 0000).
- `convert_hf_to_gguf.py`, `gguf-py/gguf/constants.py` — converter + tensor-name fixes (see `MTP/01-ARCHITECTURE` for the tensor deep-dive).

## The MTP draft → verify → rollback flow (in code terms)

The driver loop in `mtp-speculative.cpp` (linear / non-tree path), per generated cycle:

1. **Draft.** `common_speculative_mtp_draft(spec, id_last, n_draft_max)` →
   `llama_mtp_draft(ctx, id_last, n_draft, out_tokens, out_logits)` in
   `src/llama-context.cpp`. Runs the MTP head over a 1-token ubatch via the private
   `sched_mtp` (patch 06), uploading the captured `mtp_prev_hidden` + RoPE pos-zero. K-step
   greedy argmax; on a chained step (confidence gate, `MTP_CHAIN_THRESH`) it feeds the
   head's own `t_mtp_out_hidden` back as the next step's `prev_hidden` (patch 05 + opt 01).
2. **Verify.** Build one batch `[id_last, d_0, …, d_{K-1}]` and `llama_decode` it in a
   single target-model forward pass (`common_speculative_mtp_note_forward_pass`). Slot-0
   argmax = the target's true next token; each accepted draft requires
   `argmax(slot_i) == d_i`. `MTP_DEBUG_VERIFY`/`MTP_DEBUG_SLOT0` dump these.
3. **Accept / commit.** Longest matching prefix is accepted; the verify's tail-logits
   argmax is the **bonus token** committed for free. `common_speculative_mtp_accept` +
   `note_committed`.
4. **Rollback on rejection.** (a) `llama_memory_snapshot_recurrent` was taken before
   verify; on reject `llama_memory_restore_recurrent(ctx, snap)` rewinds the DeltaNet
   state. (b) `llama_memory_seq_rm_attn_only(ctx, 0, n_past, -1)` trims only the attention
   KV (patch 07) — recurrent is untouched, sidestepping the partial-tail refusal. (c)
   re-decode `[id_last, accepted…, corr]` as one T=N batch (patch 10, via `scoped_force_ar`).
   (d) **patch 11 fix:** set `id_last = argmax(tail logits of the re-decoded batch)` — NOT
   `id_last = corr` (the bug that double-wrote `corr` and shifted every position).
5. **Fast-path skip (opt 08, BROKEN).** On a top-1 ensemble hit,
   `llama_memory_seq_rm_attn_only(ctx, 0, n_past+2, -1)` + `llama_memory_seq_force_recurrent_pos(ctx, 0, n_past+1)`
   skips the commit re-decode entirely (no `llama_decode`). Leaves recurrent contamination
   → incoherent output on this hybrid model.

Core primitives by file: snapshot/restore in `llama-memory-snapshot.{cpp,h}` (patch 0000);
`seq_rm_attn_only` + `seq_force_recurrent_pos` in `llama-context.cpp` (patches 07, opt 08);
the K-step chain + confidence gate + stacked-noise in `llama_mtp_draft`
(`llama-context.cpp`, patches 05 / opt 01 / opt 09).

## Env vars read by `mtp-speculative.cpp`

`MTP_DEBUG`, `MTP_DEBUG_SLOT0`, `MTP_DEBUG_VERIFY`, `MTP_ENSEMBLE_FAST`,
`MTP_ENSEMBLE_K`, `MTP_FORCE_AR`, `MTP_FULL_SEQRM`, `MTP_REFRESH_EVERY`,
`MTP_ROLLBACK_T1`, `MTP_TREE_B`, `MTP_TREE_DEPTH`, `MTP_VERIFY_FORCE_AR`.
(The recipe knobs `MTP_CHAIN_KMAX` / `MTP_CHAIN_THRESH` are read in
`src/llama-context.cpp::llama_mtp_draft`, not in the example.)

---
*Provenance: copied from `/home/ubuntu/qwen27/llama-mtp/` (the llama.cpp fork).*
