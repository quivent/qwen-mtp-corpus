# Infrastructure patches — the ordered series

> The substrate that the [optimization variants](optimization-variants.md) build on.
> Copied verbatim in [`patches/infrastructure/`](patches/infrastructure/). Apply order
> is by filename prefix: `0000..0004` (base MTP layer) then `01..11` (qwen35 port +
> infra + the rollback bookkeeping fix).

## Base commit

All patches apply against **upstream llama.cpp parent commit `9c7911f4f`** ("fix: V
tensor extraction with transposed KV cache"). To apply: clone llama.cpp, `git checkout
9c7911f4f`, then `git am` the patches in alphanumeric order. (Recorded in
`patches/infrastructure/00-base.txt`.) Author of all patches: `quivent`, dated
2026-04-08.

## The architecture being ported

Qwen3.5-27B = **48 DeltaNet** (linear-attention recurrent) layers interleaved with
**16 full-attention** layers + **one MTP (NextN) head as layer 64**. The MTP head is a
single dense full-attention transformer block. Tensor naming: `blk.64.nextn.{eh_proj,
enorm, hnorm, embed_tokens, shared_head_norm, shared_head_head}`.

The MTP head forward pass (the "embed-first" order is the #1 correctness pitfall):

```
h_n   = hnorm(prev_hidden);  e_n = enorm(tok_embed)
h     = eh_proj(concat([e_n, h_n]))        # CRITICAL: embed FIRST in the concat
h    += o_proj(sigmoid(gate) * SDPA(Q,K,V))# gated self-attn, q/k norms, MRoPE @ pos 0
h    += mlp_down(SiLU(mlp_gate) * mlp_up)   # dense SiLU MLP
logits= shared_head_head(shared_head_norm(h))
```

---

## The arc these patches trace

Read top-to-bottom, the series is one continuous build-out, not 16 unrelated fixes. It
starts with the **base MTP layer** (`0000–0004`): the snapshot/restore primitive for
DeltaNet's recurrent state, plus the first NextN-head graph builder and speculative driver
written against the **qwen3next** (hybrid MoE) family — the reference the dense path mirrors.
From there the dense **qwen35 port** (`01–03`) carries that machinery onto Qwen3.5-27B:
patch 01 turns the draft call from a stub into a working K-step drafter, patch 02 adds the
dense `build_mtp_head`, and patch 03 lands the five-fix commit that makes a converted
Qwen3.5 GGUF actually load and run end-to-end. The middle of the series (`04–06`) is
infrastructure hardening — naming the scheduler tensors to surface a ggml bug, chaining
`prev_hidden` correctly across draft steps, and isolating the MTP graph in its own
scheduler. Then comes **host-side rollback** (`07–10`): the split-recurrence rollback
primitive (`seq_rm_attn_only` + recurrent snapshot restore), the AR re-decode path, and
finally batching that re-decode into one T=N pass — which is what exposed the cache bug. The
series ends with **patch 11's bookkeeping fix**: the one-line `id_last`-from-tail-logits
correction that unmasks every previously-broken measurement and makes the entire spec path
produce correct output. The detailed apply order and per-patch table follow.

---

## The 16 patches

### Base MTP layer (0000–0004)

| # | Patch | What it does | Why it was needed |
|---|---|---|---|
| 0000 | recurrent-state snapshot/restore for DeltaNet | New public API `llama_memory_snapshot_recurrent` / `restore_recurrent` / `release_snapshot` / `snapshot_size`. Copies only the small per-layer `conv_state`/`rnn_state` tensors (KB/layer) + cell metadata. New files `src/llama-memory-snapshot.{cpp,h}`. Leaves a `TODO(split-rec)` in `qwen3next.cpp`. | Spec decoding must capture recurrent state before drafting and restore cheaply on rejection — without re-running a forward pass. The primitive didn't exist. |
| 0001 | qwen3next MTP (NextN) head graph builder + load path | Registers `NEXTN` tensor enums for `LLM_ARCH_QWEN3NEXT`; reads `LLM_KV_NEXTN_PREDICT_LAYERS`; loads the 6 NextN tensors (all `TENSOR_NOT_REQUIRED`); adds `has_mtp_head()`; implements `build_mtp_head` (matches MLX `fused_gdn.py` `MTPHead`, embed-first concat). Exposes C API `llama_model_has_mtp_head` + `llama_mtp_draft` (returns -4 stub until graph exec wired). | The MTP head graph builder for the hybrid (qwen3next) family — the reference implementation the dense qwen35 path is later mirrored from. |
| 0002 | speculative: MTP single-model speculative decoding | New `common_speculative_mtp_*` API + `examples/mtp-speculative/`. Unlike standard spec (separate draft model), the MTP variant drafts from a head **on the target context itself**. Mirrors MLX `mtp_speculative_decode.py`: draft K, verify in one K+1 forward, greedy accept, trim KV on reject. Rollback stubbed with `TODO(rollback)`. | The driver + helper API for single-model spec decoding. Link of `llama-mtp-speculative` fails on `_llama_mtp_draft` until graph agent merges. |
| 0003 | wire rollback snapshot/restore into mtp-speculative | Connects the 0000 snapshot API into the example's rollback path. | Glue. |
| 0004 | wire `llama_mtp_draft` execution path | Adds `LLM_GRAPH_TYPE_MTP` enum. Captures **post-output-norm hidden of the last decoded token** into `mtp_prev_hidden` after each main decode (independent of `cparams.embeddings`). Exposes `get_mtp_prev_hidden()`. `llama_mtp_draft` now validates inputs and returns -3 (missing prev_hidden) / -4 (missing graph dispatch). | The hidden-state capture is the input the MTP head needs; this is partial Stage 3 wiring. |

### qwen35 port + infrastructure + the fix (01–11)

| # | Patch | What it does | Why it was needed |
|---|---|---|---|
| 01 | execute MTP draft graph for qwen3next via `LLM_GRAPH_TYPE_MTP` | Turns `llama_mtp_draft` from a -4 stub into a working **K-step greedy drafter**. qwen3next constructor builds the NextN head only over a 1-token ubatch; `mtp_graph_compute` dispatcher uploads `prev_hidden` + RoPE pos-zero, computes, pulls back vocab logits via a dedicated `gf_res_mtp`. K-step loop, greedy argmax. **Known trade-off:** reuses `mtp_prev_hidden` across all K steps (matches dense PR #20700) instead of feeding the head's own output forward. | Last-mile dispatch — the head now actually runs. |
| 02 | qwen35: add MTP draft-graph path mirroring qwen3next | Adds `build_mtp_head` + `LLM_GRAPH_TYPE_MTP` early-return to `llm_build_qwen35`. Qwen3.5 layer 64 is **dense full-attention** so FFN uses `ffn_{gate,up,down}` and attn uses `w{q,k,v,o}` with the `*2` gate factor on wq. Exports `t_mtp_prev_hidden_in`, `t_mtp_pos_zero_in`. `TODO(dedupe)` with qwen3next. | The dense Qwen3.5-27B family needs its own draft path. |
| 03 | end-to-end MTP head load+execute for qwen35 dense (**5 fixes in one commit**) | **Fix 1**: add `FFN_NORM` + 6 `NEXTN_*` to `gguf-py` QWEN35 tensor list. **Fix 2**: `convert_hf_to_gguf.py` bumps `block_count` by `mtp_num_hidden_layers` and rebuilds `tensor_map` (gated `n_mtp>0` so 80B MoE unaffected). **Fix 3**: loader reads `nextn_predict_layers`, stops marking layer 64 recurrent (MTP is dense), maps `(n_layer-nextn)==64`→`LLM_TYPE_27B`, allocates `layer.nextn.*`, swaps `ATTN_POST_NORM`→`FFN_NORM`. **Fix 4**: arch reclassifies `NEXTN_*` from `LAYER_OUTPUT`→`LAYER_REPEATING` (template is `blk.%d.nextn.*`). **Fix 5**: rewrite head attn to use `build_attn_inp_no_cache`+`build_attn` (manual K*Q+softmax had two shape bugs); switch RoPE to `ggml_rope_multi` w/ `rope_sections`; fall back to `model.output`/`tok_embd` when `shared_head_head` null. **Fix 6**: `mtp_graph_compute` uploads full `inp_pos_zero` (length 4 for MRoPE). | A converted Qwen3.5-27B (`mtp_num_hidden_layers: 1`) silently failed to load and run at every layer of the stack. This is the "model won't load → plausible drafts" commit. Known remaining: rollback hits `find_slot: non-consecutive token position` errors in verify. |
| 04 | name mask tensors to surface ggml scheduler bug | Names `kq_mask`/`kq_mask_cnv` tensors ("nc_kq_mask", "kv_kq_mask", …). Does **not** fix the bug — makes the crash message point at the real tensor (`kv_kq_mask_cnv`). | `llama_mtp_draft` between main decodes crashed at `sched_split_graph` ("pre-allocated tensor ` (copy)` in MTL0 buffer cannot run CPY"). Root cause: `ggml_cast` of an unnamed tensor → " (copy)" with leading space hid which tensor failed; hypothesis is `ggml_backend_sched_reset` not fully cleaning `prev_node_backend_ids`/gallocr state across two graphs (`gf_res_mtp` vs `gf_res_prev`) sharing one sched. |
| 05 | correctly chain `prev_hidden` across K draft steps | Adds graph-output slot `t_mtp_out_hidden` (the MTP layer's post-MLP residual hidden, pre-`shared_head_norm`). Rewrites the K-step loop to feed each step's `out_hidden` as the next step's `prev_hidden_override` (step 1 still uses the cached main-decode hidden). | Patch 01's shortcut reused stale `mtp_prev_hidden` for steps 2..K. **Empirical:** K=1 → 100% accept, 25.5 tok/s, 1.43× vs llama-completion; K=2 → 50%; K=3 → 33%. The K>1 numbers were **identical to pre-fix** → the fix is correct but Qwen3.5's single head is architecturally K=1 (chaining is out-of-distribution). DeepSeek-V3's K dedicated heads would benefit. *(Note: commit message attributes the original port shortcut to "Karpathy's original port (commit 83babcae7)".)* |
| 06 | isolate `mtp_graph_compute` in a private `ggml_backend_sched` | Gives the MTP graph its own scheduler instead of sharing the main decode's. | Fixes the cross-decode state leak surfaced by patch 04 (stale scheduler state binding the mask CPY to a wrong-backend buffer). |
| 07 | v1 host-side rollback on rejection via snapshot + attn-only seq_rm | New `llama_memory_seq_rm_attn_only()` — trims only the **attention half** of a hybrid memory, leaving recurrent untouched (avoids `llama_memory_recurrent`'s partial-tail refusal). On rejection: (1) restore recurrent snapshot, (2) attn-only `seq_rm` to pre-verify pos, (3) re-decode `[id_last, accepted_drafts, correction]` as one batch. | The missing split-recurrence rollback primitive. Eliminates the "v1 no-trim workaround" that left rejected drafts contaminating both attention KV and DeltaNet state (had dropped throughput 1.44×→0.81× at n=64). |
| 08 | AR re-decode path for rollback + `MTP_FORCE_AR` diagnostic | (1) Rollback re-decode now commits `[id_last, accepted, corr]` **one token at a time** (keeps every post-rollback token on `build_delta_net_autoregressive`, n_seq_tokens==1, matching plain greedy numerics). (2) `MTP_FORCE_AR=1` disables drafting → plain greedy AR decoder. **With it, output byte-identical to `llama-simple`**: 'capital of France' 18.3 vs 17.1 tok/s; 'Once upon a time' 18.2 vs 17.3 tok/s. | An always-on ground-truth path. Documents split-rec-v2: speculation-enabled verify has n_tokens≥2 → forces chunking → divergence compounds through rejections. True byte-identity needs in-graph split-recurrence (Option C). |
| 09 | **wip** in-graph AR loop for T≤16 verify + state-capture diag | Under `MTP_VERIFY_FORCE_AR=1`, loop `build_delta_net_autoregressive` per token while keeping input/output projections batched (gated `n_seq_tokens<=16` so `sched_reserve` still sizes buffers via chunking). Adds `MTP_DEBUG_CAPTURE=1` (dump `mtp_prev_hidden` checksums per decode). | **Investigation result: chunking vs AR is NOT the root cause** — both produce identical iter 1-3 draft/target IDs. Real drift hypothesis (at the time) pointed at `mtp_prev_hidden` capture in rollback re-decodes producing subtly different hiddens. Commit `987541157`. |
| 10 | **perf** batch rollback re-decode into single T=N llama_decode | Replaces N sequential T=1 re-decodes after a rejection with one T=N batched decode at the same positions. Uses chunking (in-graph AR has higher per-step overhead for small T; divergence bounded by n_commit). Adds a `scoped_force_ar` RAII helper in `mtp-speculative.cpp` that sets `MTP_VERIFY_FORCE_AR` for the duration of a single `llama_decode` then restores it — so the one T=N batch is numerically equivalent to N T=1 decodes at the cost of one dispatch. Keeps patch 08's happy-path skip-second-forward. | Perf. K=1 vanilla 5-prompt mean 3.68 tok/s (was ~2.0 estimated). **This patch's batching is what exposed the cache bug** — the double-write only manifests with a batched re-decode. |
| 11 | **fix** rollback re-decode bookkeeping — `id_last` from tail logits | **The one-line unblock.** After the batched re-decode of `[id_last, drafts…, corr]`, the old code did `id_last = corr` — but `corr` was already the batch's last cache slot. Next verify wrote `corr` again at `n_past+n_commit`, shifting every position by one → garbage ("1. 2. 3. … 1000000" loops within ~10 tokens of any rejection). **Fix:** read tail logits of the re-decoded batch, use `argmax` as `id_last` (mirrors the accept-all branch). Also adds debug knobs `MTP_DEBUG_SLOT0`, `MTP_ROLLBACK_T1`, `MTP_FULL_SEQRM`. | **The fix that makes the entire spec path produce correct output.** Confirmed: K=1 spec now matches plain decode structurally (planets/photosynthesis/fibonacci/haiku/translate). K=1 spec mean = **7.64 tok/s** (plain 17.90). See [`the-bug.md`](the-bug.md). Commit `b070bed01`. |

---

## Quick reference: which patch introduced which primitive / env var

| Primitive / flag | Patch | File |
|---|---|---|
| `llama_memory_snapshot_recurrent` etc. | 0000 | `llama-memory-snapshot.{cpp,h}` |
| `llama_mtp_draft` (stub) | 0001 | `llama-context.cpp` |
| `LLM_GRAPH_TYPE_MTP` | 0004 | `llama-graph.h` |
| `mtp_prev_hidden` capture | 0004 | `llama-context.cpp` |
| `mtp_graph_compute` dispatcher | 01 | `llama-context.cpp` |
| qwen35 `build_mtp_head` | 02 | `qwen35.cpp` |
| converter / loader / arch fixes | 03 | `convert_hf_to_gguf.py`, `gguf-py`, `llama-arch.cpp`, `llama-model.cpp` |
| `t_mtp_out_hidden` (K-chain) | 05 | `llama-graph.h`, `qwen35.cpp` |
| private `sched_mtp` | 06 | `llama-context.cpp` |
| `llama_memory_seq_rm_attn_only` | 07 | `llama-context.cpp` |
| `MTP_FORCE_AR` | 08 | `mtp-speculative.cpp` |
| `MTP_VERIFY_FORCE_AR`, `MTP_DEBUG_CAPTURE` | 09 | `qwen35.cpp`, `llama-context.cpp` |
| batched rollback re-decode | 10 | `mtp-speculative.cpp` |
| **`id_last` tail-logit fix**, `MTP_DEBUG_SLOT0`, `MTP_ROLLBACK_T1`, `MTP_FULL_SEQRM` | **11** | `mtp-speculative.cpp` |

---

## Related

- [`optimization-variants.md`](optimization-variants.md) — the 9 variants built on this substrate (variant 01 is the winner).
- [`the-bug.md`](the-bug.md) — the cache-bookkeeping bug that patch 11 fixes.
- [`the-recipe.md`](the-recipe.md) — the 1.99× recipe that patch 11 finally unmasks.
- [`source/`](source/) — the MTP-specific fork source files these patches modify.
- [`../01-ARCHITECTURE/tensor-layout-hf-to-gguf.md`](../01-ARCHITECTURE/tensor-layout-hf-to-gguf.md) — patch 03's five fixes, explained tensor-by-tensor.
- [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) — the split-recurrence rollback patches 07–10 implement.

---
*Provenance: distilled from the patch commit messages in
`qwen-ops/llamacpp/infrastructure/*.patch` (= `qwen-mtp-llamacpp/patches/*`,
byte-identical) and `qwen-mtp-llamacpp/README.md`.*
