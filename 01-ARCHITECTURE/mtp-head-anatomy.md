# The MTP Head — Anatomy of "Layer 64"

> Qwen3.5-27B ships a trained **Multi-Token Prediction (MTP) head**: a single transformer
> block that predicts the *next* token directly from the main model's final hidden state plus
> the just-decoded token's embedding. It is the draft model that every speculative-decoding
> result in this corpus runs on. Crucially, **every framework (transformers, vLLM, mlx-lm,
> llama.cpp) silently strips these weights on load** — so the first job in every domain was to
> reverse-engineer and re-attach them.

## What it is, in one paragraph

The MTP head is **one extra transformer layer**, stored on disk as block index **64** (the
65th layer of a 64-layer model). It takes two inputs — the main model's last hidden state
`h_t` (post layer-63) and the embedding `e` of the current/just-decoded token — RMS-normalizes
each, **concatenates them `[embed, hidden]`**, projects the `2·hidden` vector back down to
`hidden` via `eh_proj`, runs one gated self-attention + dense SiLU MLP block (same shape as the
backbone's full-attention layers), final-norms, and projects to vocab logits **through the main
model's shared `lm_head`** (because `mtp_use_dedicated_embeddings: false`). That single forward
gives you a cheap draft of the next token to verify against the full model.

## The 15 tensors

The MTP head is **15 weight tensors** (the count quoted throughout the MLX work). In the HF
checkpoint they live under `model.mtp.*`:

| # | HF name (under `model.mtp.`) | Role | Shape |
|---|---|---|---|
| 1 | `pre_fc_norm_hidden.weight` | RMSNorm on `h_t` before concat | `[5120]` |
| 2 | `pre_fc_norm_embedding.weight` | RMSNorm on `e` before concat | `[5120]` |
| 3 | `fc.weight` (= `eh_proj`) | projects concat `[embed‖hidden]` → hidden | `[5120, 10240]` (10240→5120) |
| 4 | `layers.0.input_layernorm.weight` | pre-attention RMSNorm | `[5120]` |
| 5 | `layers.0.self_attn.q_proj.weight` | gated Q (double-width: Q‖gate) | `[12288, 5120]` (5120→12288) |
| 6 | `layers.0.self_attn.k_proj.weight` | K (GQA) | `[1024, 5120]` (5120→1024) |
| 7 | `layers.0.self_attn.v_proj.weight` | V (GQA) | `[1024, 5120]` (5120→1024) |
| 8 | `layers.0.self_attn.o_proj.weight` | attention output | `[5120, 6144]` (6144→5120) |
| 9 | `layers.0.self_attn.q_norm.weight` | per-head Q RMSNorm | `[256]` (head_dim) |
| 10 | `layers.0.self_attn.k_norm.weight` | per-head K RMSNorm | `[256]` (head_dim) |
| 11 | `layers.0.post_attention_layernorm.weight` | pre-MLP RMSNorm | `[5120]` |
| 12 | `layers.0.mlp.gate_proj.weight` | SwiGLU gate | `[17408, 5120]` (5120→17408) |
| 13 | `layers.0.mlp.up_proj.weight` | SwiGLU up | `[17408, 5120]` (5120→17408) |
| 14 | `layers.0.mlp.down_proj.weight` | SwiGLU down | `[5120, 17408]` (17408→5120) |
| 15 | `norm.weight` | final RMSNorm before shared lm_head | `[5120]` |

> Shapes given as `[out, in]` (PyTorch/safetensors convention). The MLX README quotes them as
> "in → out" arrows; e.g. `q_proj (5120 -> 12288)`, `o_proj (6144 -> 5120)`, `fc (10240 -> 5120)`.
> All consistent. `head_dim = 256`, `num_heads = 24`, `num_kv_heads = 4` ⇒ q out `24·256·2 = 12288`,
> k/v out `4·256 = 1024`, o in `24·256 = 6144`.

There is **no** `shared_head.head` (lm_head) and **no** dedicated `embed_tokens` in the 15 — the
head reuses the main model's. (The HF checkpoint may expose `model.mtp.shared_head.head.weight`
and `model.mtp.shared_head.norm.weight` namespacing, but for Qwen3.5-27B the lm_head projection
falls back to the main `output.weight`; only the final `norm` is head-specific — that's tensor #15.)

## The `eh_proj` concat insight — `[embed, hidden]`, not `[hidden, embed]`

This is **the** load-bearing detail of the whole head and the single most-repeated warning in
the sources:

> **Concatenate embedding-first, hidden-second.** The fused projection is named `eh_proj`
> (**e**mbed-**h**idden), and the weight matrix is laid out for input ordered `[e_norm ‖ h_norm]`.
> Getting the order backwards (`[h, e]`) silently produces garbage — the matmul still runs, the
> shapes still match (both halves are `[5120]`), so there is no error, just wrong logits.

In MLX (`mtp_head.py`): `combined = mx.concatenate([e_norm, h_norm], axis=-1)` → `fc(combined)`.
In the llama.cpp graph (`qwen35.cpp` `build_mtp_head`): `ggml_concat(ctx0, e_norm, h_norm, 0)`
with the comment `NOTE (CRITICAL): embed first, hidden second. Matches GGUF "eh_proj" name.`

## Forward computation (the reference, from MLX `MTPHead.__call__`)

```
inputs: h_t  = main model last hidden state at position t   [B, 1, 5120]
        e    = embedding of the current token (pos t+1)      [B, 1, 5120]

1. NORM + CONCAT + PROJECT
   h_norm   = RMSNorm_hidden(h_t)
   e_norm   = RMSNorm_embedding(e)
   combined = concat([e_norm, h_norm], dim=-1)               [B, 1, 10240]   # EMBED FIRST
   h        = fc(combined)                                   [B, 1, 5120]    # eh_proj

2. GATED SELF-ATTENTION (single token; RoPE at position 0)
   residual = h
   x        = input_layernorm(h)
   q_full   = q_proj(x)            -> reshape [B,1,24, 256*2]
   q, gate  = split(q_full, 2, axis=-1)                      # gate is the second half
   k        = k_proj(x) -> [B,1,4,256];  v = v_proj(x) -> [B,1,4,256]
   q        = q_norm(q);  k = k_norm(k)
   q,k      = RoPE(q,k, offset=0, rope_dims=64, base=100000)
   o        = SDPA(q, k, v, scale=head_dim**-0.5)            # GQA, no past cache, no mask
   h        = residual + o_proj(o * sigmoid(gate))           # gated output

3. DENSE MLP (SwiGLU / SiLU)
   residual = h
   x        = post_attention_layernorm(h)
   h        = residual + down_proj( silu(gate_proj(x)) * up_proj(x) )

4. FINAL NORM -> SHARED LM HEAD
   h_normed = norm(h)
   logits   = lm_head_fn(h_normed)        # main model's lm_head (shared)
   return logits, h                       # h (pre-final-norm) returned for CHAINING
```

> **Returning `h` for chaining** is the hook that the MLX/llama.cpp "chained MTP" strategy uses:
> feed the head's own output hidden back in as the next step's `prev_hidden` to draft +2, +3, …
> from a single main forward pass. (That strategy lives in the MLX / llama.cpp domains; the head
> itself just needs to expose the pre-norm hidden.)

### llama.cpp graph notes (`qwen35.cpp build_mtp_head`)
The GGML port mirrors the MLX reference (the in-corpus copy: [`../03-MLX/src/mtp_head.py`](../03-MLX/src/mtp_head.py), with the fused-kernel variant in [`../03-MLX/src/fused_gdn.py`](../03-MLX/src/fused_gdn.py) `MTPHead`) tensor-for-tensor:
`hnorm(h)`, `enorm(e)`, `concat[e;h]`, `eh_proj`, then a gated self-attention block using the
backbone's `wq/wk/wv/wo` (with the `*2` Q/gate split), `attn_q_norm`/`attn_k_norm`, the dense
`ffn_gate/up/down` (SiLU), `shared_head_norm`, and finally `shared_head_head` (or main
`output.weight` when null). The MTP draft graph is entered via the
`LLM_GRAPH_TYPE_MTP` early-return branch in `llm_build_qwen35`, asserts
`ubatch.n_tokens == 1` (single seed token) and `nextn_predict_layers > 0`, and exports
`res->t_mtp_prev_hidden_in` (host uploads `h_t`) and `res->t_mtp_pos_zero_in` (the position-0
marker). The MTP layer index is `il_mtp = n_layer - nextn_predict_layers` (= 64).

## `mtp_use_dedicated_embeddings: false` — the shared head

Qwen3.5-27B sets **`mtp_use_dedicated_embeddings: false`**. Consequences:
- The MTP head has **no own lm_head** and **no own token embedding** — it reuses the main
  model's `output.weight` (lm_head) and `tok_embd`. This is why the head is only 15 tensors,
  not 17, and why per-head added params are only ~480M fp16 / ~120M Q4 (the giant
  `5120·151936 ≈ 778M` lm_head is *not* duplicated).
- In llama.cpp the dedicated tensors (`nextn.embed_tokens`, `nextn.shared_head_head`) are loaded
  as `TENSOR_NOT_REQUIRED`; the graph **falls back to `model.output` (or `model.tok_embd` if
  tied)** when they are null. An early loader version that *assumed* a dedicated
  `shared_head_head` crashed with a null-pointer — see the discovery in
  `tensor-layout-hf-to-gguf.md`.

## The RMSNorm `+1.0` shift (extraction gotcha)

Qwen3.5 stores RMSNorm weights as `w` where the computation is `x · (1 + w)` (a "centered"
parameterization). mlx-lm applies the `+1.0` shift in its Qwen3.5 sanitize hook for the main
model, **but it strips the MTP tensors entirely**, so when you extract MTP norms yourself you
must add `1.0` to every 1-D norm weight before use. Both extraction scripts do this for the 7
norm suffixes (`input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `norm`,
`pre_fc_norm_embedding`, `pre_fc_norm_hidden`). See `weight-extraction.md`.

## What it predicts (a precise-offset caveat)

The `qwen-mtp-tensors` README describes the head as predicting "**the +1 token** from the
layer-63 hidden state". The MLX README phrases the same head as predicting "**token t+2** from
the last hidden state at position t and the embedding of the token at position t+1". These are
**the same operation** described from different reference points: given hidden at `t` and the
embedding of the next committed token (`t+1`), the head outputs logits for the token after that.
"+1 relative to the embedding input" = "+2 relative to the hidden-state position". Not a
contradiction — just two coordinate systems. (Logged in the ledger.)

---

## Related

- [`qwen35-hybrid-architecture.md`](qwen35-hybrid-architecture.md) — the 64-layer backbone this head sits on top of.
- [`tensor-layout-hf-to-gguf.md`](tensor-layout-hf-to-gguf.md) — where these 15 tensors land in GGUF (`blk.64.nextn.*`) and the load fixes.
- [`weight-extraction.md`](weight-extraction.md) — pulling these tensors out of the safetensors shards (with the `+1.0` norm shift).
- [`../02-LLAMACPP/source/`](../02-LLAMACPP/source/) — `qwen35.cpp build_mtp_head`, the GGML port of this forward pass.
- [`../03-MLX/mtp-self-speculative-mlx.md`](../03-MLX/mtp-self-speculative-mlx.md) — chaining the head's returned hidden to draft +2, +3.
- [`../05-THEORY-AND-DESIGNS/per-position-heads-design.md`](../05-THEORY-AND-DESIGNS/per-position-heads-design.md) — the N-head alternative to one chained head.

---

### Provenance
- `mlx-qwen-mtp/README.md` "Architecture / MTP Head (15 tensors)" — the 15-tensor list, in→out shapes, "concat [embed, hidden]", t+2 framing
- `mlx-qwen-mtp/src/mtp_head.py` — exact module, forward computation, weight_map, dims, chaining return
- `qwen-mtp-tensors/README.md` — eh_proj concat insight, shared lm_head, +1 framing, layer-64 framing
- `llama-mtp/src/models/qwen35.cpp` (`build_mtp_head`, `LLM_GRAPH_TYPE_MTP` branch) — GGML port, embed-first concat, dedicated-head fallback
- `qwen-mtp-tensors/diffs/02-qwen35-graph-tensors.diff` — annotated graph build (see `tensor-diffs/`)
- extraction scripts (`extract/`) — the `+1.0` norm shift convention
