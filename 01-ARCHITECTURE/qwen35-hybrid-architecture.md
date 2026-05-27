# Qwen3.5-27B — The Hybrid Architecture

> The architectural facts everything else in this corpus depends on. If you only read
> one file before touching MTP code, read this one. The MTP head (next doc) is a 65th
> layer bolted onto this backbone; the speculative-decoding work (other domains) lives
> or dies on the DeltaNet/full-attention split described here.

## The one-paragraph picture

Qwen3.5-27B is a **64-layer hybrid** language model. 48 of those layers are **DeltaNet**
(a linear-attention recurrent / gated-delta-rule layer) and 16 are **standard full
(softmax) attention**, interleaved in a strict **3:1 pattern** — three DeltaNet layers,
then one full-attention layer, repeating. On top of the 64 backbone layers the checkpoint
carries **one extra MTP (Multi-Token Prediction) head** stored as "layer 64", giving
**65 layers total on disk**. Position encoding is **MRoPE** (multi-axis RoPE). The model
is normally run as **Qwen3.5-27B-4bit / Q4_K_M (~13.7 GB weights)**.

## Why hybrid?

Full softmax attention is O(n²) in sequence length and needs a growing KV cache. DeltaNet
(linear / recurrent attention) is O(1) per token with a **fixed-size recurrent state** — no
growing cache. A pure-DeltaNet model is fast and cheap but loses some of the exact
long-range recall that softmax attention provides. The 3:1 hybrid keeps most layers cheap
and recurrent while periodically injecting a full-attention layer to restore global recall.
This is the central tension of the whole corpus: **DeltaNet's recurrence is what makes the
model efficient, and also what makes speculative-decoding rollback hard** (see below).

## The layer table

| Layer index | Type | Count | Role |
|---|---|---|---|
| 0–47 | DeltaNet (linear-attention recurrent, "GatedDeltaNet" / GDN) | 48 | Hybrid backbone — fixed-state recurrence, O(1)/token |
| 48–63 | Full attention (softmax, GQA, MRoPE) | 16 | Hybrid backbone — interleaved global recall |
| **64** | **MTP head** | **1** | Predicts the +1 token from the layer-63 hidden state (see `mtp-head-anatomy.md`) |

> **The 3:1 pattern is implemented as `full_attention_interval = 4`.** A layer is recurrent
> (DeltaNet) when `(i + 1) % 4 != 0`, i.e. layers 3, 7, 11, … (0-indexed: every 4th layer
> counting from 1) are full attention. Over 64 layers this yields 16 full-attention layers
> and 48 DeltaNet layers. The MTP layer at index 64 is **always dense full-attention** and is
> explicitly excluded from the recurrent mask (`i < n_base` guard — see the loader fix in
> `tensor-layout-hf-to-gguf.md`).

### "Type 27B" detection
In the llama.cpp fork the model is identified as `LLM_TYPE_27B` when
`(n_layer - nextn_predict_layers) == 64`. The subtraction of the MTP predict layer is what
makes 65-on-disk resolve to the 64-layer 27B type. (24 → `LLM_TYPE_2B` is the only other case
handled for this arch.)

## Core dimensions (canonical / resolved)

These are the values used by the working extraction code, `mtp_head.py`, and confirmed by the
vLLM model configs that read the real `text_config`:

| Hyperparameter | Value | Notes |
|---|---|---|
| `hidden_size` (n_embd) | **5120** | model width |
| `num_attention_heads` | **24** | query heads (full-attn layers + MTP head) |
| `num_key_value_heads` | **4** | GQA: 24/4 = 6× key/value sharing |
| `head_dim` | **256** | per-head dim; note `n_head*head_dim = 6144 ≠ hidden_size 5120` |
| `intermediate_size` (n_ff) | **17408** | dense SwiGLU/SiLU FFN width |
| `rms_norm_eps` | **1e-6** | all RMSNorms |
| `rope_theta` (freq_base) | **100000.0** | |
| `partial_rotary_factor` | **0.25** | rotary applied to 0.25·head_dim = **64** dims (MLX path) |
| `rope_dimension_count` (`n_rot`) | **64** | `int(head_dim·partial_rotary) = int(256·0.25)`; written by converter, read into `hparams.n_rot` |
| `vocab_size` | **151936** | (from per-position-heads dedicated-head math: 5120·151936 ≈ 778M) |
| `full_attention_interval` | **4** | the 3:1 DeltaNet:attention pattern |
| `mtp_num_hidden_layers` | **1** | one MTP head |
| `mtp_use_dedicated_embeddings` | **false** | MTP reuses the main embedding + lm_head |

### Attention head shapes (full-attn layers and the MTP head)
Qwen3.5 attention uses a **gated query**: the Q projection is *double width* and split into a
query half and a gate half (sigmoid gate applied to the attention output).
- `q_proj`: 5120 → **12288** = `num_heads(24) · head_dim(256) · 2`  (Q ‖ gate)
- `k_proj`: 5120 → **1024** = `num_kv_heads(4) · head_dim(256)`
- `v_proj`: 5120 → **1024**
- `o_proj`: **6144** → 5120, where 6144 = `num_heads(24) · head_dim(256)`
- `q_norm`, `k_norm`: RMSNorm over `head_dim` = **256**

### DeltaNet (GDN) layer dimensions (canonical)
The "GatedDeltaNet" (GDN) linear-attention block. Values below are pinned against the MLX
runtime (`MTP/03-MLX/src/fused_gdn.py`), the llama.cpp GDN graph (`qwen35.cpp` /
`qwen3next.cpp`), the converter hparam mapping (diff `01`), and the vLLM configs:

| GDN hyperparameter | Value | GGUF / llama.cpp name | Source of truth |
|---|---|---|---|
| `linear_key_head_dim` (`head_k_dim`) | **128** | `ssm_d_state` (`LLM_KV_SSM_STATE_SIZE`) | converter `add_ssm_state_size`; `qwen35.cpp:241` |
| `linear_value_head_dim` (`head_v_dim`) | **128** | derived: `ssm_d_inner / ssm_dt_rank` | `qwen35.cpp:244` |
| `linear_num_key_heads` (`num_k_heads`) | **16** | `ssm_n_group` (`LLM_KV_SSM_GROUP_COUNT`) | `qwen35.cpp:242` |
| `linear_num_value_heads` (`num_v_heads`) | **48** | `ssm_dt_rank` (`LLM_KV_SSM_TIME_STEP_RANK`) | converter `add_ssm_time_step_rank`; `qwen35.cpp:243` |
| `ssm_d_inner` | **6144** = `head_v_dim·num_v_heads` (128·48) | `ssm_d_inner` (`LLM_KV_SSM_INNER_SIZE`) | converter `add_ssm_inner_size` |
| `linear_conv_kernel_dim` (`conv_kernel_size`) | **4** | `ssm_d_conv` (`LLM_KV_SSM_CONV_KERNEL`) | converter `add_ssm_conv_kernel` |
| `conv_dim` / `conv_channels` | **10240** = `d_inner + 2·n_group·d_state` (6144 + 2·16·128) | — | `qwen35.cpp:288` |
| `key_dim` | **2048** = `num_k_heads·head_k_dim` (16·128) | — | `fused_gdn.py` |

> **Two different "head_dim"s — do not conflate.** The *full-attention* layers (and the MTP head)
> use `head_dim = 256` (= llama.cpp `n_embd_head_k = n_embd_head_v`). The *DeltaNet* layers use
> `head_k_dim = head_v_dim = 128` (mapped to GGUF `ssm_d_state` / derived from `ssm_d_inner`).
> They are unrelated quantities that happen to share the word "head_dim".

> **`num_v_heads` = 48 is the real value.** One vLLM helper (`deltanet_adjuster.py`) carries a
> `linear_num_value_heads = 32` *default placeholder*; the authoritative `selective_state_snapshot.py`,
> the modal README's `[48,128,128]` state shape, the ~77 MB snapshot figure, and the converter's
> `ssm_dt_rank` mapping all confirm **48**. (Logged in the ledger.)

A DeltaNet layer's forward is: 4 input projections (QKV, Z/value-gate, B/beta, A/decay) →
causal conv1d (kernel 4) + SiLU → gated-delta recurrence (RMS norm + scale + softplus/exp
gating + sigmoid beta + recurrent state update) → output projection. Optimizing this is the MLX
domain's territory (fusing the 4 projections into 1 matmul, fusing conv1d+SiLU and the GDN
step into single Metal kernels). The exact recurrent **state shape** is settled in the next section.

## DeltaNet recurrent state (authoritative — contradiction resolved)

Each DeltaNet layer keeps **two** recurrent tensors (this is a "mamba-style" / linear-attention
state, not a KV cache): a small FIFO **conv state** and the matrix **ssm/recurrent state**.

### The recurrent (ssm) state shape

```
ssm_state  per layer:  (num_v_heads, head_v_dim, head_k_dim) = (48, 128, 128)
                       = 786,432 elements   (n_embd_s = ssm_d_state · ssm_d_inner = 128 · 6144)
conv_state per layer:  (conv_kernel_size - 1, conv_channels) = (3, 10240)
                       = 30,720 elements    (n_embd_r in llama.cpp)
```

Source-of-truth cross-check (three independent implementations agree on the same tensor):

| Implementation | Allocation site | Shape it allocates |
|---|---|---|
| MLX runtime | `fused_gdn.py:249` `mx.zeros((B, Hv, Dv, Dk))`; cache init `:903` `(1, num_v_heads, head_v_dim, head_k_dim)` | `(B, 48, 128, 128)` |
| llama.cpp graph | `qwen35.cpp:315` / `qwen3next.cpp:390` `ggml_reshape_4d(state, head_v_dim, head_v_dim, num_v_heads, n_seqs)` | `(128, 128, 48, n_seqs)` — uses `head_v_dim` twice because **`head_k_dim == head_v_dim == 128`** |
| vLLM | `selective_state_snapshot.py:124` comment `ssm_state (num_slots, num_v_heads, head_v_dim, head_k_dim)` | `(slots, 48, 128, 128)` |

The element count is anchored by llama.cpp `llama_hparams::n_embd_s() = ssm_d_state · ssm_d_inner
= 128 · 6144 = 786,432` (`llama-hparams.cpp:169`), which is exactly `head_k_dim · head_v_dim ·
num_v_heads = 128·128·48`. The axis *order* differs by framework (MLX/vLLM put heads first;
GGML's reshape lists `(head_v_dim, head_v_dim, num_v_heads, n_seqs)`) but the element count and
the three logical dims are identical.

### Per-request snapshot size (reconciling 77 MB vs 84 MB)

Per layer, BF16/FP16 (2 bytes): `786,432 · 2 = 1.573 MB` ssm + `30,720 · 2 = 0.061 MB` conv.
Over 48 DeltaNet layers:

| Quantity | BF16 | FP32 |
|---|---|---|
| ssm_state only | 1.573 MB/layer → **75.5 MB** | 3.146 MB/layer → **151 MB** |
| ssm + conv_state | **~78.4 MB** | ~154 MB |

This nails the two cited figures:

- **`modal-mtp/README.md` — state `[48, 128, 128]` per layer, ~77 MB/request: CORRECT.**
  `[48,128,128]` is exactly `(num_v_heads, head_v_dim, head_k_dim)`. 75.5 MB (ssm only) ≈ "~77 MB";
  including conv it is ~78 MB. The README's later note that `--mamba-ssm-cache-dtype float32`
  costs "150 MB per request (vs 77 MB at BF16)" matches the FP32 ssm figure (151 MB) precisely.
- **`recurrent-rollback` (README + `docs/TECHNIQUE.md`) — state `(14, 256, 256)`, 1.75 MB/layer,
  84 MB total: SHAPE IS WRONG, total is coincidentally close.** `14·256·256·2 = 1.835 MB/layer`
  (the doc rounds to 1.75) `× 48 = 88 MB` (rounded to 84). But the real shape is `(48, 128, 128)`,
  **not** `(14, 256, 256)`: the head count is 48 (not 14) and the per-head dims are 128 (not 256).
  The numbers happen to land near the true ~78 MB because `14·256·256 = 917,504` is within ~17 %
  of the true `48·128·128 = 786,432` — coincidence, not corroboration. The `(d_k × d_v)`
  *per-head* description in TECHNIQUE.md §1 is structurally right (a `head_k_dim × head_v_dim`
  matrix per head); only the *numbers* `(n_heads=14, d_k=256, d_v=256)` are wrong. The
  rollback **mechanism** (snapshot/restore the recurrent state, sub-millisecond copy at 546 GB/s)
  is unaffected by the shape error.

**Resolved canonical answer:** the GatedDeltaNet per-layer recurrent state is
`ssm_state = (num_v_heads=48, head_v_dim=128, head_k_dim=128)` (786,432 elements) plus
`conv_state = (conv_kernel_size-1=3, conv_channels=10240)` (30,720 elements). Per-request
snapshot ≈ **75.5 MB** (ssm only, BF16) / **~78 MB** (ssm+conv, BF16) / **~151 MB** (ssm FP32)
across the 48 DeltaNet layers. The modal `[48,128,128]` / 77 MB figures are authoritative;
the recurrent-rollback `(14,256,256)` / 84 MB shape is incorrect (corrected here).

> **Known with certainty** (pinned in source): the three logical dims `head_k_dim=128`,
> `head_v_dim=128`, `num_v_heads=48`, `num_k_heads=16`, `conv_kernel=4`, and the derived
> `n_embd_s=786,432`. **Assumed/derived, not read from a checkpoint config in-corpus**: the
> exact `linear_*` values come from vLLM helper configs + the converter mapping + the modal
> shape, all mutually consistent; no raw `config.json` for Qwen3.5-27B is present in the corpus
> to read `linear_num_value_heads` directly, but four independent implementations agree, so the
> values are treated as established.

## MRoPE (multi-axis RoPE)

Qwen3.5 uses **MRoPE** rather than plain RoPE. In the fork, the full-attention layers call
`ggml_rope_multi(n_rot, sections, …)` with a 4-element `hparams.rope_sections` array
(`LLM_KV_ROPE_DIMENSION_SECTIONS`, length 4 — `llama-model.cpp:1034`,
`std::array<int,4> rope_sections` in `llama-hparams.h:120`). This MRoPE detail bites the MTP
head: its position-0 marker tensor must be **length 4** for `ggml_rope_multi`, not length 1 for
`ggml_rope_ext` — see the "MRoPE inp_pos_zero" discovery in `tensor-layout-hf-to-gguf.md`.

### The rotary dimension count `n_rot = 64` (resolved)
Two RoPE quantities are distinct and both now pinned from source:

| Quantity | Value | Where it comes from |
|---|---|---|
| **`n_rot` (rope_dimension_count)** | **64** | Converter writes `add_rope_dimension_count(int(head_dim · partial_rotary_factor)) = int(256 · 0.25) = 64` (diff `01:103`). Loader reads it into `hparams.n_rot` via `LLM_KV_ROPE_DIMENSION_COUNT` (`llama-model.cpp:619`). This is the **total rotary dimension count** passed to both `ggml_rope_multi`/`ggml_rope_ext`. The MLX path computes the same `head_dim·0.25 = 64`. |
| **`rope_sections[4]`** | (from GGUF `rope_scaling.mrope_section`) | The MRoPE per-axis split, read from `LLM_KV_ROPE_DIMENSION_SECTIONS` into a length-4 array. In GGML `ggml_rope_multi`, the four `sections[]` entries partition the `n_rot/2 = 32` rotary *pairs* across the 4 MRoPE axes. |

> **`rope_dimensions = [12,12,12,12]` resolved.** The `qwen-mtp-tensors` README's
> `[12,12,12,12]` is **not** the authoritative section array and is treated as an unverified
> placeholder: it sums to 48, but a valid GGML MRoPE `sections[]` for this model must sum to
> `n_rot/2 = 32` (since `n_rot = 64`). The README likely paraphrased an HF `mrope_section` value
> from memory. The numbers the code actually uses and that ARE pinned: **`n_rot = 64`** (total
> rotary dims) and a **length-4 `rope_sections` array read from GGUF metadata**. The exact four
> section integers are not present in any in-corpus config/GGUF, so they remain unpinned — but
> nothing in the load/graph path depends on a specific split at position 0 (RoPE is identity at
> pos 0), only on the array being length 4. The earlier "open number" is now downgraded from a
> contradiction to "one source's `[12,12,12,12]` is unverified; `n_rot=64` is authoritative."

## Why this matters for speculative decoding (forward reference)

DeltaNet is **irreversible**: there is no "state at an intermediate position", only the state
after processing a whole batch. Standard speculative decoding assumes you can cheaply undo a
wrong-path decode (just don't advance the KV offset). On the 48 DeltaNet layers you can't —
you must **snapshot the recurrent state before speculation and restore it on rejection** (the
~78 MB BF16 snapshot quantified in the "DeltaNet recurrent state" section above). This
asymmetry (attention: trim KV offset; DeltaNet: snapshot/restore state) is the **split-recurrence
rollback** technique that recurs across the MLX, llama.cpp, and theory domains. It is the single
most consequential downstream fact of this architecture.

---

## Related

- [`mtp-head-anatomy.md`](mtp-head-anatomy.md) — the MTP head ("layer 64") bolted onto this backbone.
- [`tensor-layout-hf-to-gguf.md`](tensor-layout-hf-to-gguf.md) — the loader/recurrent-mask/MRoPE fixes that make this arch load in llama.cpp.
- [`../02-LLAMACPP/source/`](../02-LLAMACPP/source/) — `qwen35.cpp` / `qwen3next.cpp` GDN graph + state reshape (the source of truth for shapes here).
- [`../03-MLX/src/fused_gdn.py`](../03-MLX/src/fused_gdn.py) — the MLX runtime that allocates the same `(B,48,128,128)` recurrent state.
- [`../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md) — *why* DeltaNet's irreversibility makes rollback hard.
- [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) — the split-recurrence rollback this state shape enables.

---

### Provenance
- `qwen-mtp-tensors/README.md` (= `qwen-ops/llamacpp/tensor-mapping/README.md`) — layer table, eh_proj concat, MRoPE note
- `mlx-qwen-mtp/README.md` — "48 DeltaNet (recurrent) + 16 attention layers", split-recurrence framing
- `mlx-qwen-mtp/src/mtp_head.py` — canonical dims (5120/24/4/256/17408, rope_theta 1e5, partial_rotary 0.25)
- `qwen-ops/research/findings/modal-mtp-precision-divergence.md`, `modal-mtp/README.md` — "48 DeltaNet + 16 full attention in a strict 3:1 pattern"; state `[48,128,128]`/layer; 77 MB BF16 / 150 MB FP32 snapshot
- `qwen-ops/vllm/optimizations/selective_state_snapshot.py` — **authoritative ssm/conv state shapes** `(num_v_heads, head_v_dim, head_k_dim)` / `(conv_kernel-1, conv_dim)`; `linear_num_value_heads=48`, `head_dim=128`, `conv_kernel_size=4`
- `qwen-ops/vllm/optimizations/deltanet_adjuster.py` — `linear_num_key_heads=16`; placeholder `linear_num_value_heads=32` (NOT the real value — see ledger)
- `qwen-ops/vllm/microgreens/mtp_diversity_train.py`, `qwen-ops/vllm/optimizations/sibling_sequential.py` — config values (hidden 5120, inter 17408, heads 24/4, head_dim 256)
- `recurrent-rollback/README.md`, `recurrent-rollback/docs/TECHNIQUE.md` — rollback mechanism; cited state shape `(14,256,256)`/`(d_k×d_v)`, 1.75 MB/layer, 84 MB (shape corrected here — see "DeltaNet recurrent state")
- `MTP/03-MLX/src/fused_gdn.py` — **authoritative runtime state allocation** `mx.zeros((B, num_v_heads, head_v_dim, head_k_dim))` (`:249`, `:903`); conv `(B, conv_kernel-1, conv_dim)`
- `llama-mtp/src/models/qwen35.cpp` / `qwen3next.cpp` — GDN graph (`ssm_d_state`/`ssm_dt_rank`/`ssm_n_group`/`ssm_d_inner` mapping, state `ggml_reshape_4d`); `src/llama-hparams.cpp` `n_embd_s`/`n_embd_r`; `src/llama-model.cpp`, `src/llama-hparams.h` — `full_attention_interval`, recurrent mask, `LLM_TYPE_27B` detection, `rope_sections`
- `MTP/01-ARCHITECTURE/tensor-diffs/01-qwen35-tensor-load.diff` — converter ssm hparam mapping (`add_ssm_*` from `linear_*`)
- `qwen-mtp-research/docs/per-position-heads.md` — vocab math (5120·151936), GQA values (see Contradictions in the ledger)
