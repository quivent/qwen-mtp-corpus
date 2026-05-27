# Glossary — the vocabulary of this corpus

A newcomer's dictionary for the MTP-speculative-decoding-for-Qwen3.5-27B corpus. Each
entry is 1–3 sentences plus a **see** pointer to the canonical doc that owns the topic.
Terms are alphabetized (ignoring case and leading symbols).

## Read these first (orientation path)

1. [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md)
   — what the model *is* (the 64-layer 3:1 hybrid, dims, recurrent state). Everything
   depends on this.
2. [`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md)
   — the MTP head ("layer 64"), the draft model every result runs on.
3. [`../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md)
   — *why* spec-decode is hard on this model (the recurrent-state irreversibility).
4. [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md)
   — the split-recurrence rollback that makes it work.
5. The three platform tracks: [`../02-LLAMACPP/`](../02-LLAMACPP/),
   [`../03-MLX/`](../03-MLX/), [`../04-VLLM-GPU/`](../04-VLLM-GPU/).

---

## Terms

### acceptance rate
The fraction of drafted tokens that the target model confirms during verification. The
stock single MTP head accepts ~79% (MLX, position +1); acceptance **decays geometrically
with chain depth** (stock head: 87% → 68% → 54% → 39% → 28% → 21% → 16% over 7 chained
positions), which is why short chains (K=2) win. See
[`../04-VLLM-GPU/README.md`](../04-VLLM-GPU/README.md) (per-position table) and
[`../03-MLX/mtp-self-speculative-mlx.md`](../03-MLX/mtp-self-speculative-mlx.md).

### ANE (Apple Neural Engine)
A separate accelerator on Apple Silicon with **its own memory bus** (zero GPU-bandwidth
interference). The MTP head was confirmed to run on it via CoreML at **2.24 ms median
latency**, enabling free drafting while the GPU runs the main forward. See
[`../03-MLX/bandwidth-and-dispatch.md`](../03-MLX/bandwidth-and-dispatch.md) §Research
frontiers.

### AR kernel (autoregressive) / chunking — see chunking vs AR kernel

### AWQ (Activation-aware Weight Quantization)
A 4-bit post-training weight-quantization method. For MTP serving on RTX 5090 it wins
batch throughput but yields lower MTP draft acceptance (31%) than GPTQ. See
[`../04-VLLM-GPU/quantization.md`](../04-VLLM-GPU/quantization.md).

### bandwidth wall / bandwidth-bound
Decode is **memory-bandwidth-bound**: every token reads all the weights once, so latency
is `weight_bytes / bandwidth`, independent of compute. On M4 Max that is 13.7 GB / 546 GB/s
= a 25.1 ms hard floor. The only lever is reading less data or extracting more tokens per
read (spec decode). See
[`../03-MLX/bandwidth-and-dispatch.md`](../03-MLX/bandwidth-and-dispatch.md).

### cache-bookkeeping bug
The one-line host-side bug (`id_last = corr` after a rollback re-decode) that double-wrote
the correction token, shifting every subsequent cache slot and corrupting **every**
speculative measurement for a whole llama.cpp session. Fixed by infra patch 11. See
[`../02-LLAMACPP/the-bug.md`](../02-LLAMACPP/the-bug.md).

### cascade heads / cascade-MTP
A training design using depth-specific MTP heads to hold acceptance at larger chain depth
(vs one head reused at every depth). Tried in vLLM (cascade, depth-trained) — underperformed
due to training mismatch. See
[`../05-THEORY-AND-DESIGNS/cascade-mtp-training.md`](../05-THEORY-AND-DESIGNS/cascade-mtp-training.md).

### chain depth (K) — see K / chain depth

### chunking vs AR kernel
DeltaNet's recurrence has two numerically-equivalent kernels: a **chunking** (parallel-scan)
kernel for T≥2 verify batches and an **autoregressive** (sequential) kernel for T=1. They
diverge by a few ulps in FP16 — hypothesized as the spec-decode garbage-output cause but
proven a red herring. See
[`../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md) §4.

### confidence gating
Gating the speculative chain length by the head's own top-1 probability: chain another step
only when `max(softmax) ≥ threshold` (e.g. `MTP_CHAIN_THRESH=0.85`). The cheap, zero-training
ingredient behind the headline wins. See
[`../02-LLAMACPP/the-recipe.md`](../02-LLAMACPP/the-recipe.md).

### conv_state
The small FIFO convolution state of a DeltaNet layer: shape `(conv_kernel_size−1=3,
conv_channels=10240)` = 30,720 elements per layer, snapshotted alongside the ssm_state on
rollback. See [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) §DeltaNet recurrent state.

### DeltaNet / Gated DeltaNet (GDN)
The linear-attention **recurrent** layer type (gated delta rule) that makes up 48 of
Qwen3.5-27B's 64 layers. O(1) per token with a fixed-size recurrent state (no growing KV
cache) — efficient, but its nonlinear state update is **irreversible**, which is the source
of all the rollback difficulty. See
[`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md).

### draft / verify
The two phases of speculative decoding: cheaply **draft** several candidate tokens, then
**verify** them in one batched forward pass through the target model, committing the
matching prefix and rolling back the rest. See
[`../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md) §1.

### EAGLE
An external speculative-decoding head architecture referenced as a design comparison in the
vLLM GPU-designs analysis (PR descriptions). See
[`../04-VLLM-GPU/gpu-designs.md`](../04-VLLM-GPU/gpu-designs.md).

### eh_proj
The MTP head's fused projection (named **e**mbed-**h**idden) that maps the concatenated
`[embedding ‖ hidden]` `2·5120` vector back down to `5120`. **Critical:** the concat order
is embed-first, hidden-second — getting it backwards silently produces garbage. See
[`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md).

### full_attention_interval
The hyperparameter (`= 4`) that implements the 3:1 pattern: a layer is DeltaNet when
`(i+1) % 4 != 0`, i.e. every 4th layer is full attention (16 of 64). See
[`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md).

### fused kernels
Hand-written Metal kernels that combine several ops into one dispatch (`fused_gdn_step`,
`fused_conv1d_silu`, `rms_norm_qmv`). On MLX most hand-fusion *lost* to `mx.compile`; the
one clear win was fusing rms_norm into the matmul (saved 2.0 ms). See
[`../03-MLX/kernel-fusion.md`](../03-MLX/kernel-fusion.md).

### GDN — see DeltaNet / Gated DeltaNet

### GGUF
The llama.cpp model file format. The HF→GGUF converter **silently strips** the MTP head
weights unless patched, and uses `ssm_*` metadata names for the DeltaNet hyperparameters.
See [`../01-ARCHITECTURE/tensor-layout-hf-to-gguf.md`](../01-ARCHITECTURE/tensor-layout-hf-to-gguf.md).

### GH200 — see hardware (platforms & hardware)

### GPTQ
A 4-bit post-training quantization method. For MTP serving it is recommended over AWQ
because it keeps the MTP head's draft quality higher (51% vs 31% acceptance on RTX 5090).
See [`../04-VLLM-GPU/quantization.md`](../04-VLLM-GPU/quantization.md).

### hardware (platforms & hardware)
The three hardware/runtime combinations used. **Platforms:** **llama.cpp** (C++ fork, GGUF,
the 1.99× recipe), **MLX** (Apple Silicon Python, 51.1 tok/s), **vLLM** (datacenter GPU
serving, stock MTP spec). **Hardware:** **M4 Max** (Apple, 16-core GPU, 128 GB unified,
546 GB/s), **GH200** (NVIDIA 480GB, H100 die, 96 GB HBM3e), **RTX 5090** (32 GB, NixOS).
See [`../_meta/CONVENTIONS.md`](../_meta/CONVENTIONS.md) and each domain README
([`../02-LLAMACPP/`](../02-LLAMACPP/), [`../03-MLX/`](../03-MLX/),
[`../04-VLLM-GPU/`](../04-VLLM-GPU/)).

### hybrid architecture
The defining shape of Qwen3.5-27B: 48 DeltaNet (recurrent) + 16 full-attention layers
interleaved 3:1, plus one MTP head. The mix keeps most layers cheap and recurrent while
periodically restoring global recall — and is the root cause of the rollback challenge. See
[`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md).

### K / chain depth
The number of speculative tokens drafted per cycle (also `MTP_CHAIN_KMAX`,
`num_speculative_tokens`). K=2 is the local optimum for the single chained head; vLLM's
sweet spot is spec=5. Higher K loses to geometric acceptance decay. See
[`../02-LLAMACPP/the-recipe.md`](../02-LLAMACPP/the-recipe.md).

### KV cache
The append-only per-token key/value store of full-attention layers. Rollback on a KV cache
is trivial (`KV = KV[:i+1]`) — the property spec decode assumes, and exactly what DeltaNet's
recurrent state lacks. See
[`../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md) §1.

### llama.cpp — see hardware (platforms & hardware)

### M4 Max / GH200 / RTX 5090 — see hardware (platforms & hardware)

### megakernel
A research frontier: collapsing the ~500+ per-token Metal dispatches into one persistent
dispatch with software-managed weight streaming (est. 10–15% gain). Requires reimplementing
the forward outside MLX. See
[`../03-MLX/bandwidth-and-dispatch.md`](../03-MLX/bandwidth-and-dispatch.md) §Research
frontiers.

### MLX — see hardware (platforms & hardware)

### modal drafting / self-speculative — see self-speculative / modal drafting

### MRoPE (multi-axis RoPE)
The position encoding Qwen3.5 uses instead of plain RoPE: `ggml_rope_multi` with a length-4
`rope_sections` array. The length-4 requirement bites the MTP head's position-0 marker
tensor. See [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) §MRoPE.

### MTP / Multi-Token Prediction
Predicting more than the next single token per forward pass; here, used as the **draft**
mechanism for speculative decoding. Qwen3.5-27B ships a trained MTP head that predicts +1
from the main model's hidden state, and the corpus extends it (chaining, parallel drafts).
See [`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md).

### MTP head
The single extra transformer block stored as **"layer 64"** (15 weight tensors): RMS-norm +
concat `[embed,hidden]` → `eh_proj` → gated self-attention → SiLU MLP → shared lm_head. It
is the draft model every result in the corpus runs on; every framework strips it on load,
so re-attaching it was job #1 everywhere. See
[`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md).

### mx.compile
MLX's graph compiler/dispatch scheduler. It already fuses elementwise op chains (so
hand-fusing those is wasted effort) but **cannot fuse across kernel types** (norm+matmul) —
that barrier is the main remaining optimization target. See
[`../03-MLX/kernel-fusion.md`](../03-MLX/kernel-fusion.md) and
[`../03-MLX/bandwidth-and-dispatch.md`](../03-MLX/bandwidth-and-dispatch.md).

### n_rot
The total rotary-dimension count, `= int(head_dim·partial_rotary_factor) = int(256·0.25) =
64`. Distinct from the length-4 `rope_sections` MRoPE split (which sums to `n_rot/2 = 32`
pairs). See [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) §n_rot.

### nextn_predict_layers
The GGUF/llama.cpp hyperparameter counting MTP predict layers (= 1 for Qwen3.5-27B). It is
how `65 layers on disk` resolves to the 64-layer `LLM_TYPE_27B` (`n_layer −
nextn_predict_layers == 64`), and the MTP layer index is `il_mtp = n_layer −
nextn_predict_layers = 64`. See
[`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md).

### O(1)-per-token ceiling
The framing that spec decode is **not** capped at 2× — with N drafts at 100% acceptance one
T=(N+1) forward reads the weights once and yields N+1 tokens (e.g. N=8 → 213 tok/s on M4
Max). The real limit is draft accuracy at depth, set by bandwidth, not a fixed multiple. See
[`../05-THEORY-AND-DESIGNS/research-frontiers.md`](../05-THEORY-AND-DESIGNS/research-frontiers.md) §0.

### per-position heads
A DeepSeek-V3-style design: replace the single MTP head with N independent heads, each
predicting a fixed relative offset from the shared main hidden. Design-only here (no GPU
spent); the resolved truth is that the shipped wins came from chained-single-head, not this.
See [`../05-THEORY-AND-DESIGNS/per-position-heads-design.md`](../05-THEORY-AND-DESIGNS/per-position-heads-design.md).

### quantization (Q4_K_M / W4A16 / AWQ / GPTQ)
4-bit weight quantization schemes used per platform: **Q4_K_M** (llama.cpp/GGUF, ~13.7 GB),
**W4A16** (4-bit weights, 16-bit activations; vLLM), realized as **AWQ** or **GPTQ**. See
[`../04-VLLM-GPU/quantization.md`](../04-VLLM-GPU/quantization.md) and
[`../03-MLX/bandwidth-and-dispatch.md`](../03-MLX/bandwidth-and-dispatch.md) (weight budget).

### the recipe (1.99×)
The chained-recurrent-MTP + confidence-gating recipe ported from MLX `stacked_v2` to
llama.cpp: two env vars (`MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`) for **1.99× over K=1
vanilla** with coherent output. See
[`../02-LLAMACPP/the-recipe.md`](../02-LLAMACPP/the-recipe.md) (canonical home).

### recurrent state (ssm_state + conv_state)
The fixed-size state each DeltaNet layer carries (vs a growing KV cache). Adjudicated shape:
per-layer **`ssm_state = (num_v_heads=48, head_v_dim=128, head_k_dim=128)` = 786,432
elements** + **`conv_state = (3, 10240)`**, for **~77 MB/request in BF16** across the 48
DeltaNet layers. Irreversible — hence snapshot/restore. See
[`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) §DeltaNet recurrent state.

### self-speculative / modal drafting
Using the target model as its own drafter — no separate draft model. The **modal** insight:
the 3:1 pattern *is* a draft/verify schedule, so "draft mode" skips the 16 attention layers
(identity pass-through) and "verify mode" runs all 64. Proven 100% accurate on CPU; diverges
on GPU from a fused-kernel numerical difference. See
[`../04-VLLM-GPU/modal-self-speculative.md`](../04-VLLM-GPU/modal-self-speculative.md).

### shared_head
The MTP head shares the main model's embedding and lm_head (`mtp_use_dedicated_embeddings:
false`) rather than carrying its own — which is why the head is 15 tensors not 17, and why
its added params are small. In llama.cpp the dedicated tensors load as `TENSOR_NOT_REQUIRED`
and the graph falls back to `model.output`. See
[`../01-ARCHITECTURE/mtp-head-anatomy.md`](../01-ARCHITECTURE/mtp-head-anatomy.md).

### snapshot / restore
The platform mechanism for saving the DeltaNet recurrent state before speculation and
restoring it on rejection (vLLM/modal: ~77 MB/request; llama.cpp:
`llama_memory_seq_force_recurrent_pos`). The naive form (with redo) is what split-recurrence
rollback improves on. See
[`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) §2.

### speculative decoding
Accelerating autoregressive inference by drafting K tokens cheaply and verifying them in one
batched forward; on this bandwidth-bound model it is the *only* real lever (it extracts more
tokens per weight read). See
[`../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md).

### split-recurrence rollback
The corpus's signature technique: batch the DeltaNet projections at T=N but split **only**
the recurrence into per-token steps, saving a zero-copy state reference after each (arrays
are immutable). On rejection, restore the ref and trim the KV — **no redo**, ~6 ms/step
saved. See
[`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) (canonical home).

### ssm_state — see recurrent state

### the 3:1 pattern
The strict interleave of three DeltaNet layers then one full-attention layer, repeating to
64 layers (48 DeltaNet + 16 attention). Implemented as `full_attention_interval = 4`; read
structurally it is itself a draft/verify rhythm (3 cheap recurrent tokens, 1 expensive
correction). See
[`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md).

### vLLM — see hardware (platforms & hardware)

---

## Related docs

- [`the-big-picture.md`](the-big-picture.md) — the whole arc as one narrative (terms in context).
- [`results.md`](results.md) — the single authoritative benchmark table.
- [`corpus-map.md`](corpus-map.md) — where everything came from; "where do I find X?".
- [`../README.md`](../README.md) — the corpus front door.

---

*Sources: distilled from the domain READMEs (01–06), the architecture docs
(`qwen35-hybrid-architecture.md`, `mtp-head-anatomy.md`), the theory docs
(`speculative-decoding-on-hybrid-models.md`, `recurrent-rollback-technique.md`,
`per-position-heads-design.md`, `research-frontiers.md`), and the platform canonical docs
(`the-recipe.md`, `the-bug.md`, `bandwidth-and-dispatch.md`, `modal-self-speculative.md`,
`quantization.md`). Definitions are pointers, not new claims — each canonical doc owns its
numbers.*
