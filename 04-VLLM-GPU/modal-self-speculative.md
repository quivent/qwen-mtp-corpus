# Modal MTP — Self-Speculative Decoding ("Same Model, Two Speeds")

> Source of truth for the modal self-speculative idea. Collapses
> `modal-mtp/README.md`, `modal-mtp/docs/design.md`, and the byte-identical
> `qwen-ops/research/designs/modal-mtp-design.md`.

## The insight

Recap (full layer table + `full_attention_interval` live in
[`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md)):
Qwen3.5-27B interleaves **48 recurrent DeltaNet layers** (O(1)/token) and **16
full-attention layers** (O(N), KV-cache) in a strict **3:1 pattern** — three
DeltaNet, then one attention.

The modal observation is that **that 3:1 pattern is already a speculative-decode
schedule** — three tokens of cheap recurrence, one token of expensive
correction:

```
Layer  0: DeltaNet   ─┐
Layer  1: DeltaNet    │  Draft: fast recurrent path
Layer  2: DeltaNet   ─┘
Layer  3: Attention  ←── Verify: expensive correction
Layer  4: DeltaNet   ─┐
Layer  5: DeltaNet    │
Layer  6: DeltaNet   ─┘
Layer  7: Attention  ←──
... repeats 16 times ...
Layer 63: Attention  ←── final verification
```

The full-attention layers (indices **3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43,
47, 51, 55, 59, 63**) exist to periodically correct drift in the recurrent
state. Modal MTP makes that draft/verify rhythm explicit.

## Same model, two modes

Modal MTP is **not a proposer with its own weights** — it is a *model-runner
execution mode*. The single set of target-model weights runs two ways:

- **Draft mode** (`set_draft_mode(True)`): the 16 full-attention layers become
  **identity pass-throughs** (`self_attention_output.copy_(hidden_states)`).
  Only the 48 DeltaNet recurrences + 64 MLPs + norms execute. No KV-cache
  writes, no attention metadata needed. O(1) per token.
- **Verify mode** (`set_draft_mode(False)`): all 64 layers run normally. Full
  KV cache, exact attention.

No separate draft model. No additional weights. No memory overhead for a
drafter. The only synchronization primitive is **snapshot/restore of the
DeltaNet recurrent state**.

## Execution flow

```
For each generation step:

1. VERIFY: full 64-layer forward for current position
   → hidden states → sampler → token N
   → DeltaNet state at position N is authoritative

2. SNAPSHOT: save DeltaNet (conv + temporal) state for all 48 layers
   → ~77 MB/request, simple tensor clone

3. DRAFT: set_draft_mode(True)
   For k = 1..K:
     a. forward (DeltaNet + MLPs only, attention skipped)
     b. MTP head (or lm_head) → draft token N+k, with confidence gating
   set_draft_mode(False)

4. RESTORE: reset DeltaNet state from snapshot → back at position N

5. VERIFY: full forward on all K draft tokens (batched)
   → DeltaNet state recomputed correctly N+1..N+K
   → attention layers verify against full KV cache → accept/reject

6. DeltaNet state after verify is authoritative at last accepted position
   → no additional rollback needed
```

## Why it (should) work

The DeltaNet state matrix carries a compressed representation of the full
sequence. For short-horizon drafting (1–5 tokens) that representation is
sufficient — the model doesn't need attention corrections to stay coherent.
**Empirically validated at FP32/CPU: 100% draft accuracy at 50 tokens (5/5
prompts), and at FP16/CPU: 100% at 100 tokens (12/12 prompts).** The DeltaNet-only
path produces token-identical output to the full model on CPU.

> **But on GPU it diverges early.** This is the central plot twist and it is
> *not* a precision problem — see [`precision-divergence.md`](precision-divergence.md).
> The architecture is proven sound; the GPU divergence is a fused-CUDA-kernel
> numerical behavior difference.

## DeltaNet state snapshot details

Per DeltaNet layer per request (Qwen3.5-27B, TP=1):

| Component | Shape | Size |
|---|---|---|
| Conv state | `(kernel_size-1, conv_dim/tp)` = `(3, 10240)` | ~60 KB |
| Temporal (ssm) state | `(num_v_heads/tp, head_v_dim, head_k_dim)` = `(48, 128, 128)` | ~1.5 MB |
| **Per layer** | | **~1.56 MB** |

- 48 layers → **~77 MB per request** (design doc) / **~74.8 MB** (selective-snapshot
  geometry calc). For batch=32 → ~2.5 GB snapshot buffer (BF16); a **full**
  256-slot snapshot would be ~18.7 GB.
- `conv_dim = key_dim*2 + value_dim = 128*16*2 + 128*48 = 10240`.
- The snapshot is a synchronous memcpy — negligible vs the compute saved.

> **Snapshot was disabled in the shipped proposer.** In `modal_mtp.py` the
> snapshot was commented out ("too large, 40 GB for 256 requests") on the theory
> that the verify pass reprocesses tokens and overwrites DeltaNet state anyway.
> `selective_state_snapshot.py` is the proposed fix: snapshot only the *active
> batch* slots (O(batch_size), ~598 MB for 8 slots) rather than all 256.

## Speed analysis (GH200, BF16)

Per draft forward:
- **Skipped:** 16 attention layers × KV-cache lookup.
- **Still run:** 48 DeltaNet updates + 64 MLPs + norms (most of the compute).
- Net savings: **~25% of per-token compute** at short context (attention ≈ 25%
  of FLOPs), rising to **~60% at 32K context** (attention is O(N), DeltaNet O(1)).
- Plus: zero KV-cache memory pressure, no cache writes.

| Metric | Value |
|---|---|
| Full model (BF16) | 125 tok/s baseline |
| Full model + MTP5 | 193 tok/s (1.88×) |
| W4A16 abliterated + MTP5, single | 176.6 tok/s |
| W4A16 abliterated + MTP5, 32 concurrent | 1800+ tok/s |
| Draft-forward compute savings | ~25% (short) → ~60% (32K) |
| DeltaNet snapshot | ~77 MB/req, negligible copy |

> **Measured modal_mtp throughput was 3.2 tok/s** (`vllm-patches-benchmarks.md`)
> — i.e. catastrophically slow — because the draft path runs **without CUDA
> graphs** (see implementation status). The speed table above is the *potential*
> once draft-mode graphs are captured; it is not yet realized end-to-end.

## Partial attention skip (the fallback idea)

`test_partial_skip.py` sweeps how many of the 16 attention layers can be skipped
while keeping useful draft accuracy. Configs tested (indices to skip):

| Config | Skipped | Compute saved |
|---|---|---|
| skip_NONE (baseline) | [] | 0% |
| skip_middle_8 | [19,23,27,31,35,39,43,47] | 50% |
| skip_middle_12 | [11,15,19,23,27,31,35,39,43,47,51,55] | 75% |
| skip_all_but_first_last | all but {3,63} | 87.5% |
| skip_alternating | [7,15,23,31,39,47,55,63] | 50% |
| skip_ALL_16 (worst case) | all 16 | 100% |

The full 16 ALL_ATTN_INDICES are `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]`.
This is the harness for finding a partial-skip sweet spot if full skip proves
too lossy at FP16/GPU (it does — see precision-divergence).

## Implementation status

### Done (in `patches/modal-0{1,2,3}-*.patch` and `optimizations/`)
- `_skip_attention` flag on `Qwen3NextDecoderLayer` — identity pass-through for
  attention layers (`modal-01-skip-attention.patch` / `qwen3_next.patch`).
- `set_draft_mode(True/False)` on `Qwen3_5Model` — toggles all attention layers
  (`modal-02-draft-mode-and-state.patch` / `qwen3_5-shadow-state.patch`).
- `snapshot_deltanet_state()` / `restore_deltanet_state()` (clone/copy of conv +
  ssm state).
- `ModalMTPProposer` (`vllm/v1/spec_decode/modal_mtp.py`, see `modal_mtp.patch`
  and `optimizations/modal_mtp_proposer_reference.py`).
- 100% accuracy validation on CPU (FP32 5/5, FP16 12/12).

### To build
- vLLM model-runner draft execution loop (the 5 integration points are listed
  as comments at the bottom of `modal_mtp.py`).
- `modal_mtp` method registration in `SpeculativeConfig`.
- **CUDA graph capture for the draft-mode forward variant** (the missing piece —
  without it, modal_mtp runs eager and is 50× slower than baseline).
- Integration with adaptive chained MTP (confidence gating).
- GPU benchmarking of end-to-end throughput.

## The IndexError that blocks the draft loop

`modal-mtp/debug_indexerror.md` documents the concrete bug when wiring the draft
loop into a *compiled* model:

```
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
  at torch._dynamo.utils.call_size(x, 1)   # .size(1) on a 1D tensor
```

**Root cause:** `Qwen3_5ForConditionalGeneration` is a VL wrapper. During normal
inference the runner passes `input_ids=None` + 2D `inputs_embeds`; the inner
`@support_torch_compile`d `Qwen3_5Model` is AOT-compiled assuming `input_ids` is
always `None`. The draft forward passed `input_ids` as a 1D tensor, violating
that guard. Compounding factor: `_skip_attention` is a **data-dependent branch**
that `torch.compile` traced with `False` baked in — flipping it to `True` changes
the graph structure, incompatible with the captured artifact.

**Recommended fix (Option 1):** in the `set_forward_context` call, set
`skip_compiled=True` **and** `cudagraph_runtime_mode=CUDAGraphMode.NONE`, and pass
`input_ids=None` (since `inputs_embeds` is already provided). This bypasses both
the input-signature mismatch and the `_skip_attention` graph mismatch. The cost
is acceptable because draft forwards are lightweight. Later, a dedicated draft-mode
CUDA graph can be captured with `_skip_attention=True` baked in (the
`draft_mode=True` field on `BatchDescriptor` already exists for this).

> Note the shipped `modal_mtp.py` actually uses `CUDAGraphMode.PIECEWISE` with
> `draft_mode=True` in the `BatchDescriptor` (anticipating captured draft graphs),
> which is why it hits the compiled path and is slow/buggy until that graph is
> captured.

## Connection to prior work

Modal MTP unifies three threads:
- **Adaptive chained MTP** (the llama.cpp work, 1.99×) — confidence gating.
- **Recurrent-rollback** (the split-recurrence technique) — O(1) state rollback
  on rejection (see Agent E's track + `patches/recurrent-rollback.patch`).
- **The hybrid architecture itself** — DeltaNet drafts, attention verifies.

## Related

- [`precision-divergence.md`](precision-divergence.md) — why the proven-on-CPU draft path diverges on GPU (kernel numerics, not precision).
- [`../01-ARCHITECTURE/qwen35-hybrid-architecture.md`](../01-ARCHITECTURE/qwen35-hybrid-architecture.md) — the 3:1 hybrid layer table + DeltaNet recurrent state (canonical).
- [`../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md`](../05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md) — the structural "architecture *is* a draft/verify schedule" insight (theory side).
- [`../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md`](../05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md) — O(1) DeltaNet state rollback on rejection (the split-recurrence technique).
- [`vllm-patches-and-strategies.md`](vllm-patches-and-strategies.md) — the `modal_mtp` patches + the 3.2 tok/s (no-CUDA-graphs) measurement.
- [`../00-OVERVIEW/glossary.md`](../00-OVERVIEW/glossary.md) — DeltaNet/GDN, draft mode, self-speculative.

---
*Provenance: `modal-mtp/README.md`, `modal-mtp/docs/design.md` (≡
`qwen-ops/research/designs/modal-mtp-design.md`, collapsed),
`modal-mtp/modal_mtp.py`, `modal-mtp/debug_indexerror.md`,
`modal-mtp/test_partial_skip.py`, `qwen-ops/vllm/optimizations/{modal_mtp,selective_state_snapshot}.py`.*
