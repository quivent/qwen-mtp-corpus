# Inventory — Agent E (Theory & Forward Designs)

Domain: `MTP/05-THEORY-AND-DESIGNS/`. Cross-cutting theory of spec-decode on hybrid
attention/recurrent models, the generalizable split-recurrence rollback technique, the
per-position MTP heads design, MTP-head training, cascade training, and the open research
frontiers. Platform-specific implementation lives with Agents A–D and is cross-linked.

## Source files consumed → MTP destination

| Source file | Destination | Status |
|---|---|---|
| `recurrent-rollback/README.md` | `recurrent-rollback-technique.md` (+ theory doc §3) | distilled |
| `recurrent-rollback/docs/TECHNIQUE.md` (499 lines) | `recurrent-rollback-technique.md` (math, proofs, cost model, dispatch barriers, limitations); `speculative-decoding-on-hybrid-models.md` §2,§8; `research-frontiers.md` §0,§A.3,§A.4,§A.5,§A.6 | distilled (lossless) |
| `recurrent-rollback/src/__init__.py` | `recurrent-rollback-src/src/__init__.py` | copied |
| `recurrent-rollback/src/split_recurrence.py` | `recurrent-rollback-src/src/split_recurrence.py` | copied |
| `recurrent-rollback/src/delta_net_rollback.py` | `recurrent-rollback-src/src/delta_net_rollback.py` | copied |
| `recurrent-rollback/examples/__init__.py` (empty, 0 B) | `recurrent-rollback-src/examples/__init__.py` | copied |
| `recurrent-rollback/examples/mtp_speculative_decode.py` | `recurrent-rollback-src/examples/mtp_speculative_decode.py` | copied |
| `recurrent-rollback/examples/simple_example.py` | `recurrent-rollback-src/examples/simple_example.py` | copied |
| `qwen-mtp-research/docs/per-position-heads.md` | `per-position-heads-design.md` | distilled |
| `qwen-ops/research/designs/per-position-heads.md` | `per-position-heads-design.md` | **duplicate-of** `qwen-mtp-research/docs/per-position-heads.md` (byte-identical, verified `diff` → IDENTICAL) — collapsed |
| `qwen-mtp-research/README.md` | `per-position-heads-design.md` (§0 MLX truth, Phase-0 gate, $165, 2.23× ceiling), `speculative-decoding-on-hybrid-models.md` (§4 chunking, §5 seq_cp, §7 single-head), `research-frontiers.md` | distilled (MTP-relevant sections; the-bug story belongs to Agent B, cross-linked) |
| `qwen-ops/research/designs/CASCADE-MTP-TRAINING.md` | `cascade-mtp-training.md` (+ `training-mtp-heads.md` §5 summary) | distilled (lossless) |
| `qwen-ops/training/build_training_data.py` | `training/build_training_data.py` | copied (+ distilled into `training-mtp-heads.md` §3) |
| `qwen-ops/training/train_per_position_heads.py` | `training/train_per_position_heads.py` | copied (+ distilled into `training-mtp-heads.md` §2,§4.1,§4.3) |
| `qwen-ops/research/findings/RESEARCH_FRONTIERS.md` | `research-frontiers.md` | distilled (lossless) — canonical home for architecture/training/general directions |
| `qwen-inference-lab/docs/RESEARCH_FRONTIERS.md` | `research-frontiers.md` | **duplicate-of** `qwen-ops/research/findings/RESEARCH_FRONTIERS.md` (byte-identical, verified `diff` → IDENTICAL) — collapsed |
| `modal-mtp/README.md` (§Insight, §3:1 Pattern, §architecture) | `speculative-decoding-on-hybrid-models.md` §6 (architecture-IS-the-schedule) | distilled (cross-cutting theory only; modal vLLM impl + precision saga = Agent D) |
| `qwen-ops/README.md` §"Key Architectural Insight" | `speculative-decoding-on-hybrid-models.md` §1,§2,§4 | distilled (theory framing; the-bug + quickstart = Agent B) |
| `qwen-ops/research/findings/mlx-reference.md` | `per-position-heads-design.md` §0 (resolved truth, technique hierarchy table) | distilled (architecture/design content; MLX runtime = Agent C) |

## MTP files produced

| File | Role |
|---|---|
| `05-THEORY-AND-DESIGNS/speculative-decoding-on-hybrid-models.md` | Core theory: irreversibility, rollback problem, chunking-vs-AR, seq_cp alias, architecture-IS-the-schedule (3:1), single-head difficulty, O(1) ceiling, principles checklist |
| `05-THEORY-AND-DESIGNS/recurrent-rollback-technique.md` | Canonical split-recurrence rollback: problem, checkpoint+redo baseline (~7ms/step), the solution (zero-copy refs, ~6ms net / +2 tok/s), proofs, cost model, applicability table, limitations, platform pointers, citation |
| `05-THEORY-AND-DESIGNS/per-position-heads-design.md` | DeepSeek-V3 per-position heads design + **the resolved truth** (single head + chained recurrence) + Phase-0 kill-gate |
| `05-THEORY-AND-DESIGNS/training-mtp-heads.md` | Distill-from-logits training, freeze-main, accept-rate ladder (79%→85-90%→95%), cost, cascade-corrective summary |
| `05-THEORY-AND-DESIGNS/cascade-mtp-training.md` | Depth-specific cascade heads (CE+MSE, serving-checkpoint hidden states, what-failed) |
| `05-THEORY-AND-DESIGNS/research-frontiers.md` | Open directions; O(1)/token ceiling, 213 tok/s @N=8, parallel MTP, ANE, entropy debunk, dead ends |
| `05-THEORY-AND-DESIGNS/training/README.md` | origins of copied training scripts |
| `05-THEORY-AND-DESIGNS/recurrent-rollback-src/README.md` | origins of copied rollback source |

**Artifacts copied:** 2 training scripts + 6 rollback source/example files = **8 files**.
**Duplications collapsed:** 2 (per-position-heads.md, RESEARCH_FRONTIERS.md).

---

## Contradictions & evolving threads

### #1 (PRIMARY — owned & resolved here): "per-position trained heads" vs "single head + chained recurrence"

- **Earlier belief (in `qwen-mtp-research/docs/per-position-heads.md` framing and elsewhere):**
  the MLX 1.68× speedup proved that *trained per-position MTP heads* (DeepSeek-V3 style)
  work, so we should train N=4 heads.
- **Resolved truth (`qwen-ops/research/findings/mlx-reference.md`, confirmed against
  `~/mlx-fork/mtp_weights_vanilla.safetensors`):** the shipped Qwen3.5 checkpoint contains
  **exactly ONE** MTP block (`mtp.layers.0.*`; no `mtp.layers.1+`). The MLX 1.68×/1.73× did
  **NOT** come from trained per-position heads. It came from `stacked_v2.py`: **(1) chained
  recurrent application of the single head** (feed the head's own output hidden as the next
  step's `prev_hidden`), **(2) a ~0.8B companion draft model** for position 3, **(3)
  confidence gating** (`max(softmax) ≥ threshold`), **(4) zero training cost**. The same
  recipe in llama.cpp (components 1+3) delivered **1.99× over K=1 vanilla**
  (`MTP_CHAIN_KMAX=2 MTP_CHAIN_THRESH=0.85`).
- **Where resolved:** `per-position-heads-design.md` §0 presents the correction prominently
  *before* the design. Per-position heads are retained as a valid *alternative* path, gated
  by Phase-0.
- **Phase-0 kill-gate (also owned here):** if `head_fwd ≈ main_fwd`, per-position heads
  CANNOT win regardless of accept rate — because fixed per-pass overhead (KV bookkeeping,
  DeltaNet state, graph alloc) dominates, not head FLOPs. Evidence: K=1 MTP spec = 0.43× of
  plain (7.64 vs 17.90 tok/s). Measure `build_mtp_head` wall time before any GPU spend.

### #2: Entropy-coded weights — claimed 3.27× then debunked

- **Claim:** 4-bit weights have ~1.1–1.5 bits Shannon entropy → rANS gives 3.3× compression,
  9.3ms bandwidth floor, 80–90 tok/s prize (ECQ measured 3.27× on M3 Pro).
- **Debunk (2026-04-05, same doc):** measured on 1.4B Qwen3.5-4bit values, entropy = **3.72
  bits**, compression = **1.08× (7% savings)**. The 1.1-bit figure was a different
  quantization format with learned codebooks. rANS on standard 4-bit is not worth a custom
  kernel.
- **Resolved in:** `research-frontiers.md` §B.1 + §C caveat — both the claim and the debunk
  are preserved; the combined-projection table is flagged as depending on the debunked
  number.

### #3: MTP-head-on-ANE latency — estimate vs confirmed

- Early estimate: 0.037ms compute, 10–22ms weight load. **Confirmed (2026-04-05):** 2.24 ms
  median latency, 424.7M params / 849.5 MB, exact MLX output match. Both kept in
  `research-frontiers.md` §B.2. (Detail owned by Agent C.)

### #4: Net-savings figure for split-recurrence rollback — ~6ms vs 5.3ms

- README says **~6ms net / +2 tok/s** (per-token dispatch accounting; overhead ~1ms /
  redo ~7ms). TECHNIQUE.md detailed model says **5.3ms net** (overhead 3.4ms over 48 layers
  / redo 8.7ms), and elsewhere ~7.85ms (0.85ms overhead / 8.7ms redo). Not a true
  contradiction — different per-layer dispatch accounting; all land in **~5–8ms net saved**.
  Documented as such in `recurrent-rollback-technique.md` §4.2.

### #5: DeltaNet state shape — `[48,128,128]` (modal) vs `(14,256,256)` / d_k=d_v=256, n_heads=14 (rollback)

- `modal-mtp/README.md` cites a per-layer state matrix `[48, 128, 128]`; `recurrent-rollback`
  src/docs cite `(n_heads=14, d_k=256, d_v=256)` → 1.75 MB/layer, 84 MB total over 48 layers.
- **Not adjudicated here** — exact tensor shapes are Agent A's domain
  (`../01-ARCHITECTURE/`). Theory docs carry each number as its source stated it and
  cross-link A for the authoritative shape. Flagged for the orchestrator.

### #6: Layer-count phrasing — "48 DeltaNet + 16 attention = 64" vs forward-pass "64 layers"

- Consistent across sources (48 recurrent + 16 attention = 64). The rollback cost table once
  says "full forward pass (64 layers)" while the split overhead is computed over "48 DeltaNet
  layers" — consistent (only recurrent layers are split). No contradiction; noted for
  clarity.

### Gaps / deferred

- The actual `stacked_v2.py`, `adaptive_mtp.py`, `collect_hidden_states.py`,
  `cascade_mtp_corrective.py` source files were referenced but not present in the consumed
  corpus paths (they live at `~/optimizations/...` outside the repos). Described from the
  docs; cross-linked to Agents C (MLX runtime) and D (vLLM/GH200 cascade).
- The-bug story, the-recipe operational detail, and llama.cpp env-var mechanics are
  cross-linked to Agent B, not duplicated here.
