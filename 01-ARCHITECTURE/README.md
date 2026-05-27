# 01 — Architecture: Qwen3.5-27B, the MTP head, and its tensors

> The foundational domain. What the model *is* (a 64-layer 3:1 DeltaNet/full-attention
> hybrid), what the shipped MTP head *is* ("layer 64", the drafter every result runs on),
> and exactly where every weight lives across the HuggingFace checkpoint, the GGUF file,
> and the llama.cpp layer struct. Everything in the platform domains (02–04) and the theory
> domain (05) depends on the facts established here.

## Reading path

Read these in order — each builds on the last:

1. [`qwen35-hybrid-architecture.md`](qwen35-hybrid-architecture.md) — what the model *is*:
   the 64-layer 3:1 hybrid, core dims, MRoPE, and the DeltaNet recurrent state. Start here.
2. [`mtp-head-anatomy.md`](mtp-head-anatomy.md) — the MTP head ("layer 64"), its 15 tensors,
   the `eh_proj` embed-first concat, and the forward pass.
3. [`tensor-layout-hf-to-gguf.md`](tensor-layout-hf-to-gguf.md) — the HF→GGUF→llama.cpp name
   map and the five-part converter/loader/classifier fix needed before a real GGUF loads.
4. [`weight-extraction.md`](weight-extraction.md) — pulling the MTP head out of the raw
   safetensors shards (the MLX/vLLM route), including the RMSNorm `+1.0` shift gotcha.

**Subfolders** (copied artifacts): [`extract/`](extract/) — the two MTP extraction scripts;
[`tensor-diffs/`](tensor-diffs/) — the verbatim converter/loader (`01`) and graph (`02`)
diffs referenced throughout `tensor-layout-hf-to-gguf.md`.

## What this domain owns authoritatively

The **DeltaNet recurrent-state shape** is adjudicated here (and only here): per DeltaNet
layer, `ssm_state = (num_v_heads=48, head_v_dim=128, head_k_dim=128)` = 786,432 elements,
plus `conv_state = (conv_kernel_size−1=3, conv_channels=10240)` = 30,720 elements, for a
per-request snapshot of ~78 MB (ssm+conv, BF16) across the 48 DeltaNet layers. This is the
fact the rollback work in every other domain rests on — see
[`qwen35-hybrid-architecture.md`](qwen35-hybrid-architecture.md) §DeltaNet recurrent state
for the three-implementation cross-check and the correction of the `(14,256,256)`/84 MB
figure cited elsewhere.
