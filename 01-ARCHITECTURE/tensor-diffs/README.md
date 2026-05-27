# Tensor-mapping diffs (verbatim)

The two commits that make a converted Qwen3.5-27B MTP GGUF load and run end-to-end. Copied
verbatim from `qwen-mtp-tensors/diffs/` (byte-identical copies also existed at
`qwen-ops/llamacpp/tensor-mapping/`). Distilled explanation: `../tensor-layout-hf-to-gguf.md`.

| File | Commit | What it touches |
|---|---|---|
| `01-qwen35-tensor-load.diff` | `53075d24cc96fbfe629c6376e55dd81c57641f53` (Apr 8 2026, 17:16) | converter (`convert_hf_to_gguf.py` `Qwen3NextModel.__init__` + `modify_tensors`), `gguf-py/gguf/constants.py`, classifier (`src/llama-arch.cpp`), loader (`src/llama-model.cpp`), graph RoPE fix (`src/models/qwen35.cpp`), `src/llama-context.cpp`. Carries the canonical fix commit message enumerating Fixes 1–6. |
| `02-qwen35-graph-tensors.diff` | `83babcae7a23045dd5c2c7ab4ddb9e8f3c3145d0` (Apr 8 2026, 16:39) | adds `build_mtp_head` + the `LLM_GRAPH_TYPE_MTP` branch to `src/models/qwen35.cpp`. **Earlier** of the two — uses `ggml_rope_ext` + length-1 `inp_pos_zero`; superseded by diff 01's switch to `ggml_rope_multi` + length-4. |

Read 02 first (the graph builder it introduces), then 01 (the load/convert/classify fixes that
make it loadable, plus the RoPE correction that supersedes 02's pos-zero handling).
