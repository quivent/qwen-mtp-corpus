# llama.cpp server logs — raw captures (copied verbatim)

Two `llama-server` run logs from the M4 Max inference lab. **Copied byte-for-byte**
(verified `md5sum`-identical) from `qwen-inference-lab/logs/`. They are the only
end-to-end **server-mode** captures of the Qwen3.5-27B MTP path; the numbers folded
out of them live in [`../benchmarks.md`](../benchmarks.md).

| File | Origin | What it captures |
|---|---|---|
| `llama_cpp_server.log` | `qwen-inference-lab/logs/llama_cpp_server.log` (418 KB, 3628 lines) | **MTP + 0.8B companion draft** server run: 27B target + Qwen3.5-0.8B-Q8_0 draft model, MTP head auto-enabled, 4 slots, several `/completion` requests. |
| `llama_cpp_server_nomtp.log` | `qwen-inference-lab/logs/llama_cpp_server_nomtp.log` (61 KB, 864 lines) | **No-draft baseline** server run: 27B only, **no draft model loaded**, MTP head present but fires 0 drafts for the single request → pure-decode ground truth. |

> The lab README only advertised `llama_cpp_server.log` ("Server output showing
> MTP+draft performance"). `llama_cpp_server_nomtp.log` was an unadvertised sibling —
> the matched no-MTP/no-draft baseline — captured here so the comparison is complete.

## Reading these logs

Both logs contain large volumes of `[MTP-FINDSLOT]` and `[MTP-SEQRM]` debug spam plus
binary-ish bytes that make plain `grep` treat them as binary. Use `grep -a` (or `grep -an`).
The load-time `print_info` / `llama_model_loader` blocks are identical between the two
files (same 27B GGUF); only the runtime sections differ.

The signal lines:

- `statistics mtp:` — the **MTP head** drafter's counters (`#calls`, `#gen drafts`, `#acc drafts`).
- `statistics draft:` — the **0.8B companion** drafter's counters (only in the MTP+draft log).
- `[MTP-VERIFY] pos=N: sampled=… draft=… ACCEPTED|REJECTED` — per-token verify decisions.
- `[MTP-VERIFY] bonus pos=1: sampled=…` — the free bonus token from the verify forward.
- `prompt eval time = … / eval time = … / total time =` — per-request server timing blocks.
- `llama_perf_context_print:` — per-slot final perf prints at shutdown.
- `[MTP-SEQRM] … NO checkpoint found — seq_rm FAILED` then
  `common_speculative_is_compat: seq_rm not supported, but MTP model detected — using
  checkpoint/restore for rollback` — the runtime decision to fall back to the
  snapshot/restore rollback path (infra patches 0000/07), because plain `seq_rm` on the
  recurrent half is unsupported.

---
*Provenance: copied verbatim from `/home/ubuntu/qwen27/qwen-inference-lab/logs/`.*
