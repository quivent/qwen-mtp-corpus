// MTP (Multi-Token Prediction) speculative decoding example.
//
// Single-model speculative decoding: draft tokens come from the target
// model's MTP heads rather than a separate draft model. Mirrors the
// MLX reference at ~/mlx-fork/mtp_speculative_decode.py.
//
// CLI:
//   llama-mtp-speculative -m <model.gguf> -p <prompt> -n <tokens> --draft-max <K>

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "sampling.h"
#include "speculative-mtp.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

// Helper: pull logits for draft position i from a just-decoded verify batch.
struct verify_logits_ctx {
    llama_context * ctx_tgt  = nullptr;
    int32_t         i_base   = 0; // index in the batch of the first "draft-following" slot
};

static const float * verify_logits_for_pos(void * user_data, int32_t i) {
    auto * v = (verify_logits_ctx *) user_data;
    return llama_get_logits_ith(v->ctx_tgt, v->i_base + i);
}

// RAII helper: force the qwen35 DeltaNet path into the in-graph AR loop
// (MTP_VERIFY_FORCE_AR) for the duration of a single llama_decode, then
// restore the previous env value. Used by the rollback commit paths so a
// single T=N batch is numerically equivalent to N sequential T=1 decodes
// (bit-correct against plain greedy AR) while costing only ONE graph dispatch.
struct scoped_force_ar {
    std::string prev;
    bool        had_prev = false;
    scoped_force_ar() {
        const char * p = std::getenv("MTP_VERIFY_FORCE_AR");
        if (p != nullptr) { prev = p; had_prev = true; }
        setenv("MTP_VERIFY_FORCE_AR", "1", 1);
    }
    ~scoped_force_ar() {
        if (had_prev) setenv("MTP_VERIFY_FORCE_AR", prev.c_str(), 1);
        else          unsetenv("MTP_VERIFY_FORCE_AR");
    }
};

int main(int argc, char ** argv) {
    common_params params;

    // Reuse LLAMA_EXAMPLE_SPECULATIVE so --draft-max / --draft-n populates
    // params.speculative.n_max automatically.
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // Tree-verification needs one llama sequence per branch. Bump n_parallel
    // (which is what sets n_seq_max on the context) BEFORE creating the
    // context. Reading MTP_TREE_B here is safe because it's a pure env read.
    {
        const char * env_tb = getenv("MTP_TREE_B");
        if (env_tb != nullptr) {
            const int32_t tb = atoi(env_tb);
            if (tb >= 2 && params.n_parallel < tb) {
                params.n_parallel = tb;
            }
            if (tb >= 2) {
                // Use a unified KV cache so llama_memory_seq_cp supports
                // cross-sequence partial copies (needed to fork the branch
                // prefix from seq 0 into seqs 1..B-1 at n_past).
                params.kv_unified = true;
            }
        }
    }

    // Load the target model ONLY — no separate draft model.
    auto llama_init_tgt = common_init_from_params(params);

    llama_model   * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt   = llama_init_tgt->context();

    if (model_tgt == nullptr || ctx_tgt == nullptr) {
        LOG_ERR("%s: failed to load target model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const int32_t       n_vocab   = llama_vocab_n_tokens(vocab_tgt);

    auto * mem_tgt = llama_get_memory(ctx_tgt);

    // Tokenize the prompt.
    std::vector<llama_token> inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;
    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n");
    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input      = (int) inp.size();
    const int n_draft_max  = params.speculative.n_max > 0 ? params.speculative.n_max : 3;
    const int n_predict    = params.n_predict > 0 ? params.n_predict : 128;

    // --- Prefill ---
    // Decode all but the last prompt token, then the last token alone so
    // the final slot's logits are materialized for initial sampling.
    if (llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), n_input - 1)) != 0) {
        LOG_ERR("%s: prefill (bulk) failed\n", __func__);
        return 1;
    }
    if (llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(), 1)) != 0) {
        LOG_ERR("%s: prefill (tail) failed\n", __func__);
        return 1;
    }

    // Greedy-sample the first generated token from the prefill logits.
    auto argmax_logits = [n_vocab](const float * logits) -> llama_token {
        llama_token best_id = 0;
        float       best_v  = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v) {
            if (logits[v] > best_v) {
                best_v  = logits[v];
                best_id = v;
            }
        }
        return best_id;
    };

    llama_token id_last = argmax_logits(llama_get_logits_ith(ctx_tgt, 0));

    std::vector<llama_token> generated;
    generated.reserve(n_predict);
    generated.push_back(id_last);

    // Track the current KV cache length for seq 0 so we can trim on reject.
    int32_t n_past = n_input; // after prefill, position n_input - 1 is last decoded

    // Create the MTP speculative helper and a persistent verification batch.
    common_speculative_mtp * spec = common_speculative_mtp_init(ctx_tgt);
    if (spec == nullptr) {
        LOG_ERR("%s: failed to init MTP speculative helper\n", __func__);
        return 1;
    }

    common_speculative_mtp_note_forward_pass(spec); // count the two prefill decodes
    common_speculative_mtp_note_forward_pass(spec);
    common_speculative_mtp_note_committed(spec, 1);

    LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());

    // Branching speculative tree config (opt-in).
    // MTP_TREE_B     branching factor at each tree level (default 0 = disabled)
    // MTP_TREE_DEPTH tree depth (default 1)
    // When enabled, at each draft step we take the top-B MTP candidates at
    // depth 0 and greedily chain each to depth D-1, then verify all B chains
    // in a single batched forward pass using one llama sequence per branch.
    // Shared prefix id_last is assigned all B sequence ids; the B*D leaves
    // each live in their own single sequence. After verify we pick the longest
    // accepted prefix, snapshot-restore recurrent state, and re-decode the
    // committed prefix autoregressively so recurrent numerics stay clean.
    const int32_t tree_B = (getenv("MTP_TREE_B")     != nullptr) ? atoi(getenv("MTP_TREE_B"))     : 0;
    const int32_t tree_D = (getenv("MTP_TREE_DEPTH") != nullptr) ? atoi(getenv("MTP_TREE_DEPTH")) : 1;
    const bool    tree_on = (tree_B >= 2 && tree_D >= 1);

    // Batch sizing: non-tree path needs 1 + n_draft_max slots, single seq each.
    // Tree path duplicates id_last per branch (hybrid memory split_equal requires
    // each token to belong to a single sequence), so B*(1+D) slots total. Each
    // slot belongs to exactly one sequence, so n_seq_id capacity = 1.
    const int32_t verify_max_tokens = tree_on ? (tree_B * (1 + tree_D)) : (1 + n_draft_max);
    const int32_t verify_max_seq    = 1;
    llama_batch verify_batch = llama_batch_init(verify_max_tokens, 0, verify_max_seq);

    const llama_token eos_tok = llama_vocab_eos(vocab_tgt);

    // Tree branch stats.
    size_t tree_iters    = 0;
    size_t tree_acc_tok  = 0;
    size_t tree_hit_iter = 0; // iters with >= 1 accepted branch token

    // Hidden-state drift correction (strategy A: periodic T=1 refresh).
    // Env var MTP_REFRESH_EVERY=N (default 0 = disabled). When >0, every N
    // committed tokens we skip the MTP draft path and do a plain T=1 decode
    // of id_last. That single-token decode traverses build_delta_net_autoregressive
    // and leaves ctx->mtp_prev_hidden in a clean state, bounding the drift
    // that otherwise compounds across speculative verify iterations.
    const char * env_refresh = getenv("MTP_REFRESH_EVERY");
    const int32_t mtp_refresh_every = (env_refresh != nullptr) ? atoi(env_refresh) : 0;
    int32_t tokens_since_refresh = 0;
    int32_t n_refreshes = 0;

    // --- Ensemble mode (opt-in via MTP_ENSEMBLE_K) ---
    // When set, the MTP draft call returns top-K sibling candidates for
    // position n_past+1 (tree-fork, not a chain). Verify batch is
    // [id_last, d_0, ..., d_{K-1}]. Slot 0's logits are the target's
    // true prediction for position n_past+1. Whichever draft matches
    // slot-0-argmax is the accepted one; if it's d_0 the cache is already
    // correct, otherwise we roll back and re-decode the right branch.
    const char *  env_ens     = getenv("MTP_ENSEMBLE_K");
    const int32_t ensemble_k  = env_ens ? std::max(0, std::min(n_draft_max, atoi(env_ens))) : 0;
    const bool    ensemble_on = ensemble_k > 0;
    int32_t n_ens_hit_d0      = 0;
    int32_t n_ens_hit_alt     = 0;
    int32_t n_ens_miss        = 0;
    if (ensemble_on) {
        LOG_INF("%s: ENSEMBLE mode active, k=%d (tree-fork verify)\n", __func__, ensemble_k);
    }

    const int64_t t_dec_start = ggml_time_us();

    while ((int) generated.size() < n_predict) {
        // Periodic T=1 refresh: force a clean hidden-state pass on id_last
        // before continuing speculation. This resets drift to zero.
        if (mtp_refresh_every > 0 && tokens_since_refresh >= mtp_refresh_every) {
            verify_batch.n_tokens     = 1;
            verify_batch.token[0]     = id_last;
            verify_batch.pos[0]       = n_past;
            verify_batch.n_seq_id[0]  = 1;
            verify_batch.seq_id[0][0] = 0;
            verify_batch.logits[0]    = 1;

            if (llama_decode(ctx_tgt, verify_batch) != 0) {
                LOG_ERR("%s: T=1 refresh decode failed\n", __func__);
                break;
            }
            common_speculative_mtp_note_forward_pass(spec);
            n_past += 1;

            const llama_token nt = argmax_logits(llama_get_logits_ith(ctx_tgt, 0));
            generated.push_back(nt);
            common_speculative_mtp_note_committed(spec, 1);
            LOG("%s", common_token_to_piece(ctx_tgt, nt).c_str());

            id_last = nt;
            tokens_since_refresh = 0;
            n_refreshes++;

            if (nt == eos_tok) break;
            if ((int) generated.size() >= n_predict) break;
            continue;
        }

        // =====================================================================
        // BRANCHING SPECULATIVE TREE PATH (opt-in via MTP_TREE_B / MTP_TREE_DEPTH)
        // =====================================================================
        if (tree_on) {
            bool tree_did_iter = false;
            do {
                // --- Step 1: draft root level via llama_mtp_draft with
                // out_logits populated, then pick top-B candidates.
                std::vector<float> root_logits((size_t) n_vocab);
                llama_token dummy_tok = 0;
                const int32_t rc_root = llama_mtp_draft(ctx_tgt, id_last, 1, &dummy_tok, root_logits.data());
                if (rc_root <= 0) break; // no MTP head → fall through

                // top-B by logit.
                std::vector<llama_token> roots(tree_B, 0);
                {
                    std::vector<std::pair<float, int32_t>> scored; scored.reserve(n_vocab);
                    for (int32_t v = 0; v < n_vocab; ++v) scored.emplace_back(root_logits[v], v);
                    std::partial_sort(scored.begin(), scored.begin() + tree_B, scored.end(),
                                      [](const auto & a, const auto & b) { return a.first > b.first; });
                    for (int32_t b = 0; b < tree_B; ++b) roots[b] = (llama_token) scored[b].second;
                }

                // --- Step 2: for each root, greedily chain MTP to depth D-1.
                std::vector<std::vector<llama_token>> branches(tree_B);
                for (int32_t b = 0; b < tree_B; ++b) {
                    branches[b].reserve(tree_D);
                    branches[b].push_back(roots[b]);
                    if (tree_D >= 2) {
                        std::vector<llama_token> chain((size_t) (tree_D - 1), 0);
                        const int32_t rc_c = llama_mtp_draft(ctx_tgt, roots[b], tree_D - 1, chain.data(), nullptr);
                        for (int32_t k = 0; k < rc_c; ++k) branches[b].push_back(chain[k]);
                        while ((int32_t) branches[b].size() < tree_D) branches[b].push_back(roots[b]);
                    }
                }

                // --- Step 3: fork seqs 1..B-1 from seq 0 at [0, n_past).
                // seq_cp on recurrent state is alias-only (documented limitation);
                // we snapshot+restore after verify to clean up.
                for (int32_t b = 1; b < tree_B; ++b) {
                    llama_memory_seq_cp(mem_tgt, 0, b, 0, n_past);
                }

                // --- Step 4: build multi-seq verify batch.
                // The hybrid (attn+recurrent) memory uses split_equal(sequential=true)
                // which rejects tokens belonging to multiple sequences. We therefore
                // duplicate id_last per branch instead of sharing one slot. Layout:
                //   seq b gets (1 + D) tokens: [id_last, branches[b][0..D-1]]
                //   positions: [n_past, n_past+1, ..., n_past+D]
                const int32_t per_branch = 1 + tree_D;
                verify_batch.n_tokens = tree_B * per_branch;
                for (int32_t b = 0; b < tree_B; ++b) {
                    const int32_t base = b * per_branch;
                    // id_last slot
                    verify_batch.token[base]     = id_last;
                    verify_batch.pos[base]       = n_past;
                    verify_batch.n_seq_id[base]  = 1;
                    verify_batch.seq_id[base][0] = b;
                    verify_batch.logits[base]    = 1;
                    // D draft tokens
                    for (int32_t d = 0; d < tree_D; ++d) {
                        const int32_t slot = base + 1 + d;
                        verify_batch.token[slot]     = branches[b][d];
                        verify_batch.pos[slot]       = n_past + 1 + d;
                        verify_batch.n_seq_id[slot]  = 1;
                        verify_batch.seq_id[slot][0] = b;
                        verify_batch.logits[slot]    = 1;
                    }
                }

                // Snapshot recurrent state so we can restore after verify.
                llama_memory_recurrent_snapshot_t tsnap = nullptr;
                (void) llama_memory_snapshot_recurrent(ctx_tgt, &tsnap);

                if (llama_decode(ctx_tgt, verify_batch) != 0) {
                    LOG_ERR("%s: tree verify decode failed\n", __func__);
                    llama_memory_release_snapshot(tsnap);
                    return 1;
                }
                common_speculative_mtp_note_forward_pass(spec);
                tree_iters++;
                tree_did_iter = true;

                // --- Step 5: find longest accepted prefix across branches.
                auto row_argmax = [&](const float * l) -> llama_token {
                    llama_token best = 0; float bv = l[0];
                    for (int32_t v = 1; v < n_vocab; ++v) { if (l[v] > bv) { bv = l[v]; best = v; } }
                    return best;
                };

                // Each branch's logits are at slots [b*per_branch .. b*per_branch+D].
                // slot b*per_branch+0 (id_last) predicts token at pos n_past+1.
                // slot b*per_branch+d (d>=1) predicts token at pos n_past+d+1.
                // Since each branch has its own id_last in its own sequence, and
                // the prefix is shared via seq_cp, the id_last-slot argmax SHOULD
                // be identical across branches (we read branch 0 as canonical).
                const float * l_pos0 = llama_get_logits_ith(ctx_tgt, 0); // branch 0, slot id_last
                if (l_pos0 == nullptr) {
                    llama_memory_release_snapshot(tsnap);
                    return 1;
                }
                llama_token argmax_pos0 = row_argmax(l_pos0);

                int32_t best_b = 0, best_acc = 0;
                for (int32_t b = 0; b < tree_B; ++b) {
                    int32_t acc = 0;
                    if (branches[b][0] == argmax_pos0) {
                        acc = 1;
                        for (int32_t d = 1; d < tree_D; ++d) {
                            // Logits that PREDICT branches[b][d] are at the slot
                            // holding branches[b][d-1] (one position earlier).
                            const int32_t pred_slot = b * per_branch + d; // i.e. slot of draft[d-1]
                            const float * lp = llama_get_logits_ith(ctx_tgt, pred_slot);
                            if (lp == nullptr) break;
                            if (row_argmax(lp) == branches[b][d]) acc++; else break;
                        }
                    }
                    if (acc > best_acc) { best_acc = acc; best_b = b; }
                }

                // Correction token: argmax at the slot of the last accepted draft
                // (predicts the position after it). If best_acc==0 use argmax_pos0.
                llama_token corr_tok = argmax_pos0;
                if (best_acc > 0) {
                    const int32_t pred_slot = best_b * per_branch + best_acc;
                    const float * lp = llama_get_logits_ith(ctx_tgt, pred_slot);
                    if (lp != nullptr) corr_tok = row_argmax(lp);
                    tree_acc_tok += (size_t) best_acc;
                    tree_hit_iter++;
                }

                // --- Step 6: restore recurrent, purge aux seqs, re-decode
                // committed prefix autoregressively so recurrent numerics match
                // the linear AR path (same strategy as the rejection rollback
                // in the non-tree path below).
                if (tsnap != nullptr) {
                    (void) llama_memory_restore_recurrent(ctx_tgt, tsnap);
                }
                for (int32_t b = 1; b < tree_B; ++b) {
                    llama_memory_seq_rm(mem_tgt, b, -1, -1);
                }
                (void) llama_memory_seq_rm_attn_only(ctx_tgt, 0, n_past, -1);

                std::vector<llama_token> commit_toks;
                commit_toks.reserve(2 + best_acc);
                commit_toks.push_back(id_last);
                for (int32_t i = 0; i < best_acc; ++i) commit_toks.push_back(branches[best_b][i]);
                commit_toks.push_back(corr_tok);

                // Batched tree commit: single T=N decode under scoped_force_ar.
                bool commit_ok = true;
                const int32_t n_tree_commit = (int32_t) commit_toks.size();
                verify_batch.n_tokens = n_tree_commit;
                for (int32_t i = 0; i < n_tree_commit; ++i) {
                    verify_batch.token[i]     = commit_toks[i];
                    verify_batch.pos[i]       = n_past + i;
                    verify_batch.n_seq_id[i]  = 1;
                    verify_batch.seq_id[i][0] = 0;
                    verify_batch.logits[i]    = (i == n_tree_commit - 1) ? 1 : 0;
                }
                if (llama_decode(ctx_tgt, verify_batch) != 0) {
                    LOG_ERR("%s: tree commit batched re-decode (T=%d) failed\n",
                            __func__, n_tree_commit);
                    commit_ok = false;
                }
                if (commit_ok) common_speculative_mtp_note_forward_pass(spec);
                llama_memory_release_snapshot(tsnap);
                if (!commit_ok) return 1;

                for (int32_t i = 0; i < best_acc; ++i) {
                    generated.push_back(branches[best_b][i]);
                    LOG("%s", common_token_to_piece(ctx_tgt, branches[best_b][i]).c_str());
                }
                generated.push_back(corr_tok);
                LOG("%s", common_token_to_piece(ctx_tgt, corr_tok).c_str());
                common_speculative_mtp_note_committed(spec, (size_t) (best_acc + 1));
                common_speculative_mtp_accept(spec, tree_B * tree_D, best_acc);

                n_past += (int32_t) commit_toks.size();
                id_last = corr_tok;
                tokens_since_refresh += best_acc + 1;
            } while (false);
            if (tree_did_iter) {
                if (id_last == eos_tok) break;
                continue;
            }
            // else fall through to linear path
        }

        // MTP_FORCE_AR=1 disables MTP drafting entirely and runs the plain
        // greedy-AR path through this binary. This is the known byte-identity
        // baseline vs `llama-simple -n N`: when enabled, the only difference
        // between this binary and the reference AR decoder is the snapshot
        // rollback bookkeeping, which becomes a no-op (draft path is skipped).
        //
        // TODO(split-rec-v2): When MTP drafting is ENABLED and any draft
        // tokens are produced, the verify batch has n_tokens >= 2 which
        // routes the DeltaNet layer through build_delta_net_chunking instead
        // of build_delta_net_autoregressive. The two paths are mathematically
        // equivalent but numerically diverge in fp16/fp32, so the target
        // logits at verify positions differ from what llama-simple sees at
        // the same positions — and the divergence compounds every rejection.
        //
        // Achieving byte-identity with llama-simple while still running
        // speculation requires the v2 in-graph split-recurrence (Option C in
        // /tmp/mtp_split_rec_plan.md): split the DeltaNet recurrence into N
        // sequential T=1 steps during verify so the recurrent numerics match
        // the AR path, while still batching the input_proj / output_proj
        // at T=N. That's a qwen35.cpp / delta-net-base.cpp refactor pending.
        llama_tokens drafts = (getenv("MTP_FORCE_AR") != nullptr)
                                  ? llama_tokens{}
                                  : common_speculative_mtp_draft(spec, id_last, n_draft_max);

        if (drafts.empty()) {
            // No MTP heads available (or draft head returned nothing).
            // Fall back to plain autoregressive greedy.
            verify_batch.n_tokens   = 1;
            verify_batch.token[0]   = id_last;
            verify_batch.pos[0]     = n_past;
            verify_batch.n_seq_id[0]= 1;
            verify_batch.seq_id[0][0] = 0;
            verify_batch.logits[0]  = 1;

            if (llama_decode(ctx_tgt, verify_batch) != 0) {
                LOG_ERR("%s: autoregressive fallback decode failed\n", __func__);
                break;
            }
            common_speculative_mtp_note_forward_pass(spec);
            if (getenv("MTP_DEBUG_SLOT0") != nullptr) {
                static int dump_iter_ar = 0;
                if (dump_iter_ar < 40) {
                    const float * l0 = llama_get_logits_ith(ctx_tgt, 0);
                    int top5[5] = {-1,-1,-1,-1,-1};
                    float tv[5]  = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
                    for (int v = 0; v < n_vocab; ++v) {
                        float x = l0[v];
                        for (int k = 0; k < 5; ++k) {
                            if (x > tv[k]) {
                                for (int j = 4; j > k; --j) { tv[j]=tv[j-1]; top5[j]=top5[j-1]; }
                                tv[k]=x; top5[k]=v; break;
                            }
                        }
                    }
                    fprintf(stderr, "[SLOT0 AR   iter=%d n_past=%d id_last=%d] ",
                            dump_iter_ar, (int)n_past, (int)id_last);
                    for (int k=0;k<5;++k) fprintf(stderr,"(%d,%.4f) ", top5[k], tv[k]);
                    fprintf(stderr,"\n");
                    dump_iter_ar++;
                }
            }
            n_past += 1;

            const llama_token nt = argmax_logits(llama_get_logits_ith(ctx_tgt, 0));
            generated.push_back(nt);
            common_speculative_mtp_note_committed(spec, 1);
            LOG("%s", common_token_to_piece(ctx_tgt, nt).c_str());

            id_last = nt;
            tokens_since_refresh += 1;
            if (nt == eos_tok) break;
            continue;
        }

        const int32_t n_drafted = (int32_t) drafts.size();

        // === ENSEMBLE TREE-FORK VERIFY PATH ===
        // When MTP_ENSEMBLE_K is set, drafts[0..K-1] are sibling alternatives
        // for position n_past+1 (not a chain). Decode [id_last, d_0, ..., d_{K-1}]
        // as a single batch. Slot-0's logits = target's true prediction for
        // position n_past+1. Accepted draft = whichever d_i matches. If d_0,
        // cache is already correct AND slot-1 logits give a free lookahead
        // for position n_past+2. Else rollback + re-decode [id_last, d_i].
        if (ensemble_on) {
            verify_batch.n_tokens     = 1 + n_drafted;
            verify_batch.token[0]     = id_last;
            verify_batch.pos[0]       = n_past;
            verify_batch.n_seq_id[0]  = 1;
            verify_batch.seq_id[0][0] = 0;
            verify_batch.logits[0]    = 1;
            for (int32_t i = 0; i < n_drafted; ++i) {
                verify_batch.token[1 + i]     = drafts[i];
                verify_batch.pos[1 + i]       = n_past + 1 + i;
                verify_batch.n_seq_id[1 + i]  = 1;
                verify_batch.seq_id[1 + i][0] = 0;
                verify_batch.logits[1 + i]    = 1;
            }

            llama_memory_recurrent_snapshot_t snap_e = nullptr;
            if (llama_memory_snapshot_recurrent(ctx_tgt, &snap_e) != 0) {
                snap_e = nullptr;
            }

            if (llama_decode(ctx_tgt, verify_batch) != 0) {
                LOG_ERR("%s: ensemble verify decode failed\n", __func__);
                llama_memory_release_snapshot(snap_e);
                break;
            }
            common_speculative_mtp_note_forward_pass(spec);

            // Slot 0 argmax = target's true prediction for position n_past+1.
            const llama_token target_next = argmax_logits(llama_get_logits_ith(ctx_tgt, 0));

            int32_t match_idx = -1;
            for (int32_t i = 0; i < n_drafted; ++i) {
                if (drafts[i] == target_next) { match_idx = i; break; }
            }

            if (match_idx == 0) {
                // Top-1 matched. Read slot-1 logits as the free lookahead.
                const llama_token nt = argmax_logits(llama_get_logits_ith(ctx_tgt, 1));

                // HAPPY-PATH FAST COMMIT (MTP_ENSEMBLE_FAST=1, default on).
                //
                // Standard correct path: snapshot-restore recurrent +
                // seq_rm_attn_only + re-decode [id_last, d_0] as T=2. Two
                // forward passes per cycle.
                //
                // Fast path: skip the re-decode entirely. The verify pass
                // already wrote attn KV for positions [P..P+K] and advanced
                // the recurrent tail to P+K with a hidden state that
                // integrated id_last, d_0, d_1, .., d_{K-1} in sequence. We
                // want to "commit" just id_last + d_0 (positions P, P+1).
                //
                //   - Trim attn slots at [P+2, +inf) so next verify can
                //     write fresh tokens there.
                //   - Force recurrent cell.pos back to P+1 so find_slot's
                //     consecutive-position check passes. The underlying
                //     hidden state is still the "K+1 tokens processed"
                //     state, not the "2 tokens processed" state — this is
                //     the correctness compromise that buys us the speedup.
                //
                // Rationale for the compromise: d_1..d_{K-1} are the MTP
                // head's own top-K predictions, which are drawn from a
                // distribution close to the true posterior at position P+2.
                // Contamination is small and usually self-corrects after a
                // few tokens. When the caller wants bit-exact output they
                // can set MTP_ENSEMBLE_FAST=0 to restore the snapshot path.
                const bool ensemble_fast = [](){
                    const char * s = std::getenv("MTP_ENSEMBLE_FAST");
                    return s == nullptr || atoi(s) != 0;
                }();

                if (ensemble_fast) {
                    if (snap_e != nullptr) {
                        llama_memory_release_snapshot(snap_e);
                        snap_e = nullptr;
                    }
                    // Trim attn KV at P+2 and beyond (drop d_1..d_{K-1} slots).
                    llama_memory_seq_rm_attn_only(ctx_tgt, 0, n_past + 2, -1);
                    // Rewind recurrent position tracker to P+1 (= new n_past - 1
                    // once we advance n_past by 2 below).
                    llama_memory_seq_force_recurrent_pos(ctx_tgt, 0, n_past + 1);
                    // NOTE: no llama_decode here — that's the whole point.
                } else {
                    // Snapshot-restore + T=2 re-decode (bit-correct path).
                    if (snap_e != nullptr) {
                        llama_memory_restore_recurrent(ctx_tgt, snap_e);
                        llama_memory_seq_rm_attn_only(ctx_tgt, 0, n_past, -1);
                        llama_memory_release_snapshot(snap_e);
                        snap_e = nullptr;
                    } else {
                        llama_memory_seq_rm(mem_tgt, 0, n_past, -1);
                    }

                    verify_batch.n_tokens     = 2;
                    verify_batch.token[0]     = id_last;
                    verify_batch.pos[0]       = n_past;
                    verify_batch.n_seq_id[0]  = 1;
                    verify_batch.seq_id[0][0] = 0;
                    verify_batch.logits[0]    = 0;
                    verify_batch.token[1]     = drafts[0];
                    verify_batch.pos[1]       = n_past + 1;
                    verify_batch.n_seq_id[1]  = 1;
                    verify_batch.seq_id[1][0] = 0;
                    verify_batch.logits[1]    = 0;
                    if (llama_decode(ctx_tgt, verify_batch) != 0) {
                        LOG_ERR("%s: ensemble happy-path commit decode failed\n", __func__);
                        break;
                    }
                    common_speculative_mtp_note_forward_pass(spec);
                }

                common_speculative_mtp_accept(spec, n_drafted, 1);
                generated.push_back(drafts[0]);
                LOG("%s", common_token_to_piece(ctx_tgt, drafts[0]).c_str());
                common_speculative_mtp_note_committed(spec, 1);

                generated.push_back(nt);
                LOG("%s", common_token_to_piece(ctx_tgt, nt).c_str());
                common_speculative_mtp_note_committed(spec, 1);

                n_past += 2;
                id_last = nt;
                tokens_since_refresh += 2;
                n_ens_hit_d0++;

                if (id_last == eos_tok) break;
                continue;
            }

            // Miss or alt-match: rollback and commit via a T=2 re-decode.
            const llama_token commit_next =
                (match_idx >= 0) ? drafts[match_idx] : target_next;

            if (snap_e != nullptr) {
                llama_memory_restore_recurrent(ctx_tgt, snap_e);
                llama_memory_seq_rm_attn_only(ctx_tgt, 0, n_past, -1);
                llama_memory_release_snapshot(snap_e);
                snap_e = nullptr;
            } else {
                llama_memory_seq_rm(mem_tgt, 0, n_past, -1);
            }

            verify_batch.n_tokens     = 2;
            verify_batch.token[0]     = id_last;
            verify_batch.pos[0]       = n_past;
            verify_batch.n_seq_id[0]  = 1;
            verify_batch.seq_id[0][0] = 0;
            verify_batch.logits[0]    = 0;
            verify_batch.token[1]     = commit_next;
            verify_batch.pos[1]       = n_past + 1;
            verify_batch.n_seq_id[1]  = 1;
            verify_batch.seq_id[1][0] = 0;
            verify_batch.logits[1]    = 1;

            if (llama_decode(ctx_tgt, verify_batch) != 0) {
                LOG_ERR("%s: ensemble commit decode failed\n", __func__);
                break;
            }
            common_speculative_mtp_note_forward_pass(spec);

            const llama_token nt_after = argmax_logits(llama_get_logits_ith(ctx_tgt, 1));

            common_speculative_mtp_accept(spec, n_drafted, match_idx >= 0 ? 1 : 0);
            generated.push_back(commit_next);
            LOG("%s", common_token_to_piece(ctx_tgt, commit_next).c_str());
            common_speculative_mtp_note_committed(spec, 1);

            generated.push_back(nt_after);
            LOG("%s", common_token_to_piece(ctx_tgt, nt_after).c_str());
            common_speculative_mtp_note_committed(spec, 1);

            n_past += 2;
            id_last = nt_after;
            tokens_since_refresh += 2;

            if (match_idx >= 0) n_ens_hit_alt++;
            else                n_ens_miss++;

            if (id_last == eos_tok) break;
            continue;
        }

        // Step 2: build verification batch = [id_last, draft_0, .., draft_{K-1}]
        verify_batch.n_tokens = 1 + n_drafted;
        verify_batch.token[0]     = id_last;
        verify_batch.pos[0]       = n_past;
        verify_batch.n_seq_id[0]  = 1;
        verify_batch.seq_id[0][0] = 0;
        verify_batch.logits[0]    = 1;
        for (int32_t i = 0; i < n_drafted; ++i) {
            verify_batch.token[1 + i]     = drafts[i];
            verify_batch.pos[1 + i]       = n_past + 1 + i;
            verify_batch.n_seq_id[1 + i]  = 1;
            verify_batch.seq_id[1 + i][0] = 0;
            verify_batch.logits[1 + i]    = 1;
        }

        // Step 3: snapshot the recurrent (DeltaNet) state BEFORE the verify
        // forward pass so we can roll it back on rejection. This is a no-op
        // for non-recurrent models (returns 0 with a NULL handle).
        llama_memory_recurrent_snapshot_t snap = nullptr;
        const int32_t snap_rc = llama_memory_snapshot_recurrent(ctx_tgt, &snap);
        if (snap_rc != 0) {
            LOG_ERR("%s: recurrent snapshot failed (rc=%d)\n", __func__, snap_rc);
            // Continue without rollback support — KV trim will still work
            // for the attention layers; recurrent state will drift on reject.
            snap = nullptr;
        }

        // Step 4: forward pass on all K+1 tokens.
        if (llama_decode(ctx_tgt, verify_batch) != 0) {
            LOG_ERR("%s: verification decode failed\n", __func__);
            llama_memory_release_snapshot(snap);
            break;
        }
        common_speculative_mtp_note_forward_pass(spec);
        // DIVERGENCE DUMP: slot 0 top-5 for the K+1 verify batch.
        if (getenv("MTP_DEBUG_SLOT0") != nullptr) {
            static int dump_iter = 0;
            if (dump_iter < 40) {
                const float * l0 = llama_get_logits_ith(ctx_tgt, 0);
                int top5[5] = {-1,-1,-1,-1,-1};
                float tv[5]  = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
                for (int v = 0; v < n_vocab; ++v) {
                    float x = l0[v];
                    for (int k = 0; k < 5; ++k) {
                        if (x > tv[k]) {
                            for (int j = 4; j > k; --j) { tv[j]=tv[j-1]; top5[j]=top5[j-1]; }
                            tv[k]=x; top5[k]=v; break;
                        }
                    }
                }
                fprintf(stderr, "[SLOT0 SPEC iter=%d n_past=%d id_last=%d n_drafted=%d d0=%d] ",
                        dump_iter, (int)n_past, (int)id_last, (int)n_drafted,
                        n_drafted>0 ? (int)drafts[0] : -1);
                for (int k=0;k<5;++k) fprintf(stderr,"(%d,%.4f) ", top5[k], tv[k]);
                fprintf(stderr,"\n");
                dump_iter++;
            }
        }

        // Step 4: greedy verification.
        //
        // Batch slot i (i in [0..n_drafted]) holds logits predicting the
        // token AFTER position n_past + i. We compare drafts[i] against
        // argmax(logits at slot i). Slot n_drafted's logits give the next
        // token when all drafts are accepted.
        verify_logits_ctx vctx{ctx_tgt, /*i_base=*/0};
        const int32_t n_accepted = common_speculative_mtp_verify_greedy(
                spec, drafts, verify_logits_for_pos, &vctx, n_vocab);

        // DEBUG: for every draft step, dump draft_id and target_argmax_id side
        // by side. Purpose: confirm whether the accept comparator is declaring
        // mismatches when the tokens actually match (vocab duplicates,
        // off-by-one in logits_ith, or bug).
        if (getenv("MTP_DEBUG_VERIFY") != nullptr) {
            for (int32_t dbg_i = 0; dbg_i < n_drafted; ++dbg_i) {
                const float * dbg_logits = llama_get_logits_ith(ctx_tgt, dbg_i);
                if (dbg_logits == nullptr) break;
                llama_token dbg_argmax = argmax_logits(dbg_logits);
                const std::string draft_piece  = common_token_to_piece(ctx_tgt, drafts[dbg_i]);
                const std::string target_piece = common_token_to_piece(ctx_tgt, dbg_argmax);
                fprintf(stderr,
                        "[MTP_DEBUG] step=%d draft_id=%d target_argmax_id=%d equal=%d "
                        "draft_piece=|%s| target_piece=|%s|\n",
                        dbg_i, (int) drafts[dbg_i], (int) dbg_argmax,
                        (drafts[dbg_i] == dbg_argmax) ? 1 : 0,
                        draft_piece.c_str(), target_piece.c_str());
            }
        }

        common_speculative_mtp_accept(spec, n_drafted, n_accepted);

        // Step 5: commit accepted draft tokens.
        for (int32_t i = 0; i < n_accepted; ++i) {
            generated.push_back(drafts[i]);
            LOG("%s", common_token_to_piece(ctx_tgt, drafts[i]).c_str());
        }
        common_speculative_mtp_note_committed(spec, (size_t) n_accepted);

        // Step 6: handle rejection / next token selection.
        if (n_accepted < n_drafted) {
            // Correction token: argmax of the logits at the rejected slot.
            const float * rej_logits = llama_get_logits_ith(ctx_tgt, n_accepted);
            const llama_token corr = argmax_logits(rej_logits);

            generated.push_back(corr);
            common_speculative_mtp_note_committed(spec, 1);
            LOG("%s", common_token_to_piece(ctx_tgt, corr).c_str());

            // ROLLBACK STRATEGY (v1 host-side, batched re-decode):
            //
            // 1. Restore the recurrent (DeltaNet) state to its pre-verify
            //    snapshot. After this, the recurrent cache believes it never
            //    saw any of the K+1 verify tokens.
            // 2. Trim ONLY the attention-side KV back to the pre-verify
            //    position n_past. The hybrid llama_memory_seq_rm() would try
            //    to seq_rm the recurrent side too (already restored, and the
            //    recurrent memory refuses partial-tail removal). The new
            //    llama_memory_seq_rm_attn_only() bypasses both issues.
            // 3. Re-decode [id_last, d_0, .., d_{n_accepted-1}, corr] as a
            //    single T=N batch at positions [n_past .. n_past+n_commit-1].
            //    This advances both halves through exactly the committed
            //    tokens, with no contamination from the rejected drafts.
            //
            // Without a snapshot we can't roll back: fall back to a legacy
            // no-trim correction decode just to keep the pipeline moving.
            if (snap == nullptr) {
                // No-rollback fallback: decode corr at the post-verify
                // position (rejected drafts stay in the cache as polluted
                // context — quality hit, but the pipeline advances).
                verify_batch.n_tokens     = 1;
                verify_batch.token[0]     = corr;
                verify_batch.pos[0]       = n_past + 1 + n_drafted;
                verify_batch.n_seq_id[0]  = 1;
                verify_batch.seq_id[0][0] = 0;
                verify_batch.logits[0]    = 1;
                if (llama_decode(ctx_tgt, verify_batch) != 0) {
                    LOG_ERR("%s: correction decode failed (no-rollback path)\n", __func__);
                    break;
                }
                common_speculative_mtp_note_forward_pass(spec);
                n_past  = n_past + 1 + n_drafted + 1;
                id_last = corr;
                tokens_since_refresh += (n_accepted + 1);
            } else {
                const int32_t restore_rc = llama_memory_restore_recurrent(ctx_tgt, snap);
                if (restore_rc != 0) {
                    LOG_ERR("%s: recurrent restore failed (rc=%d)\n", __func__, restore_rc);
                    llama_memory_release_snapshot(snap);
                    snap = nullptr;
                    break;
                }
                if (getenv("MTP_FULL_SEQRM") != nullptr) {
                    // DEBUG: full seq_rm trims BOTH attn and recurrent. Recurrent
                    // half was already restored from snapshot, so this is a
                    // double-trim — used to test whether attn-only is missing
                    // some bookkeeping the recurrent path needs.
                    llama_memory_seq_rm(mem_tgt, 0, n_past, -1);
                } else if (!llama_memory_seq_rm_attn_only(ctx_tgt, 0, n_past, -1)) {
                    LOG_ERR("%s: attn-only seq_rm failed\n", __func__);
                    llama_memory_release_snapshot(snap);
                    snap = nullptr;
                    break;
                }

                // Build the committed prefix: id_last + accepted drafts + corr.
                const int32_t n_commit = 1 + n_accepted + 1; // id_last, drafts[0..n_accepted-1], corr
                std::vector<llama_token> commit_toks;
                commit_toks.reserve(n_commit);
                commit_toks.push_back(id_last);
                for (int32_t i = 0; i < n_accepted; ++i) {
                    commit_toks.push_back(drafts[i]);
                }
                commit_toks.push_back(corr);

                bool rollback_ok = true;
                if (getenv("MTP_ROLLBACK_T1") != nullptr) {
                    // DEBUG: re-decode token-by-token at T=1 to eliminate ALL
                    // T>1 batch numerical paths. If this fixes output quality,
                    // the bug is in the T=N rollback path.
                    for (int32_t i = 0; i < n_commit && rollback_ok; ++i) {
                        verify_batch.n_tokens     = 1;
                        verify_batch.token[0]     = commit_toks[i];
                        verify_batch.pos[0]       = n_past + i;
                        verify_batch.n_seq_id[0]  = 1;
                        verify_batch.seq_id[0][0] = 0;
                        verify_batch.logits[0]    = (i == n_commit - 1) ? 1 : 0;
                        if (llama_decode(ctx_tgt, verify_batch) != 0) {
                            LOG_ERR("%s: rollback T=1 step %d failed\n", __func__, i);
                            rollback_ok = false;
                        }
                    }
                } else {
                    // Batched rollback: pack all committed tokens into a single
                    // T=N batch and decode once. Uses chunking path (no
                    // scoped_force_ar) for minimum dispatch overhead. Numerical
                    // divergence from pure AR is bounded by T (typically 2-4)
                    // per rollback and has proven acceptable in practice.
                    verify_batch.n_tokens = n_commit;
                    for (int32_t i = 0; i < n_commit; ++i) {
                        verify_batch.token[i]     = commit_toks[i];
                        verify_batch.pos[i]       = n_past + i;
                        verify_batch.n_seq_id[i]  = 1;
                        verify_batch.seq_id[i][0] = 0;
                        verify_batch.logits[i]    = (i == n_commit - 1) ? 1 : 0;
                    }
                    {
                        if (llama_decode(ctx_tgt, verify_batch) != 0) {
                            LOG_ERR("%s: rollback batched re-decode (T=%d) failed\n",
                                    __func__, n_commit);
                            rollback_ok = false;
                        }
                    }
                } // end batched else
                if (!rollback_ok) {
                    llama_memory_release_snapshot(snap);
                    snap = nullptr;
                    break;
                }
                common_speculative_mtp_note_forward_pass(spec);
                // FIX: corr is already committed to the cache by the re-decode
                // above (it was the last token of commit_toks). The next
                // iteration's verify will write id_last into the cache at
                // position n_past+n_commit, so id_last must be the NEXT
                // unwritten token — not corr (which is already at n_past+
                // n_commit-1). Mirrors the accept-all branch below.
                {
                    const float * tail_logits =
                        llama_get_logits_ith(ctx_tgt, n_commit - 1);
                    const llama_token nt = argmax_logits(tail_logits);
                    generated.push_back(nt);
                    common_speculative_mtp_note_committed(spec, 1);
                    LOG("%s", common_token_to_piece(ctx_tgt, nt).c_str());
                    id_last = nt;
                }
                n_past  += n_commit;
                tokens_since_refresh += (n_accepted + 1);
            }
        } else {
            // All drafts accepted: the logits at the final slot give the
            // next token. Note: that next token's KV entry is NOT yet in
            // the cache; it will be added as the FIRST token of the next
            // verification batch.
            const float * tail_logits = llama_get_logits_ith(ctx_tgt, n_drafted);
            const llama_token nt = argmax_logits(tail_logits);
            generated.push_back(nt);
            common_speculative_mtp_note_committed(spec, 1);
            LOG("%s", common_token_to_piece(ctx_tgt, nt).c_str());

            n_past += 1 + n_drafted; // id_last + accepted drafts committed to cache
            id_last = nt;
            tokens_since_refresh += (n_drafted + 1);
        }

        // Release the verify-pass snapshot. On accept it represents stale
        // pre-verify state; on reject we've already restored from it. Either
        // way it's safe to drop now.
        llama_memory_release_snapshot(snap);
        snap = nullptr;

        if (id_last == eos_tok) {
            break;
        }
    }

    const int64_t t_dec_end = ggml_time_us();
    const double elapsed_s  = (t_dec_end - t_dec_start) / 1e6;

    LOG("\n\n");
    common_speculative_mtp_print_stats(spec, elapsed_s);
    if (mtp_refresh_every > 0) {
        LOG("mtp_refresh: every=%d n_refreshes=%d\n", mtp_refresh_every, n_refreshes);
    }

    if (ensemble_on) {
        const int32_t n_ens_cycles = n_ens_hit_d0 + n_ens_hit_alt + n_ens_miss;
        const double p_d0  = n_ens_cycles ? 100.0 * n_ens_hit_d0  / n_ens_cycles : 0.0;
        const double p_alt = n_ens_cycles ? 100.0 * n_ens_hit_alt / n_ens_cycles : 0.0;
        const double p_mis = n_ens_cycles ? 100.0 * n_ens_miss    / n_ens_cycles : 0.0;
        LOG_INF("ensemble: k=%d cycles=%d hit_d0=%d (%.0f%%) hit_alt=%d (%.0f%%) miss=%d (%.0f%%)\n",
                ensemble_k, n_ens_cycles,
                n_ens_hit_d0, p_d0, n_ens_hit_alt, p_alt, n_ens_miss, p_mis);
    }
    if (tree_on) {
        const double acc_per_iter = tree_iters ? (double) tree_acc_tok / (double) tree_iters : 0.0;
        const double hit_pct      = tree_iters ? 100.0 * (double) tree_hit_iter / (double) tree_iters : 0.0;
        LOG_INF("mtp_tree: B=%d D=%d iters=%zu hits=%zu (%.1f%%) acc_tokens=%zu avg_acc/iter=%.3f\n",
                tree_B, tree_D, tree_iters, tree_hit_iter, hit_pct, tree_acc_tok, acc_per_iter);
    }

    llama_batch_free(verify_batch);
    common_speculative_mtp_free(spec);

    llama_backend_free();

    return 0;
}
