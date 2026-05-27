#include "speculative-mtp.h"

#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <vector>

struct common_speculative_mtp {
    llama_context * ctx_tgt = nullptr;

    // Scratch buffer reused across calls to llama_mtp_draft to avoid
    // reallocations in the hot path.
    std::vector<llama_token> scratch_tokens;

    common_speculative_mtp_stats stats;
};

struct common_speculative_mtp * common_speculative_mtp_init(llama_context * ctx_tgt) {
    if (ctx_tgt == nullptr) {
        return nullptr;
    }

    auto * spec = new common_speculative_mtp;
    spec->ctx_tgt = ctx_tgt;
    return spec;
}

void common_speculative_mtp_free(struct common_speculative_mtp * spec) {
    if (spec == nullptr) {
        return;
    }
    delete spec;
}

llama_tokens common_speculative_mtp_draft(
        struct common_speculative_mtp * spec,
        llama_token                     id_last,
        int32_t                         n_draft) {
    llama_tokens result;

    if (spec == nullptr || n_draft <= 0) {
        return result;
    }

    if ((int32_t) spec->scratch_tokens.size() < n_draft) {
        spec->scratch_tokens.resize(n_draft);
    }

    // Call into the core-library MTP drafting entry point. This is a
    // cross-agent boundary: if the MTP graph agent has not yet merged,
    // this symbol will fail to resolve at link time (but the translation
    // unit will still compile cleanly).
    const int32_t n = llama_mtp_draft(
            spec->ctx_tgt,
            id_last,
            n_draft,
            spec->scratch_tokens.data(),
            /* out_logits */ nullptr);

    spec->stats.draft_calls++;

    if (n <= 0) {
        return result;
    }

    result.assign(spec->scratch_tokens.begin(), spec->scratch_tokens.begin() + n);
    return result;
}

int32_t common_speculative_mtp_verify_greedy(
        struct common_speculative_mtp * spec,
        const llama_tokens            & draft_tokens,
        const float *                 (*logits_for_pos)(void * user_data, int32_t i),
        void                          * user_data,
        int32_t                         n_vocab) {
    if (spec == nullptr || logits_for_pos == nullptr || n_vocab <= 0) {
        return 0;
    }

    int32_t n_accepted = 0;
    for (int32_t i = 0; i < (int32_t) draft_tokens.size(); ++i) {
        const float * logits = logits_for_pos(user_data, i);
        if (logits == nullptr) {
            break;
        }

        // argmax over the full vocabulary — mirrors verify_draft() in
        // mtp_speculative_decode.py. Stochastic verification can be
        // layered on later; greedy is sufficient for v1.
        int32_t best_id  = 0;
        float   best_val = logits[0];
        for (int32_t v = 1; v < n_vocab; ++v) {
            const float l = logits[v];
            if (l > best_val) {
                best_val = l;
                best_id  = v;
            }
        }

        if (best_id == draft_tokens[i]) {
            n_accepted++;
        } else {
            break;
        }
    }

    return n_accepted;
}

void common_speculative_mtp_accept(
        struct common_speculative_mtp * spec,
        int32_t                         n_drafted,
        int32_t                         n_accepted) {
    if (spec == nullptr || n_drafted <= 0) {
        return;
    }

    spec->stats.draft_tokens    += (size_t) n_drafted;
    spec->stats.accepted_tokens += (size_t) n_accepted;
    spec->stats.rejected_tokens += (size_t) (n_drafted - n_accepted);

    if (n_accepted < n_drafted) {
        // The caller is responsible for invoking
        // llama_memory_restore_recurrent() against a snapshot captured
        // before the verify pass, then trimming the KV cache via
        // llama_memory_seq_rm(), then re-decoding the corrected token.
        // This helper only records the rollback for stats purposes.
        spec->stats.rollbacks++;
    }
}

void common_speculative_mtp_note_forward_pass(struct common_speculative_mtp * spec) {
    if (spec == nullptr) {
        return;
    }
    spec->stats.forward_passes++;
}

void common_speculative_mtp_note_committed(struct common_speculative_mtp * spec, size_t n) {
    if (spec == nullptr) {
        return;
    }
    spec->stats.total_tokens += n;
}

const common_speculative_mtp_stats & common_speculative_mtp_get_stats(const struct common_speculative_mtp * spec) {
    static const common_speculative_mtp_stats empty;
    if (spec == nullptr) {
        return empty;
    }
    return spec->stats;
}

void common_speculative_mtp_print_stats(const struct common_speculative_mtp * spec, double elapsed_s) {
    if (spec == nullptr) {
        return;
    }

    const auto & s = spec->stats;
    const double tps = elapsed_s > 0.0 ? (double) s.total_tokens / elapsed_s : 0.0;

    const size_t denom = s.draft_tokens > 0 ? s.draft_tokens : 1;
    const double acc_pct = 100.0 * (double) s.accepted_tokens / (double) denom;
    const double rej_pct = 100.0 * (double) s.rejected_tokens / (double) denom;

    LOG_INF("\n");
    LOG_INF("============================================================\n");
    LOG_INF("MTP speculative decoding stats:\n");
    LOG_INF("  Tokens generated:    %zu\n", s.total_tokens);
    LOG_INF("  Wall time:           %.2fs\n", elapsed_s);
    LOG_INF("  Throughput:          %.1f tok/s\n", tps);
    LOG_INF("  Forward passes:      %zu\n", s.forward_passes);
    LOG_INF("  Draft calls:         %zu\n", s.draft_calls);
    LOG_INF("  Draft tokens:        %zu\n", s.draft_tokens);
    LOG_INF("  Accepted:            %zu (%.0f%%)\n", s.accepted_tokens, acc_pct);
    LOG_INF("  Rejected:            %zu (%.0f%%)\n", s.rejected_tokens, rej_pct);
    LOG_INF("  Rollbacks:           %zu\n", s.rollbacks);
    LOG_INF("============================================================\n");
}
