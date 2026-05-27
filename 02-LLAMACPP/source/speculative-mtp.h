#pragma once

// MTP (Multi-Token Prediction) speculative decoding helpers.
//
// Unlike the standard common_speculative_* API in speculative.h, which
// assumes a separate draft model, these helpers source draft tokens from
// an MTP head on the target model itself. The target context therefore
// plays both the drafting and verification roles.
//
// The drafting side relies on a C API that is being added to the core
// llama library by a separate agent. We forward-declare it here so this
// file can compile independently; the symbol will be resolved at link
// time once the MTP graph agent merges:
//
//     int32_t llama_mtp_draft(
//         struct llama_context * ctx,
//         llama_token            last_token,
//         int32_t                n_draft,
//         llama_token          * out_tokens,
//         float                * out_logits);   // optional, may be NULL
//
// Contract (as agreed with the MTP graph agent):
//   * `ctx`        - target context whose last decoded token is `last_token`.
//                    The KV cache state is NOT mutated by this call.
//   * `last_token` - the most recently verified / committed token id.
//   * `n_draft`    - maximum number of additional tokens to draft from
//                    the MTP heads (K in the MLX reference).
//   * `out_tokens` - caller-provided buffer of at least `n_draft` slots.
//                    On return, holds the drafted token ids (argmax of
//                    each MTP head) in order.
//   * `out_logits` - optional. If non-NULL, a flat buffer of at least
//                    `n_draft * n_vocab` floats receiving the per-head
//                    logits for downstream stochastic verification.
//   * returns      - number of drafted tokens actually written (0 .. n_draft).
//                    A return value of 0 means the target model exposes no
//                    MTP heads and the caller should fall back to plain
//                    autoregressive decoding.

#include "llama.h"
#include "common.h"

#include <cstdint>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of the core-library MTP drafting entry point.
// The definition is provided by the MTP graph agent; until that lands
// this symbol will be unresolved at link time (compile-only usage here).
int32_t llama_mtp_draft(
        struct llama_context * ctx,
        llama_token            last_token,
        int32_t                n_draft,
        llama_token          * out_tokens,
        float                * out_logits);

#ifdef __cplusplus
}
#endif

struct common_speculative_mtp;

// Aggregate statistics that mirror the MLX reference's stats dict.
struct common_speculative_mtp_stats {
    size_t total_tokens    = 0; // tokens ultimately committed to the output
    size_t draft_tokens    = 0; // tokens proposed by the MTP head
    size_t accepted_tokens = 0; // draft tokens accepted during verification
    size_t rejected_tokens = 0; // draft tokens rejected during verification
    size_t forward_passes  = 0; // target-model forward passes
    size_t rollbacks       = 0; // recurrent-state rollbacks performed
    size_t draft_calls     = 0; // number of llama_mtp_draft invocations
};

// Construct / destroy an MTP speculative helper bound to a target context.
struct common_speculative_mtp * common_speculative_mtp_init(llama_context * ctx_tgt);
void                            common_speculative_mtp_free(struct common_speculative_mtp * spec);

// Draft up to `n_draft` tokens from the target model's MTP head, given
// the id of the most recently verified token. Returns the draft tokens
// in order. An empty result means the target has no MTP heads and the
// caller should fall back to standard single-token decoding.
llama_tokens common_speculative_mtp_draft(
        struct common_speculative_mtp * spec,
        llama_token                     id_last,
        int32_t                         n_draft);

// Greedy verification: compare draft tokens against argmax of the
// corresponding target logits. Returns the number of accepted tokens.
// `logits_for_pos(i)` must return a pointer to the target-model logits
// that predict the token immediately AFTER draft position i-1 (i.e. the
// logits output by the batch slot corresponding to the i-th draft input).
int32_t common_speculative_mtp_verify_greedy(
        struct common_speculative_mtp * spec,
        const llama_tokens            & draft_tokens,
        const float *                 (*logits_for_pos)(void * user_data, int32_t i),
        void                          * user_data,
        int32_t                         n_vocab);

// Inform the helper of how many draft tokens were accepted in the last
// verification pass. Updates stats and invokes the recurrent-state
// rollback hook for the rejected suffix.
void common_speculative_mtp_accept(
        struct common_speculative_mtp * spec,
        int32_t                         n_drafted,
        int32_t                         n_accepted);

// Record a target forward pass.
void common_speculative_mtp_note_forward_pass(struct common_speculative_mtp * spec);

// Bump the "total tokens committed" counter.
void common_speculative_mtp_note_committed(struct common_speculative_mtp * spec, size_t n);

// Accessors.
const common_speculative_mtp_stats & common_speculative_mtp_get_stats(const struct common_speculative_mtp * spec);

// Print a stats summary that mirrors the MLX reference's output format.
void common_speculative_mtp_print_stats(const struct common_speculative_mtp * spec, double elapsed_s);
