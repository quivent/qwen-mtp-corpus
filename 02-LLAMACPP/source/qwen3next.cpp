#include "models.h"

#include "llama-memory-recurrent.h"

llm_build_qwen3next::llm_build_qwen3next(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    // MTP draft graph: skip the full forward pass and run only the NextN
    // (MTP) head over a single seed token plus the previous hidden state.
    // The host buffer behind `prev_hidden` is uploaded by
    // llama_context::mtp_graph_compute via res->t_mtp_prev_hidden_in.
    if (gtype == LLM_GRAPH_TYPE_MTP) {
        GGML_ASSERT(hparams.nextn_predict_layers > 0 && "MTP graph requested but model has no NextN layers");
        GGML_ASSERT(ubatch.n_tokens == 1 && "MTP graph expects a single seed token per dispatch");

        // Token embedding for the seed token. The standard build_inp_embd
        // path registers an llm_graph_input_embd whose set_input() copies
        // the token id from the ubatch.
        ggml_tensor * tok_embed = build_inp_embd(model.tok_embd);
        cb(tok_embed, "mtp_tok_embed", -1);

        // Previous hidden state input — host-uploaded by the dispatcher.
        ggml_tensor * prev_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, 1);
        ggml_set_input(prev_hidden);
        res->t_mtp_prev_hidden_in = prev_hidden;
        cb(prev_hidden, "mtp_prev_hidden", -1);

        const int il_mtp = (int) (n_layer - hparams.nextn_predict_layers);

        ggml_tensor * mtp_logits = build_mtp_head(prev_hidden, tok_embed, il_mtp, nullptr);
        cb(mtp_logits, "result_output", -1);
        res->t_logits = mtp_logits;

        ggml_build_forward_expand(gf, mtp_logits);
        return;
    }

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    auto * inp = build_inp_mem_hybrid();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Determine layer type and build appropriate attention mechanism
        if (hparams.is_recurrent(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Full attention layer
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        // Save the tensor before post-attention norm for residual connection
        ggml_tensor * ffn_residual = cur;

        // Post-attention norm
        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        // FFN layer (MoE or dense) - without residual connection
        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        // Residual connection for FFN - add to the tensor from before post_attention_layernorm
        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_moe", il);

        // Input for next layer
        inpL = cur;
    }
    cur = inpL;

    // Final norm
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// utility to get one slice from the third dimension
// input dim:  [x, y, c, b]
// output dim: [x, y, 1, b]
static ggml_tensor * get_slice_2d(ggml_context * ctx0, ggml_tensor * t, int64_t c) {
    return ggml_view_4d(ctx0, t, t->ne[0], t->ne[1], 1, t->ne[3],
        t->nb[1], t->nb[2], t->nb[3], t->nb[2] * c);
}

ggml_tensor * llm_build_qwen3next::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llm_build_qwen3next::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur);
    cb(Qcur_full, "Qcur_full", il);

    Qcur_full = ggml_reshape_4d(ctx0, Qcur_full, n_embd_head * 2, n_head, n_tokens, 1);

    // Split Q projection into query and gate
    // The split should be along dimension 0 (the feature dimension)
    ggml_tensor * Qcur = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
                                            Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], 0);
    cb(Qcur, "Qcur_view", il);

    ggml_tensor * gate =
        ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens, 1,
                     Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], n_embd_head * ggml_element_size(Qcur_full));
    cb(gate, "gate", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
    cb(Vcur, "Vcur", il);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    Qcur = ggml_rope_ext(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    Kcur = ggml_rope_ext(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base,
            freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    // TODO: CUDA is missing non-contiguous unary ops. when implemented: remove this cont
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);

    gate = ggml_sigmoid(ctx0, gate);
    cb(gate, "gate_sigmoid", il);

    gate = ggml_reshape_2d(ctx0, gate, n_embd_head * n_head, n_tokens);

    cur = ggml_mul(ctx0, cur, gate);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur);
    cb(cur, "attn_output", il);

    return cur;
}

std::pair<ggml_tensor *, ggml_tensor *> llm_build_qwen3next::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    if (model.layers[il].wqkv) {
        // optimized path
        ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input);
        qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
        cb(qkv_mixed, "linear_attn_qkv_mixed", il);

        ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input);
        cb(z, "z", il);

        return { qkv_mixed, z };
    } else {
        // legacy (slower) path
        ggml_tensor * mixed_qkvz = build_lora_mm(model.layers[il].ssm_in, input);
        cb(mixed_qkvz, "linear_attn_mixed_qkvz", il);

        int64_t       qkvz_new_dim        = 2 * head_k_dim + 2 * head_v_dim * (num_v_heads / num_k_heads);
        ggml_tensor * mixed_qkvz_reshaped = ggml_reshape_4d(ctx0, mixed_qkvz, qkvz_new_dim, num_k_heads, n_seq_tokens, n_seqs);

        // Split mixed_qkvz into query, key, value, z
        int64_t split_sizes_qkvz[4] = {
            head_k_dim,                              // query size
            head_k_dim,                              // key size
            head_v_dim * num_v_heads / num_k_heads,  // value size
            head_v_dim * num_v_heads / num_k_heads   // z size
        };

        ggml_tensor * query =
            ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[0], num_k_heads, n_seq_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3], 0);
        cb(query, "q", il);

        ggml_tensor * key = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[1], num_k_heads, n_seq_tokens, n_seqs,
                                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                                        split_sizes_qkvz[0] * ggml_element_size(mixed_qkvz_reshaped));
        cb(key, "k", il);

        ggml_tensor * value =
            ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[2], num_k_heads, n_seq_tokens, n_seqs,
                        mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                        (split_sizes_qkvz[0] + split_sizes_qkvz[1]) * ggml_element_size(mixed_qkvz_reshaped));
        cb(value, "v", il);

        ggml_tensor * z = ggml_view_4d(ctx0, mixed_qkvz_reshaped, split_sizes_qkvz[3], num_k_heads, n_seq_tokens, n_seqs,
                                    mixed_qkvz_reshaped->nb[1], mixed_qkvz_reshaped->nb[2], mixed_qkvz_reshaped->nb[3],
                                    (split_sizes_qkvz[0] + split_sizes_qkvz[1] + split_sizes_qkvz[2]) * ggml_element_size(mixed_qkvz_reshaped));
        z = ggml_cont(ctx0, z);
        cb(z, "z", il);

        // After creating query, key, and value_reshaped, reshape each to flatten the head dimensions
        // query: [head_k_dim, num_k_heads, n_tokens, n_seqs] -> [head_k_dim * num_k_heads, n_tokens, n_seqs]
        ggml_tensor * query_flat = ggml_cont_3d(ctx0, query, head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
        cb(query_flat, "query_flat", il);

        // key: [head_k_dim, num_k_heads, n_tokens, n_seqs] -> [head_k_dim * num_k_heads, n_tokens, n_seqs]
        ggml_tensor * key_flat = ggml_cont_3d(ctx0, key, head_k_dim * num_k_heads, n_seq_tokens, n_seqs);
        cb(key_flat, "key_flat", il);

        // value_reshaped: [head_v_dim, num_v_heads, n_tokens, n_seqs] -> [head_v_dim * num_v_heads, n_tokens, n_seqs]
        ggml_tensor * value_flat = ggml_cont_3d(ctx0, value, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
        cb(value_flat, "value_flat", il);

        // Now concatenate along the feature dimension (dim 0) to get [conv_dim, n_tokens, n_seqs]
        ggml_tensor * qkv_mixed = ggml_concat(ctx0, query_flat, key_flat, 0);
        qkv_mixed               = ggml_concat(ctx0, qkv_mixed, value_flat, 0);
        cb(qkv_mixed, "qkv_mixed", il);

        return { qkv_mixed, z };
    }
}

ggml_tensor * llm_build_qwen3next::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto kv_head = mctx_cur->get_head();

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * mixed_ba = build_lora_mm(model.layers[il].ssm_beta_alpha, cur);
    cb(mixed_ba, "linear_attn_mixed_ba", il);

    // Reshape mixed_ba: [batch, seq_len, hidden_size] -> [batch, seq_len, num_k_heads, 2*num_v_heads/num_k_heads]
    int64_t       ba_new_dim        = 2 * num_v_heads / num_k_heads;
    ggml_tensor * mixed_ba_reshaped = ggml_reshape_4d(ctx0, mixed_ba, ba_new_dim, num_k_heads, n_seq_tokens, n_seqs);

    // Split mixed_ba into b and a (beta and alpha parameters)
    int64_t split_sizes_ba[2] = {
        num_v_heads / num_k_heads,  // beta size
        num_v_heads / num_k_heads   // alpha size
    };

    ggml_tensor * b = ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[0], num_k_heads, n_seq_tokens, n_seqs,
                                   mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3], 0);
    cb(b, "b", il);

    ggml_tensor * a = ggml_view_4d(ctx0, mixed_ba_reshaped, split_sizes_ba[1], num_k_heads, n_seq_tokens, n_seqs,
                                   mixed_ba_reshaped->nb[1], mixed_ba_reshaped->nb[2], mixed_ba_reshaped->nb[3],
                                   split_sizes_ba[0] * ggml_element_size(mixed_ba_reshaped));
    cb(a, "a", il);

    // TODO: CUDA is missing non-contiguous unary ops. when implemented: remove this cont
    b = ggml_cont(ctx0, b);

    ggml_tensor * beta = ggml_sigmoid(ctx0, b);

    // Reshape a to merge head dimensions: [batch, seq_len, num_k_heads, num_v_heads/num_k_heads] -> [batch, seq_len, num_v_heads]
    ggml_tensor * alpha = ggml_cont_3d(ctx0, a, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    // Get convolution states from cache
    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    // Build the convolution states tensor
    ggml_tensor * conv_states = build_rs(inp, conv_states_all, hparams.n_embd_r(), n_seqs);
    cb(conv_states, "conv_states", il);

    // Calculate convolution kernel size
    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    conv_states = ggml_reshape_3d(ctx0, conv_states, conv_kernel_size - 1, conv_channels, n_seqs);
    cb(conv_states, "conv_states_reshaped", il);

    qkv_mixed = ggml_transpose(ctx0, qkv_mixed);
    cb(qkv_mixed, "qkv_mixed_transposed", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, conv_states, qkv_mixed, 0);
    cb(conv_input, "conv_input", il);

    // Update convolution state cache
    // Extract the last (conv_kernel_size - 1) states from conv_input
    ggml_tensor * last_conv_states =
        ggml_view_3d(ctx0, conv_input, conv_kernel_size - 1, conv_channels, n_seqs, conv_input->nb[1],
                     conv_input->nb[2], (conv_input->ne[0] - conv_states->ne[0]) * ggml_element_size(conv_input));
    cb(last_conv_states, "last_conv_states", il);

    ggml_tensor * state_update_target =
        ggml_view_1d(ctx0, conv_states_all, (conv_kernel_size - 1) * conv_channels * n_seqs,
                     kv_head * (conv_kernel_size - 1) * conv_channels * ggml_element_size(conv_states_all));
    cb(state_update_target, "state_update_target", il);

    ggml_build_forward_expand(gf, ggml_cpy(ctx0, last_conv_states, state_update_target));
    cb(conv_states_all, "conv_states_updated", il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);

    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = ggml_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = ggml_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    if (num_k_heads != num_v_heads) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        int64_t repeat_factor = num_v_heads / num_k_heads;

        // repeat interleave: reshape to (repeat part, 1, remaining part), do repeat, then reshape back
        ggml_tensor * q_reshaped = ggml_reshape_3d(ctx0, q_conv, head_k_dim, 1, num_k_heads * n_seq_tokens * n_seqs);
        ggml_tensor * k_reshaped = ggml_reshape_3d(ctx0, k_conv, head_k_dim, 1, num_k_heads * n_seq_tokens * n_seqs);

        // Repeat along the third dimension (the new dimension with size 1)
        ggml_tensor * q_repeated =
            ggml_repeat_4d(ctx0, q_reshaped, head_k_dim, repeat_factor, num_k_heads * n_seq_tokens * n_seqs, 1);
        ggml_tensor * k_repeated =
            ggml_repeat_4d(ctx0, k_reshaped, head_k_dim, repeat_factor, num_k_heads * n_seq_tokens * n_seqs, 1);

        // Reshape back to merge the head and repeat dimensions
        // From [head_dim, num_k_heads, repeat_factor, n_seq_tokens * n_seqs]
        // Back to [head_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs]
        q_conv = ggml_reshape_4d(ctx0, q_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
        k_conv = ggml_reshape_4d(ctx0, k_repeated, head_k_dim, num_k_heads * repeat_factor, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    // Choose between build_delta_net_chunking, build_delta_net_recurrent, and build_delta_net_autoregressive based on n_tokens
    // TODO(split-rec): for speculative decoding with a small draft length (T=N), ideally the
    // in_proj / out_proj matmuls should stay batched at T=N while the GDN recurrence is split
    // into T=1 steps that capture intermediate (conv_state, rnn_state) snapshots after each token,
    // mirroring the MLX reference in ~/mlx-fork/delta_net_rollback.py. ggml currently lacks a
    // graph primitive for "run this sub-graph N times, capture state after each step", so v1
    // uses a coarse snapshot/restore at token boundaries via llama_memory_snapshot_recurrent().
    // Follow-up: expose a split-recurrence graph-build path that writes conv_states_all / ssm_states_all
    // N times in one graph and returns handles to each intermediate state view.
    std::pair<ggml_tensor *, ggml_tensor *> attn_out; // pair of (output, new_state)
    if (n_seq_tokens == 1) {
        attn_out = build_delta_net_autoregressive(q_conv, k_conv, v_conv, gate, beta, state, il);
    } else {
        attn_out = build_delta_net_chunking(q_conv, k_conv, v_conv, gate, beta, state, il);
    }
    ggml_tensor * output    = attn_out.first;
    ggml_tensor * new_state = attn_out.second;
    cb(output, "attn_output", il);
    cb(new_state, "new_state", il);

    // Update the recurrent states
    ggml_build_forward_expand(gf,
            ggml_cpy(ctx0, new_state,
                ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                    kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_tokens, n_seqs, n_heads * head_dim]
    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output);
    cb(cur, "linear_attn_out", il);

    // Reshape back to original dimensions
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llm_build_qwen3next::build_mtp_head(
        ggml_tensor *  prev_hidden,
        ggml_tensor *  tok_embed,
                  int  il_mtp,
        ggml_tensor ** out_hidden) {
    // Reference: MTPHead.__call__ in ~/mlx-fork/fused_gdn.py
    //
    // Step 1: e_norm = enorm(tok_embed); h_norm = hnorm(prev_hidden)
    // Step 2: h = fc(concat([e_norm, h_norm], axis=-1))  (embed first, hidden second)
    // Step 3: residual = h; h' = input_layernorm(h)  -- we reuse `attn_norm` slot on the layer
    // Step 4: gated self-attention (single token, no past cache, no mask)
    // Step 5: residual += o_proj(output * sigmoid(gate))
    // Step 6: residual = h; h' = post_attention_layernorm(h); MLP(SiLU(gate_proj) * up_proj)
    // Step 7: final_norm; lm_head projection (shared_head_head) -> logits
    //
    // Storage mapping:
    //   nextn.enorm            -> enorm
    //   nextn.hnorm            -> hnorm
    //   nextn.eh_proj          -> fc
    //   shared_head_norm       -> final norm
    //   shared_head_head       -> shared lm head
    //
    // The per-MTP-layer attention / MLP weights reuse the normal layer slots on
    // model.layers[il_mtp]: attn_norm, wq/wk/wv/wo, attn_q_norm, attn_k_norm,
    // attn_post_norm, ffn_*_shexp (for dense MLP) OR ffn_*. For Qwen3-Next MTP
    // weights converted into the NextN layer, the per-layer attention/MLP slots
    // follow the same GGUF naming as a normal attention layer at index il_mtp.

    const auto & layer = model.layers[il_mtp];

    // sanity: MTP head requires these to exist
    GGML_ASSERT(layer.nextn.enorm            && "MTP enorm missing");
    GGML_ASSERT(layer.nextn.hnorm            && "MTP hnorm missing");
    GGML_ASSERT(layer.nextn.eh_proj          && "MTP eh_proj missing");
    GGML_ASSERT(layer.nextn.shared_head_norm && "MTP shared_head_norm missing");
    GGML_ASSERT(layer.nextn.shared_head_head && "MTP shared_head_head missing");

    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    // Shape conventions: prev_hidden and tok_embed are [n_embd, 1] column vectors

    // --- 1. Norm + concat + project ---
    ggml_tensor * h_norm = build_norm(prev_hidden, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il_mtp);
    cb(h_norm, "mtp_h_norm", il_mtp);

    ggml_tensor * e_norm = build_norm(tok_embed, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il_mtp);
    cb(e_norm, "mtp_e_norm", il_mtp);

    // Concat along the feature dim: [e; h] -> [2*n_embd, 1]
    // NOTE (CRITICAL): embed first, hidden second. Matches GGUF "eh_proj" name.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, 0);
    cb(concat, "mtp_eh_concat", il_mtp);

    ggml_tensor * h = build_lora_mm(layer.nextn.eh_proj, concat);
    cb(h, "mtp_fc_out", il_mtp);

    // --- 2. Gated self-attention (single token, no past cache, no mask) ---
    ggml_tensor * residual = h;
    ggml_tensor * h_attn = build_norm(h, layer.attn_norm, nullptr, LLM_NORM_RMS, il_mtp);
    cb(h_attn, "mtp_attn_norm", il_mtp);

    // Q projection produces [Q | gate] interleaved per-head (see existing build_layer_attn)
    ggml_tensor * Qcur_full = build_lora_mm(layer.wq, h_attn);
    cb(Qcur_full, "mtp_Qcur_full", il_mtp);

    Qcur_full = ggml_reshape_4d(ctx0, Qcur_full, n_embd_head * 2, n_head, 1, 1);

    ggml_tensor * Qcur = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, 1, 1,
                                      Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3], 0);
    ggml_tensor * gate = ggml_view_4d(ctx0, Qcur_full, n_embd_head, n_head, 1, 1,
                                      Qcur_full->nb[1], Qcur_full->nb[2], Qcur_full->nb[3],
                                      n_embd_head * ggml_element_size(Qcur_full));

    ggml_tensor * Kcur = build_lora_mm(layer.wk, h_attn);
    ggml_tensor * Vcur = build_lora_mm(layer.wv, h_attn);

    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, 1);
    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, 1);

    Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il_mtp);
    Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il_mtp);

    // RoPE at position 0 (the MTP head does not need absolute positions — it's
    // a stateless one-token transformer with no past context; using pos=0 is
    // what the MLX reference does via offset=0 for a fresh draft step).
    ggml_tensor * inp_pos_zero = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_input(inp_pos_zero);
    res->t_mtp_pos_zero_in = inp_pos_zero;

    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos_zero, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos_zero, nullptr,
            n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(Qcur, "mtp_Qcur", il_mtp);
    cb(Kcur, "mtp_Kcur", il_mtp);
    cb(Vcur, "mtp_Vcur", il_mtp);

    // Single-token attention: no past keys, no mask, no cache.
    // Compute manually: softmax(Q @ K^T * scale) @ V
    // Shapes:
    //   Q: [head_dim, n_head,    1]
    //   K: [head_dim, n_head_kv, 1]
    //   V: [head_dim, n_head_kv, 1]
    // For GQA with n_head > n_head_kv, repeat K/V along head axis.
    const float kq_scale = hparams.f_attention_scale == 0.0f
        ? 1.0f / sqrtf(float(n_embd_head))
        : hparams.f_attention_scale;

    if (n_head_kv != n_head) {
        GGML_ASSERT(n_head % n_head_kv == 0);
        const int64_t rep = n_head / n_head_kv;
        Kcur = ggml_repeat_4d(ctx0, Kcur, n_embd_head, rep, n_head_kv, 1);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head, 1);
        Vcur = ggml_repeat_4d(ctx0, Vcur, n_embd_head, rep, n_head_kv, 1);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head, 1);
    }

    // KQ = Q . K (along head_dim) -> [1, 1, n_head]
    // Use ggml_mul_mat which does (K^T @ Q): out[i,j] = sum_k K[k,i]*Q[k,j]
    ggml_tensor * KQ = ggml_mul_mat(ctx0, Kcur, Qcur);
    KQ = ggml_scale(ctx0, KQ, kq_scale);
    KQ = ggml_soft_max(ctx0, KQ);

    // attn = V . KQ -> [head_dim, 1, n_head]
    ggml_tensor * KQV = ggml_mul_mat(ctx0, ggml_cont(ctx0, ggml_transpose(ctx0, Vcur)), KQ);
    ggml_tensor * attn_out = ggml_reshape_2d(ctx0, KQV, n_embd_head * n_head, 1);
    cb(attn_out, "mtp_attn_pregate", il_mtp);

    // Apply the per-head gate (sigmoid)
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, 1);
    gate = ggml_sigmoid(ctx0, gate);
    attn_out = ggml_mul(ctx0, attn_out, gate);

    attn_out = build_lora_mm(layer.wo, attn_out);
    cb(attn_out, "mtp_attn_out", il_mtp);

    h = ggml_add(ctx0, residual, attn_out);

    // --- 3. MLP ---
    residual = h;
    ggml_tensor * h_mlp_norm = build_norm(h, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il_mtp);
    cb(h_mlp_norm, "mtp_mlp_norm", il_mtp);

    // MTP head uses a dense MLP (not MoE) — weights land in ffn_gate/ffn_up/ffn_down
    // on the NextN layer during GGUF conversion. Fall back to shared-expert MLP
    // slots if the dense slots are not populated.
    ggml_tensor * mlp_out;
    if (layer.ffn_up && layer.ffn_gate && layer.ffn_down) {
        mlp_out = build_ffn(h_mlp_norm,
            layer.ffn_up,   NULL, NULL,
            layer.ffn_gate, NULL, NULL,
            layer.ffn_down, NULL, NULL,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il_mtp);
    } else {
        GGML_ASSERT(layer.ffn_up_shexp && layer.ffn_gate_shexp && layer.ffn_down_shexp
                    && "MTP head requires either dense FFN or shared-expert FFN weights");
        mlp_out = build_ffn(h_mlp_norm,
            layer.ffn_up_shexp,   NULL, NULL,
            layer.ffn_gate_shexp, NULL, NULL,
            layer.ffn_down_shexp, NULL, NULL,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il_mtp);
    }
    cb(mlp_out, "mtp_mlp_out", il_mtp);

    h = ggml_add(ctx0, residual, mlp_out);

    if (out_hidden) {
        *out_hidden = h;
    }

    // --- 4. Final norm + shared lm head ---
    ggml_tensor * h_final = build_norm(h, layer.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, il_mtp);
    cb(h_final, "mtp_final_norm", il_mtp);

    ggml_tensor * logits = build_lora_mm(layer.nextn.shared_head_head, h_final);
    cb(logits, "mtp_logits", il_mtp);

    return logits;
}

ggml_tensor * llm_build_qwen3next::build_layer_ffn(ggml_tensor * cur, const int il) {
    // Check if this is an MoE layer
    if (model.layers[il].ffn_gate_inp != nullptr) {
        // MoE branch
        ggml_tensor * moe_out =
            build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
                model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps,
                nullptr,
                n_expert, n_expert_used, LLM_FFN_SILU,
                true, false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
        cb(moe_out, "ffn_moe_out", il);

        // Add shared experts if present - following Qwen3Next reference implementation
        if (model.layers[il].ffn_up_shexp != nullptr) {
            ggml_tensor * ffn_shexp =
                build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            // Apply shared expert gating as in the reference implementation
            // The shared expert has its own gate that is sigmoided
            // Note: ffn_gate_inp_shexp is the shared expert gate (outputs 1 value per token)
            ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
            cb(shared_gate, "shared_expert_gate", il);

            shared_gate = ggml_sigmoid(ctx0, shared_gate);
            cb(shared_gate, "shared_expert_gate_sigmoid", il);

            ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
            cb(ffn_shexp, "ffn_shexp_gated", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        } else {
            cur = moe_out;
        }
    } else {
        // Dense FFN branch (not currently used I believe)
        cur = build_ffn(cur,
            model.layers[il].ffn_up, NULL, NULL,
            model.layers[il].ffn_gate, NULL, NULL,
            model.layers[il].ffn_down, NULL, NULL,
            NULL,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);
    }
    return cur;
}
