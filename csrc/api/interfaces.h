#pragma once

#include <optional>
#include <tuple>
#include <vector>

#include "common.h"
#include "kernels/sm100/prefill/dense/interface.h"

std::vector<Tensor> sparse_attn_prefill_interface(
    const Tensor &q,
    const Tensor &kv,
    const Tensor &indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<Tensor> &attn_sink,
    const std::optional<Tensor> &topk_length,
    const std::optional<Tensor> &out_);

std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>>
sparse_attn_decode_interface(
    const Tensor &q,
    const Tensor &kv,
    const Tensor &indices,
    const std::optional<Tensor> &topk_length,
    const std::optional<Tensor> &attn_sink,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    const std::optional<Tensor> &extra_kv,
    const std::optional<Tensor> &extra_indices,
    const std::optional<Tensor> &extra_topk_length,
    int64_t d_v,
    double sm_scale,
    const std::optional<Tensor> &out_);

std::tuple<Tensor, Tensor, std::optional<Tensor>, std::optional<Tensor>>
dense_attn_decode_interface(
    Tensor q,
    const Tensor &kcache,
    int64_t head_size_v,
    const Tensor &seqlens_k,
    const Tensor &block_table,
    double softmax_scale,
    bool is_causal,
    std::optional<Tensor> tile_scheduler_metadata,
    std::optional<Tensor> num_splits,
    const std::optional<Tensor> &out_);

std::vector<Tensor> fused_norm_rope_attn_rope_cast_fwd(
    const Tensor &q,
    const Tensor &kv,
    const Tensor &indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<Tensor> &attn_sink,
    const std::optional<Tensor> &topk_length,
    bool enable_q_norm,
    double rms_norm_eps,
    const Tensor &token_positions,
    bool is_rope_neox_style,
    int64_t rope_dim,
    const Tensor &cos_sin_cache,
    int64_t n_wv_group,
    int64_t num_per_channels,
    bool use_tma_aligned_col_major_sf,
    bool round_sf,
    bool use_packed_ue8m0);

std::vector<Tensor> fused_norm_rope_attn_rope_cast_decode(
    const Tensor &q,
    const Tensor &kv,
    const Tensor &indices,
    double sm_scale,
    int64_t d_v,
    const std::optional<Tensor> &attn_sink,
    const std::optional<Tensor> &topk_length,
    const std::optional<Tensor> &extra_kv,
    const std::optional<Tensor> &extra_indices,
    const std::optional<Tensor> &extra_topk_length,
    bool enable_q_norm,
    double rms_norm_eps,
    const Tensor &token_positions,
    bool is_rope_neox_style,
    int64_t rope_dim,
    const Tensor &cos_sin_cache,
    int64_t n_wv_group,
    int64_t num_per_channels,
    bool use_tma_aligned_col_major_sf,
    bool round_sf,
    bool use_packed_ue8m0);

std::vector<Tensor> permute_q_b_proj(
    const Tensor &q_b_proj,
    const Tensor &scale_factors,
    int64_t h_q,
    int64_t d_q);

std::vector<Tensor> permute_wv_proj(
    const Tensor &wv_proj,
    const Tensor &scale_factors,
    int64_t wv_group_size,
    int64_t d_o);
