#pragma once

#include <algorithm>
#include <initializer_list>

#include "params.h"

namespace legacy::sm90::decode::sparse {

// Host dispatch policy. These thresholds are tuned on H20 with cold L2.
// Keep layout eligibility separate from the workload heuristic so additional
// cache formats or persistent kernels can extend one without changing the other.
namespace small_heads_dispatch {

constexpr int KEY_TILE_SIZE = 32;
constexpr int PERSISTENT_KEY_TILE_SIZE = 64;

template<typename T>
inline bool rows_are_aligned(const T* data, int alignment, std::initializer_list<int> strides) {
    if (reinterpret_cast<uintptr_t>(data) % alignment != 0) return false;
    for (int stride : strides) {
        if ((int64_t(stride) * sizeof(T)) % alignment != 0) return false;
    }
    return true;
}

inline bool persistent_layout_is_supported(const SparseAttnDecodeParams& params, bool have_extra_kcache) {
    const bool main_aligned =
        rows_are_aligned(params.q, 16, {params.stride_q_b, params.stride_q_s_q, params.stride_q_h_q})
        && rows_are_aligned(reinterpret_cast<const uint8_t*>(params.kv), 16, {params.stride_kv_block})
        && rows_are_aligned(params.indices, 8, {params.stride_indices_b, params.stride_indices_s_q})
        && rows_are_aligned(params.out, 16, {params.stride_o_b, params.stride_o_s_q, params.stride_o_h_q});
    const bool extra_aligned = !have_extra_kcache || (params.extra_num_blocks > 0
        && rows_are_aligned(reinterpret_cast<const uint8_t*>(params.extra_kv), 16, {params.stride_extra_kv_block})
        && rows_are_aligned(params.extra_indices, 8, {params.stride_extra_indices_b, params.stride_extra_indices_s_q}));
    return main_aligned && extra_aligned;
}

}  // namespace small_heads_dispatch

inline bool can_use_persistent_small_heads_decode(
    const SparseAttnDecodeParams& params, bool have_extra_kcache, bool batch_invariant) {
    using namespace small_heads_dispatch;
    const int num_queries = params.b * params.s_q;
    const bool enough_work = num_queries >= 32
        || (params.topk + params.extra_topk > 128 && num_queries >= 4);
    const bool supported_features = params.h_q == 32 && !batch_invariant
        && !params.topk_length && !params.extra_topk_length && params.num_blocks > 0
        && params.topk % PERSISTENT_KEY_TILE_SIZE == 0
        && params.extra_topk % PERSISTENT_KEY_TILE_SIZE == 0
        && (params.model_type == ModelType::V4 || !have_extra_kcache);
    // TMA masks unused rows of the 64-row WGMMA tile. Dynamic lengths and
    // layouts that cannot use TMA stay on the native small-tile kernel.
    return enough_work && supported_features && persistent_layout_is_supported(params, have_extra_kcache);
}

inline bool small_heads_combine_output_is_aligned(const SparseAttnDecodeParams& params) {
    // The shared combine kernel writes 8 bytes at a time; the unsplit kernel
    // also handles output views that do not meet this requirement.
    return small_heads_dispatch::rows_are_aligned(
        params.out, 8, {params.stride_o_b, params.stride_o_s_q, params.stride_o_h_q});
}

inline int small_heads_num_splits(int num_queries, int heads, int topk, int extra_topk, bool force_unsplit) {
    if (force_unsplit) return 1;
    constexpr int tile_size = small_heads_dispatch::KEY_TILE_SIZE;
    const int tiles = (topk + tile_size - 1) / tile_size + (extra_topk + tile_size - 1) / tile_size;
    // Expose individual tiles at low concurrency, but bound partial-output traffic
    // once there are enough queries to occupy the GPU.
    if (num_queries <= 4) return std::min(tiles, 32);
    const int target_ctas = tiles > 4 ? (heads == 32 ? 256 : 512) : 128;
    const int limit = heads == 32 ? 4 : 8;
    return std::min(tiles, std::clamp(target_ctas / num_queries, 1, limit));
}

void run_small_heads_decode(const SparseAttnDecodeParams& params, int num_splits);

}
