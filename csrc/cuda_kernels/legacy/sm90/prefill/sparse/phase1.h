#pragma once

#include "params.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace legacy::sm90::prefill::sparse_fwd {

struct EtapSplitParams : SparseAttnFwdParams {
    int etap_splits = 1;
    float* etap_out_accum = nullptr; // [s_q, etap_splits, h_q, d_v]
    float* etap_lse_accum = nullptr; // [s_q, etap_splits, h_q], natural logarithm
    float* etap_max_accum = nullptr;
};

inline int get_etap_splits(const SparseAttnFwdParams& params, bool allow_auto) {
    if (params.s_q == 0 || params.s_q > 32 || (params.h_q != 8 && params.h_q != 16)) return 1;
    const char* mma = std::getenv("USE_MMA_FOR_SMALL_N_HEAD");
    if (mma && mma[0] != '\0' && std::strcmp(mma, "0") != 0) return 1;
    int requested = 0;
    if (const char* value = std::getenv("FLASH_MLA_ETAP_SPLITS")) {
        char* end;
        long splits = std::strtol(value, &end, 10);
        if (end == value || *end != '\0' || splits < 1 || splits > 32)
            throw std::invalid_argument("FLASH_MLA_ETAP_SPLITS must be an integer in [1, 32]");
        requested = static_cast<int>(splits);
    }
    // A device length can make a wide index row nearly empty. Avoid a host read
    // (and synchronization); callers with known long rows can opt in explicitly.
    if (requested == 1 || (requested == 0 && (!allow_auto || params.topk_length != nullptr))) return 1;
    int limit = std::min({32, params.num_sm / params.s_q, params.topk / 128});
    if (requested == 0 && (params.topk < 640 || (params.topk < 1024 && limit < 4))) return 1;
    if (requested > 0 && requested <= limit) return requested;
    int splits = 1;
    while (splits * 2 <= limit && (requested == 0 || splits * 2 <= requested)) splits *= 2;
    return splits;
}

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_fwd_phase1_kernel(const SparseAttnFwdParams& params);

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_etap_split_kernel(const EtapSplitParams& params);

}
