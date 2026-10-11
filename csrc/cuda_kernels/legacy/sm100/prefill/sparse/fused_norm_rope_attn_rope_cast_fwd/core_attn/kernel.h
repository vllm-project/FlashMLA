#pragma once

#include "params.h"

// The parameter struct is shared with the current kernel: the two differ only in which
// ModelType values they accept, so the API fills one struct whichever it dispatches to.
#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.h"

namespace legacy::sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

using ::sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::ParamsTemplate;
using ::sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::ParamT;

// Compile-time configuration for the fused_norm_rope_attn_rope_cast core attention kernel.
// Notes:
//   - Only Decode mode is instantiated here. V4 and V41 prefill are identical (the KV of a
//     prefill is bf16), so prefill goes to the current kernel.
//   - MODEL_TYPE (the format of `kv`) is V4; EXTRA_MODEL_TYPE equals it.
struct Config {
    SparseAttnFwdMode FWD_MODE;
    ModelType MODEL_TYPE;          // Format of `kv`: V4
    ModelType EXTRA_MODEL_TYPE;    // Format of `extra_kv`, equal to MODEL_TYPE
    uint32_t H_Q;
    bool ENABLE_Q_NORM;
};

template<Config CONFIG>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel(const ParamT<CONFIG.FWD_MODE>& params);

}
