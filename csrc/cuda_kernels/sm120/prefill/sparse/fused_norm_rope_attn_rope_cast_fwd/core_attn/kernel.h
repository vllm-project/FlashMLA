#pragma once

#include <cutlass/float8.h>

#include "params.h"

// The sm120 fused core_attn kernel shares the exact same parameter structs and
// Config as its sm100 counterpart (same fields, same layout); only the kernel
// implementation differs (register mma.sync vs tcgen05). Reuse the sm100 types
// so the API can build one Params object and dispatch it to either arch.
#include "cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.h"

namespace sm120::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

using fp8_e4m3 = cutlass::float_e4m3_t;

using Config = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::Config;
template<typename Base>
using ParamsTemplate = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::ParamsTemplate<Base>;
template<SparseAttnFwdMode FWD_MODE>
using ParamT = sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::ParamT<FWD_MODE>;

// Distinct name from the sm100 entry (run_fused_norm_rope_attn_rope_cast_fwd_kernel):
// both live in the same TU (this header reuses the sm100 types), and EDG applies
// argument-dependent lookup even to explicit instantiations, which would make an
// identically-named template ambiguous for the sm100-typed ParamsTemplate argument.
//
// The definition in kernel.cuh MUST spell its parameter exactly as
// `const ParamT<CONFIG.FWD_MODE>&` — EDG treats `const typename Kernel<CONFIG>::Params&`
// (even though it deduces to the same type) as a distinct dependent signature, which
// turns the definition into a second overload and makes every explicit instantiation
// in the .cu files ambiguous ("more than one instance ... matches the argument list").
template<Config CONFIG>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel_sm120(const ParamT<CONFIG.FWD_MODE>& params);

}
