/*
 * Taken from FlashMLA PR https://github.com/deepseek-ai/FlashMLA/pull/54
 * originally authored by @endurehero
 */

#pragma once

#include "../kernels/params.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// FP8-specific extension of the original Flash_fwd_mla_params
struct Flash_fwd_mla_params_fp8 : public Flash_fwd_mla_params {
    int h_h_k_ratio;
    float* __restrict__ descale_q_ptr = nullptr;
    float* __restrict__ descale_k_ptr = nullptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params_fp8 &params, cudaStream_t stream);
