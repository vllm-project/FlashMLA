/*
 * Taken from FlashMLA PR https://github.com/deepseek-ai/FlashMLA/pull/54
 * originally authored by @endurehero
 */

#pragma once

#include "../../../params.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

// FP8-specific extension of the original DecodingParams
struct DecodingParams_fp8 : public DecodingParams {
    int h_h_k_ratio;
    float* __restrict__ descale_q_ptr = nullptr;
    float* __restrict__ descale_k_ptr = nullptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla(DecodingParams_fp8 &params, cudaStream_t stream);
