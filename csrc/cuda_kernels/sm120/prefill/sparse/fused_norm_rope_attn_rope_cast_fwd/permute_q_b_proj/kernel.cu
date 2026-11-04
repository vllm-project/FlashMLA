// permute_q_b_proj: same row permutation as the sm100 version, but
// sm_120a has no TMA (cp.async.bulk), so the row move is a plain per-thread
// 4-byte global copy instead of SM90_BULK_COPY_G2S/S2G. One warp permutes one
// row; the scale-factor permutation is unchanged (plain global i32 reads).

#include "kernel.h"

#include <kerutils/kerutils.cuh>

namespace sm120::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_q_b_proj {

__launch_bounds__(32)
__global__ void permute_q_b_proj_kernel(__grid_constant__ const Params params) {
    uint32_t row_idx = blockIdx.x;
    uint32_t head_idx = row_idx / params.d_q;
    uint32_t head_dim_idx = row_idx % params.d_q;
    uint32_t out_row_idx = head_dim_idx % 16 + head_idx * 16 + (head_dim_idx / 16u) * (params.h_q * 16);
    uint32_t num_scales_per_row = params.q_lora_rank / params.gran / 4;

    // Copy the row payload [q_lora_rank] bytes. q_lora_rank is a multiple of 4
    // (1024/1536), and both rows are 4-byte aligned, so a uint32 walk is safe.
    const uint32_t* src = (const uint32_t*)(params.q_b_proj + row_idx * params.stride_q_b_dim0);
    uint32_t* dst = (uint32_t*)(params.q_b_proj_permuted + out_row_idx * params.stride_q_b_permuted_dim0);
    const uint32_t num_words = params.q_lora_rank >> 2;
    CUTE_UNROLL
    for (uint32_t i = threadIdx.x; i < num_words; i += 32) {
        dst[i] = src[i];
    }

    for (uint32_t scale_idx = threadIdx.x; scale_idx < num_scales_per_row; scale_idx += 32) {
        auto cur_scale = params.scale_factors[row_idx + scale_idx * params.stride_scale_factors_dim1];
        params.scale_factors_permuted[out_row_idx + scale_idx * params.stride_scale_factors_permuted_dim1] = cur_scale;
    }
}

void run_permute_q_b_proj_kernel(const Params& params) {
    permute_q_b_proj_kernel<<<params.h_q*params.d_q, 32, 0, params.stream>>>(params);
}

}
