// permute_wv_proj: same 32-element-chunk transposition as the sm100
// version, but sm_120a has no TMA (cp.async.bulk), so each 32-byte chunk is
// moved with plain 4-byte global accesses instead of SM90_BULK_COPY_G2S +
// smem permute + SM90_BULK_COPY_S2G. The scale-factor permutation is unchanged.

#include "kernel.h"

#include <kerutils/kerutils.cuh>

namespace sm120::prefill::fused_norm_rope_attn_rope_cast_fwd::permute_wv_proj {

static constexpr uint32_t INPUT_GRAN = 32;
static constexpr uint32_t CHUNK_SIZE = 32;  // along d_o, 32 elements to a chunk
static constexpr uint32_t D_O = 512;

__launch_bounds__(32)
__global__ void permute_wv_proj_kernel(__grid_constant__ const Params params) {
    uint32_t row_idx = blockIdx.x;
    uint32_t wv_group_idx = blockIdx.y;

    const fp8_e4m3* src_row = params.wv_proj
        + (uint64_t)wv_group_idx * params.stride_wv_proj_dim0
        + (uint64_t)row_idx * params.stride_wv_proj_dim1;
    fp8_e4m3* dst_row = params.wv_proj_permuted
        + (uint64_t)wv_group_idx * params.stride_wv_proj_permuted_dim0
        + (uint64_t)row_idx * params.stride_wv_proj_permuted_dim1;

    // Permute SF (same logic as sm100; plain global i8 accesses).
    uint32_t num_sf_per_head = D_O / 32;  // The output granularity is fixed to 32
    uint32_t num_sf = params.wv_group_size * num_sf_per_head;
    for (uint32_t i = threadIdx.x; i < num_sf; i += 32) {
        uint32_t head_idx = i / num_sf_per_head;
        uint32_t head_dim_chunk_idx = i % num_sf_per_head;
        uint32_t input_sf_idx_in_row = i;
        uint8_t cur_sf = *((uint8_t*)(params.scale_factors + wv_group_idx * params.stride_scale_factors_dim0 + row_idx + (input_sf_idx_in_row / 4) * params.stride_scale_factors_dim2) + input_sf_idx_in_row % 4);
        uint32_t output_sf_idx_in_row = head_idx + head_dim_chunk_idx * params.wv_group_size;
        *((uint8_t*)(params.scale_factors_permuted + wv_group_idx * params.stride_scale_factors_permuted_dim0 + row_idx + (output_sf_idx_in_row / 4) * params.stride_scale_factors_permuted_dim2) + output_sf_idx_in_row % 4) = cur_sf;
    }

    // Permute the row: move each 32-byte chunk (head, dim_chunk) to
    // (dim_chunk, head). Direct global 4-byte copies keep alignment safe.
    const uint32_t num_chunks = params.wv_group_size * (D_O / CHUNK_SIZE);
    for (uint32_t i = threadIdx.x; i < num_chunks; i += 32) {
        uint32_t head_idx = i / (D_O / CHUNK_SIZE);
        uint32_t head_dim_chunk_idx = i % (D_O / CHUNK_SIZE);
        uint32_t dst_i = head_dim_chunk_idx * params.wv_group_size + head_idx;
        const uint32_t* sc = (const uint32_t*)(src_row + (uint64_t)i * CHUNK_SIZE);
        uint32_t* dc = (uint32_t*)(dst_row + (uint64_t)dst_i * CHUNK_SIZE);
        CUTE_UNROLL
        for (int k = 0; k < (int)(CHUNK_SIZE / 4); ++k) {
            dc[k] = sc[k];
        }
    }
}

void run_permute_wv_proj_kernel(const Params& params) {
    KU_ASSERT(params.d_o == D_O);
    KU_ASSERT(params.input_gran == INPUT_GRAN);
    KU_ASSERT(D_O % (params.input_gran * 4) == 0);
    permute_wv_proj_kernel<<<dim3(params.d_proj_out, params.n_wv_group), 32, 0, params.stream>>>(params);
}

}
