#pragma once

#include <kerutils/kerutils.cuh>

#include "small_heads.h"
#include "../../small_heads.cuh"
#include "components/dequant.h"
#include "cuda_kernels/kv_cache_format.h"

namespace legacy::sm90::decode::sparse {

template<ModelType MODEL_TYPE, bool SPLIT, int NUM_WARPS>
struct SmallHeadsDecodeIO {
    static_assert(MODEL_TYPE == ModelType::V4 || MODEL_TYPE == ModelType::V32);
    using Format = KVCacheFormat<MODEL_TYPE>;
    static constexpr int D_QK = Format::D_QK;
    static constexpr int BK = small_heads_dispatch::KEY_TILE_SIZE;
    static constexpr int ROW_BYTES = Format::TMA_K_STRIDE;
    static constexpr int SCALE_ROW_BYTES = Format::NUM_SCALES_EACH_TOKEN * Format::SCALE_BYTES;
    static constexpr int ROPE_OFFSET = Format::QUANT_BYTES + (Format::IS_V32 ? SCALE_ROW_BYTES : 0);

    SparseAttnDecodeParams params;
    static constexpr int WARPS = NUM_WARPS, PV_FRAGS = 64 / WARPS;
    static constexpr bool STAGE_SCALES = WARPS == 4;
    // V4 scales live after all data rows in a page; pad their staging slot to
    // 16 bytes so each raw KV row remains aligned for vector loads.
    static constexpr int RAW_STRIDE = ROW_BYTES + (STAGE_SCALES && !Format::IS_V32 ? 16 : 0);
    int num_splits;

    template<int DIM, int BH>
    struct Shared : small_heads::SmallHeadsShared<DIM, BH> {
        const uint8_t* kv_rows[BK];
        union {
            const uint8_t* scale_rows[BK];
            __nv_bfloat16 decoded_scales[BK * 8];
        };
        static_assert(DIM == D_QK);
        __align__(16) uint8_t raw_kv[BK * RAW_STRIDE];
    };

    __device__ int64_t q_row(int query) const {
        return int64_t(query / params.s_q) * params.stride_q_b
            + int64_t(query % params.s_q) * params.stride_q_s_q;
    }
    __device__ int key_length(int query, bool extra) const {
        const int* lengths = extra ? params.extra_topk_length : params.topk_length;
        int width = extra ? params.extra_topk : params.topk;
        return lengths ? min(width, max(0, lengths[query / params.s_q])) : width;
    }
    __device__ int length(int query) const {
        return ((key_length(query, false) + BK - 1) / BK
              + (key_length(query, true) + BK - 1) / BK) * BK;
    }
    __device__ int begin(int length) const {
        return blockIdx.y * ((length / BK + num_splits - 1) / num_splits) * BK;
    }
    __device__ int end(int length) const {
        return min(length, int(blockIdx.y + 1) * ((length / BK + num_splits - 1) / num_splits) * BK);
    }

    template<typename Shared>
    __device__ __forceinline__ void prepare_kv_rows(Shared& smem, int query, int start) const {
        const int tid = threadIdx.x, batch = query / params.s_q, seq = query % params.s_q;
        const int orig_tiles = (key_length(query, false) + BK - 1) / BK;
        const bool extra = start / BK >= orig_tiles;
        const int rel_start = start - (extra ? orig_tiles * BK : 0);
        const int count = key_length(query, extra);
        const auto* cache = reinterpret_cast<const uint8_t*>(extra ? params.extra_kv : params.kv);
        const int page_size = extra ? params.extra_page_block_size : params.page_block_size;
        const int page_stride = extra ? params.stride_extra_kv_block : params.stride_kv_block;
        const int num_pages = extra ? params.extra_num_blocks : params.num_blocks;
        const int* indices = extra ? params.extra_indices : params.indices;
        const int64_t indices_row = int64_t(batch) * (extra ? params.stride_extra_indices_b : params.stride_indices_b)
                                 + int64_t(seq) * (extra ? params.stride_extra_indices_s_q : params.stride_indices_s_q);
        if (tid < BK) {
            int index = rel_start + tid < count ? indices[indices_row + rel_start + tid] : -1;
            const bool valid = index >= 0 && int64_t(index) < int64_t(num_pages) * page_size;
            smem.indices[tid] = valid ? index : -1;
            smem.kv_rows[tid] = cache;
            if constexpr (STAGE_SCALES && !Format::IS_V32) smem.scale_rows[tid] = cache;
            if (valid) {
                const int page = uint32_t(index) / uint32_t(page_size);
                const int slot = uint32_t(index) % uint32_t(page_size);
                const auto* page_ptr = cache + int64_t(page) * page_stride;
                const auto* data = page_ptr + int64_t(slot) * ROW_BYTES;
                smem.kv_rows[tid] = data;
                if constexpr (STAGE_SCALES) {
                    if constexpr (!Format::IS_V32)
                        smem.scale_rows[tid] = page_ptr + page_size * ROW_BYTES + slot * SCALE_ROW_BYTES;
                } else if constexpr (!Format::IS_V32) {
                    const auto* scales = page_ptr + page_size * ROW_BYTES + slot * SCALE_ROW_BYTES;
                    unsigned long long encoded;
                    if ((reinterpret_cast<uintptr_t>(scales) & 7) == 0)
                        encoded = __ldg(reinterpret_cast<const unsigned long long*>(scales));
                    else
                        memcpy(&encoded, scales, 8);
                    #pragma unroll
                    for (int scale = 0; scale < Format::D_NOPE / Format::QUANT_TILE_SIZE; ++scale)
                        smem.decoded_scales[tid * 8 + scale] = __float2bfloat16(exp2f(float(uint8_t(encoded >> (scale * 8))) - 127.f));
                } else {
                    float4 scales;
                    if ((reinterpret_cast<uintptr_t>(data + Format::QUANT_BYTES) & 15) == 0)
                        scales = load_128b_from_gmem<float4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(data + Format::QUANT_BYTES);
                    else
                        memcpy(&scales, data + Format::QUANT_BYTES, 16);
                    #pragma unroll
                    for (int scale = 0; scale < Format::NUM_SCALES_EACH_TOKEN; ++scale)
                        smem.decoded_scales[tid * 8 + scale] = __float2bfloat16((&scales.x)[scale]);
                }
            }
        }
        __syncthreads();
    }

    template<int BH, typename Shared>
    __device__ __forceinline__ void stage_kv(Shared& smem) const {
        constexpr int THREADS = BH / 16 * WARPS * 32;
        const int tid = threadIdx.x;
        #pragma unroll
        for (int i = tid; i < BK * (ROW_BYTES / 16); i += THREADS) {
            const int row = i / (ROW_BYTES / 16), col = (i % (ROW_BYTES / 16)) * 16;
            const auto* src = smem.kv_rows[row] + col;
            auto* dst = smem.raw_kv + row * RAW_STRIDE + col;
            const bool valid = smem.indices[row] >= 0;
            if ((reinterpret_cast<uintptr_t>(src) & 15) == 0) {
                const unsigned addr = cute::cast_smem_ptr_to_uint(dst);
                const int bytes = valid ? 16 : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(addr), "l"(src), "r"(bytes));
            } else {
                uint4 value{};
                if (valid) memcpy(&value, src, 16);
                *reinterpret_cast<uint4*>(dst) = value;
            }
        }
        if constexpr (STAGE_SCALES && !Format::IS_V32) {
            if (tid < BK) {
                const auto* src = smem.scale_rows[tid];
                auto* dst = smem.raw_kv + tid * RAW_STRIDE + ROW_BYTES;
                const bool valid = smem.indices[tid] >= 0;
                if ((reinterpret_cast<uintptr_t>(src) & 7) == 0) {
                    const unsigned addr = cute::cast_smem_ptr_to_uint(dst);
                    const int bytes = valid ? 8 : 0;
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(addr), "l"(src), "r"(bytes));
                } else {
                    unsigned long long value = 0;
                    if (valid) memcpy(&value, src, 8);
                    *reinterpret_cast<unsigned long long*>(dst) = value;
                }
            }
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
        __syncthreads();
    }

    template<int BH, typename Shared>
    __device__ __forceinline__ void dequantize_kv(Shared& smem) const {
        constexpr int THREADS = BH / 16 * WARPS * 32, LD = Shared::LD;
        const int tid = threadIdx.x;
        #pragma unroll 4
        for (int i = tid; i < BK * (D_QK / 8); i += THREADS) {
            const int row = i / (D_QK / 8), col = (i % (D_QK / 8)) * 8;
            const int index = smem.indices[row];
            bf16x8 value{};
            if (index >= 0) {
                const auto* data = smem.raw_kv + row * RAW_STRIDE;
                if (col < Format::D_NOPE) {
                    fp8x8 raw;
                    *reinterpret_cast<unsigned long long*>(&raw) = *reinterpret_cast<const unsigned long long*>(data + col);
                    __nv_bfloat16 scale;
                    if constexpr (STAGE_SCALES) {
                        if constexpr (!Format::IS_V32)
                            scale = __float2bfloat16(exp2f(float(data[ROW_BYTES + col / Format::QUANT_TILE_SIZE]) - 127.f));
                        else
                            scale = __float2bfloat16(reinterpret_cast<const float*>(data + Format::QUANT_BYTES)[col / Format::QUANT_TILE_SIZE]);
                    } else {
                        scale = smem.decoded_scales[row * 8 + col / Format::QUANT_TILE_SIZE];
                    }
                    value = cvt_fp8x8_bf16x8(raw, __bfloat162bfloat162(scale));
                } else {
                    const auto* rope = data + ROPE_OFFSET + (col - Format::D_NOPE) * 2;
                    *reinterpret_cast<uint4*>(&value) = *reinterpret_cast<const uint4*>(rope);
                }
            }
            *reinterpret_cast<uint4*>(smem.buffer.kv + row * LD + col) = *reinterpret_cast<const uint4*>(&value);
        }
    }

    // Complete each producer stage before reusing its shared-memory storage.
    // The common attention kernel synchronizes after the final dequantization.
    template<int DIM, int BH, typename Shared>
    __device__ void load_kv(Shared& smem, int query, int start) const {
        static_assert(DIM == D_QK);
        prepare_kv_rows(smem, query, start);
        stage_kv<BH>(smem);
        dequantize_kv<BH>(smem);
    }

    template<int DIM, int BH, typename Shared>
    __device__ void store(Shared& smem, float (&acc)[PV_FRAGS][4], int query, int head_base) const {
        const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
        const int group = warp / WARPS, kw = warp % WARPS, grp = lane / 4, tig = lane % 4;
        const int batch = query / params.s_q, seq = query % params.s_q;
        const int split = batch * num_splits + blockIdx.y;
        if constexpr (SPLIT) {
            // The existing combine kernel consumes cumulative per-request split offsets.
            if (tid == 0 && seq == 0 && blockIdx.y == 0) {
                params.num_splits_ptr[batch] = batch * num_splits;
                if (batch == params.b - 1) params.num_splits_ptr[batch + 1] = params.b * num_splits;
            }
        }
        if (tid < BH) {
            const int head = head_base + tid;
            const float l = smem.denominator[tid], m = smem.maximum[tid];
            float denominator = l;
            if constexpr (!SPLIT) {
                if (params.attn_sink && head < params.h_q)
                    denominator += exp2f(params.attn_sink[head] * CUDART_L2E_F - m);
            }
            smem.alpha[tid] = l == 0.f ? 0.f : 1.f / denominator;
            if (head < params.h_q) {
                if constexpr (SPLIT) {
                    params.lse_accum[int64_t(split) * params.stride_lse_accum_split + seq * params.stride_lse_accum_s_q + head]
                        = l == 0.f ? -CUDART_INF_F : log2f(l) + m;
                } else {
                    params.lse[int64_t(batch) * params.stride_lse_b + seq * params.stride_lse_s_q + head]
                        = l == 0.f ? CUDART_INF_F : logf(l) + m * CUDART_LN2_F;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < PV_FRAGS; ++j) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = group * 16 + grp + half * 8;
                const int head = head_base + row, col = kw * (512 / WARPS) + j * 8 + 2 * tig;
                if (head < params.h_q) {
                    const float factor = smem.alpha[row];
                    const float2 value = {acc[j][half * 2] * factor, acc[j][half * 2 + 1] * factor};
                    if constexpr (SPLIT) {
                        float* dst = params.o_accum + int64_t(split) * params.stride_o_accum_split
                            + seq * params.stride_o_accum_s_q + head * params.stride_o_accum_h_q + col;
                        *reinterpret_cast<float2*>(dst) = value;
                    } else {
                        auto* dst = reinterpret_cast<__nv_bfloat16*>(params.out) + int64_t(batch) * params.stride_o_b
                            + seq * params.stride_o_s_q + head * params.stride_o_h_q + col;
                        dst[0] = __float2bfloat16(value.x);
                        dst[1] = __float2bfloat16(value.y);
                    }
                }
            }
        }
    }
};

template<ModelType MODEL_TYPE, int BH, bool SPLIT>
void launch_small_heads_decode(const SparseAttnDecodeParams& params, int num_splits) {
    using IO = SmallHeadsDecodeIO<MODEL_TYPE, SPLIT, BH == 32 ? 4 : 8>;
    constexpr int D_QK = IO::D_QK;
    constexpr int smem_size = sizeof(typename IO::template Shared<D_QK, BH>);
    auto kernel = small_heads::attention_kernel<D_QK, BH, IO>;
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<dim3(params.b * params.s_q, num_splits), BH / 16 * IO::WARPS * 32, smem_size, params.stream>>>({params, num_splits});
    KU_CHECK_KERNEL_LAUNCH();
}

void run_small_heads_decode(const SparseAttnDecodeParams& params, int num_splits) {
    auto launch = [&]<ModelType MODEL_TYPE, int BH>() {
        if (num_splits > 1) launch_small_heads_decode<MODEL_TYPE, BH, true>(params, num_splits);
        else launch_small_heads_decode<MODEL_TYPE, BH, false>(params, 1);
    };
    KU_ASSERT(params.model_type == ModelType::V4 || params.model_type == ModelType::V32);
    if (params.model_type == ModelType::V4) {
        if (params.h_q == 32) launch.template operator()<ModelType::V4, 32>();
        else launch.template operator()<ModelType::V4, 16>();
    } else {
        if (params.h_q == 32) launch.template operator()<ModelType::V32, 32>();
        else launch.template operator()<ModelType::V32, 16>();
    }
}

}
