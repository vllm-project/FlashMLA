#pragma once

#include "../../small_heads.cuh"

namespace legacy::sm90::prefill::sparse_fwd {

template<bool HAVE_TOPK_LENGTH>
struct SmallHeadsPrefillIO {
    SparseAttnFwdParams params;
    static constexpr int WARPS = 4, PV_FRAGS = 16;
    template<int D_QK, int BH>
    using Shared = small_heads::SmallHeadsShared<D_QK, BH>;

    __device__ int64_t q_row(int query) const {
        return int64_t(query) * params.stride_q_s_q;
    }
    __device__ int length(int query) const {
        if constexpr (HAVE_TOPK_LENGTH)
            return min(params.topk, max(0, params.topk_length[query]));
        return params.topk;
    }
    __device__ int begin(int) const { return 0; }
    __device__ int end(int length) const { return length; }

    template<int D_QK, int BH, typename Shared>
    __device__ void load_kv(Shared& smem, int query, int start) const {
        constexpr int BK = 32, THREADS = BH * 8, LD = Shared::LD;
        const int tid = threadIdx.x;
        const int length = this->length(query);
        const auto* kv = reinterpret_cast<const __nv_bfloat16*>(params.kv);
        const int64_t cache_policy = legacy::sm90::createpolicy_evict_last();
        if (tid < BK) {
            int index = start + tid < length
                ? params.indices[int64_t(query) * params.stride_indices_s_q + start + tid] : -1;
            smem.indices[tid] = index >= 0 && index < params.s_kv ? index : -1;
        }
        __syncthreads();
        if (params.stride_kv_s_kv % 8 == 0 && (reinterpret_cast<uintptr_t>(kv) & 15) == 0) {
            for (int i = tid; i < BK * (D_QK / 8); i += THREADS) {
                int row = i / (D_QK / 8), col = (i % (D_QK / 8)) * 8;
                int index = smem.indices[row];
                const auto* src = kv + int64_t(max(index, 0)) * params.stride_kv_s_kv + col;
                legacy::sm90::cp_async_cacheglobal_l2_prefetch_256B(
                    src, smem.buffer.kv + row * LD + col, index >= 0, cache_policy);
            }
            asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
        } else {
            for (int i = tid; i < BK * D_QK; i += THREADS) {
                int index = smem.indices[i / D_QK];
                smem.buffer.kv[(i / D_QK) * LD + i % D_QK] = index >= 0
                    ? kv[int64_t(index) * params.stride_kv_s_kv + i % D_QK] : __float2bfloat16(0.f);
            }
        }
    }

    template<int D_QK, int BH, typename Shared>
    __device__ void store(Shared& smem, float (&acc)[16][4], int query, int head_base) const {
        constexpr int THREADS = BH * 8;
        const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
        const int group = warp / 4, kw = warp % 4, grp = lane / 4, tig = lane % 4;
        if (tid < BH) {
            int head = head_base + tid;
            float m = smem.maximum[tid], l = smem.denominator[tid];
            float sink = params.attn_sink && head < params.h_q ? params.attn_sink[head] : -CUDART_INF_F;
            smem.alpha[tid] = l == 0.f ? 0.f : 1.f / (l + exp2f(sink * CUDART_L2E_F - m));
            if (head < params.h_q) {
                params.max_logits[int64_t(query) * params.h_q + head] = l == 0.f ? -CUDART_INF_F : m * CUDART_LN2_F;
                params.lse[int64_t(query) * params.h_q + head] = l == 0.f ? CUDART_INF_F : logf(l) + m * CUDART_LN2_F;
            }
        }
        __syncthreads();
        const float f_lo = smem.alpha[group * 16 + grp];
        const float f_hi = smem.alpha[group * 16 + grp + 8];
        #pragma unroll
        for (int j = 0; j < PV_FRAGS; ++j) {
            int c = kw * 128 + j * 8 + 2 * tig;
            *reinterpret_cast<__nv_bfloat162*>(smem.buffer.out + (group * 16 + grp) * 512 + c) =
                __floats2bfloat162_rn(acc[j][0] * f_lo, acc[j][1] * f_lo);
            *reinterpret_cast<__nv_bfloat162*>(smem.buffer.out + (group * 16 + grp + 8) * 512 + c) =
                __floats2bfloat162_rn(acc[j][2] * f_hi, acc[j][3] * f_hi);
        }
        __syncthreads();
        // `out` may be a head slice of a padded buffer, so index through the
        // output strides; the vector path additionally needs 16B-aligned rows.
        const bool vec_out = (reinterpret_cast<uintptr_t>(params.out) & 15) == 0
            && params.stride_out_s_q % 8 == 0 && params.stride_out_h_q % 8 == 0;
        const int64_t out_row = int64_t(query) * params.stride_out_s_q;
        if (vec_out) {
            for (int i = tid; i < BH * 64; i += THREADS) {
                int head = head_base + i / 64;
                if (head < params.h_q) {
                    uint4 packed;
                    auto* pairs = reinterpret_cast<__nv_bfloat162*>(&packed);
                    const __nv_bfloat16* src = smem.buffer.out + (i / 64) * 512 + (i % 64) * 8;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) pairs[j] = reinterpret_cast<const __nv_bfloat162*>(src)[j];
                    int64_t dst = out_row + int64_t(head) * params.stride_out_h_q + (i % 64) * 8;
                    reinterpret_cast<uint4*>(reinterpret_cast<__nv_bfloat16*>(params.out) + dst)[0] = packed;
                }
            }
        } else {
            auto* out = reinterpret_cast<__nv_bfloat16*>(params.out);
            for (int i = tid; i < BH * 512; i += THREADS) {
                int head = head_base + i / 512;
                if (head < params.h_q)
                    out[out_row + int64_t(head) * params.stride_out_h_q + i % 512] = smem.buffer.out[i];
            }
        }
    }
};

template<int D_QK, bool HAVE_TOPK_LENGTH, int BH>
void launch_small_heads(const SparseAttnFwdParams& params) {
    KU_ASSERT(params.d_v == 512 && params.h_kv == 1);
    if (params.s_q == 0) return;
    constexpr int smem_size = sizeof(small_heads::SmallHeadsShared<D_QK, BH>);
    auto kernel = small_heads::attention_kernel<D_QK, BH, SmallHeadsPrefillIO<HAVE_TOPK_LENGTH>>;
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<params.s_q * ((params.h_q + BH - 1) / BH), BH * 8, smem_size, params.stream>>>({params});
    KU_CHECK_KERNEL_LAUNCH();
}

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_small_heads(const SparseAttnFwdParams& params) {
    // One CTA per query row, so the 32-key KV tile is loaded once: BH=32 covers
    // h_q == 32 outright, two BH=16 CTAs would each re-read it.
    if (params.h_q > 16) {
        launch_small_heads<D_QK, HAVE_TOPK_LENGTH, 32>(params);
    } else {
        launch_small_heads<D_QK, HAVE_TOPK_LENGTH, 16>(params);
    }
}

}
