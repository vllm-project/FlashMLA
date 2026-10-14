#pragma once

#include <cuda_bf16.h>

#include "params.h"
#include "cuda_kernels/legacy/utils.h"

namespace legacy::sm90::prefill::sparse_fwd {

// A BH-head CTA over a 32-key tile. Driving both matmuls with ldmatrix +
// mma.sync avoids the scattered smem operand loads that wmma expands into.
// Each warp owns 16 heads x 128 value dims as m16n8 accumulators, so the warps
// of a CTA form BH/16 head groups of four value-dim slices.

template<int D_QK, int BH>
struct SmallHeadsShared {
    static_assert(D_QK == 512 || D_QK == 576);
    static_assert(BH == 16 || BH == 32);
    static constexpr int LD = D_QK + 8;
    static constexpr int BK = 32;
    static constexpr int LP = BK + 8;
    // m16n8 accumulator tiles covering one warp's 128 value dims.
    static constexpr int PV_FRAGS = 16;
    __nv_bfloat16 q[BH * LD];
    union {
        __nv_bfloat16 kv[BK * LD];
        __nv_bfloat16 out[BH * 512];
    } buffer;
    float scores[BH * LP];
    __nv_bfloat16 prob[BH * LP];
    float maximum[BH], denominator[BH], alpha[BH];
    int indices[BK];
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

// 512 resident threads per SM: BH=16 is register bound (4 CTAs), BH=32 is
// shared-memory bound (2 CTAs).
template<int D_QK, bool HAVE_TOPK_LENGTH, int BH>
__global__ void __launch_bounds__(BH * 8, 512 / (BH * 8))
sparse_attn_fwd_small_heads_kernel(const SparseAttnFwdParams params) {
    constexpr int BK = 32;
    using Shared = SmallHeadsShared<D_QK, BH>;
    constexpr int THREADS = BH * 8;
    constexpr int LD = Shared::LD, LP = Shared::LP;
    extern __shared__ __align__(32) unsigned char storage[];
    auto& smem = *reinterpret_cast<Shared*>(storage);
    const int tid = threadIdx.x, warp = tid / 32;
    const int group = warp / 4, kw = warp % 4;
    const int lane = tid & 31, grp = lane >> 2, tig = lane & 3;
    const int head_tiles = (params.h_q + BH - 1) / BH;
    const int query = blockIdx.x / head_tiles;
    const int head_base = (blockIdx.x % head_tiles) * BH;
    const auto* q = reinterpret_cast<const __nv_bfloat16*>(params.q);
    const auto* kv = reinterpret_cast<const __nv_bfloat16*>(params.kv);

    if (params.stride_q_s_q % 8 == 0 && params.stride_q_h_q % 8 == 0 &&
        (reinterpret_cast<uintptr_t>(q) & 15) == 0) {
        for (int i = tid; i < BH * (D_QK / 8); i += THREADS) {
            int row = i / (D_QK / 8), col = (i % (D_QK / 8)) * 8;
            int head = head_base + row;
            const auto* src = q + int64_t(query) * params.stride_q_s_q + min(head, params.h_q - 1) * params.stride_q_h_q + col;
            unsigned dst = smem_u32(smem.q + row * LD + col);
            int bytes = head < params.h_q ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes));
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    } else {
        for (int i = tid; i < BH * D_QK; i += THREADS) {
            int row = i / D_QK, head = head_base + row;
            smem.q[row * LD + i % D_QK] = head < params.h_q
                ? q[int64_t(query) * params.stride_q_s_q + head * params.stride_q_h_q + i % D_QK]
                : __float2bfloat16(0.f);
        }
    }
    if (tid < BH) { smem.maximum[tid] = -1e30f; smem.denominator[tid] = 0.f; }
    __syncthreads();

    // The PV keeps 16 heads x 128 value dims per warp as m16n8 accumulators.
    float acc[Shared::PV_FRAGS][4];
    #pragma unroll
    for (int j = 0; j < Shared::PV_FRAGS; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
    __syncthreads();

    // ldmatrix.x4 A-tile address: thread lane -> sub-tile (m=lane/8) and row (lane%8).
    const int a_mat = lane >> 3, a_row = (lane & 7) + ((a_mat & 1) << 3);
    const int a_off = a_row * LD + ((a_mat & 2) << 2);
    const int p_off = a_row * LP + ((a_mat & 2) << 2);
    const __nv_bfloat16* q_tile = smem.q + group * 16 * LD + a_off;
    const __nv_bfloat16* kv_row = smem.buffer.kv + (kw * 8 + grp) * LD;
    // Value tile for the PV: lanes 0..15 address keys 0..15 of one 8-dim slice.
    const __nv_bfloat16* pv_row = smem.buffer.kv + (lane & 15) * LD;

    int length = params.topk;
    if constexpr (HAVE_TOPK_LENGTH) length = min(length, max(0, params.topk_length[query]));
    for (int start = 0; start < length; start += BK) {
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
                unsigned dst = smem_u32(smem.buffer.kv + row * LD + col);
                int bytes = index >= 0 ? 16 : 0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes));
            }
            asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
        } else {
            for (int i = tid; i < BK * D_QK; i += THREADS) {
                int index = smem.indices[i / D_QK];
                smem.buffer.kv[(i / D_QK) * LD + i % D_QK] = index >= 0
                    ? kv[int64_t(index) * params.stride_kv_s_kv + i % D_QK] : __float2bfloat16(0.f);
            }
        }
        __syncthreads();
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        {
            const uint32_t b_off0 = smem_u32(kv_row + 2 * tig);
            const uint32_t b_off1 = smem_u32(kv_row + 2 * tig + 8);
            #pragma unroll 4
            for (int d = 0; d < D_QK; d += 16) {
                uint32_t a0, a1, a2, a3, b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                             : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                             : "r"(smem_u32(q_tile + d)));
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(b0) : "r"(b_off0 + d * 2));
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(b1) : "r"(b_off1 + d * 2));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        {
            float* s = smem.scores + (group * 16 + grp) * LP + kw * 8 + 2 * tig;
            s[0] = d0; s[1] = d1;
            s[8 * LP] = d2; s[8 * LP + 1] = d3;
        }
        __syncthreads();
        const int row = tid / 8, slane = tid % 8;
        float values[BK / 8];
        float maximum = smem.maximum[row];
        #pragma unroll
        for (int j = 0; j < BK / 8; ++j) {
            const int col = slane + 8 * j;
            values[j] = smem.indices[col] >= 0
                ? smem.scores[row * LP + col] * params.sm_scale_div_log2 : -CUDART_INF_F;
            maximum = fmaxf(maximum, values[j]);
        }
        #pragma unroll
        for (int offset = 4; offset; offset /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset, 8));
        float alpha = exp2f(smem.maximum[row] - maximum);
        float sum = 0.f;
        #pragma unroll
        for (int j = 0; j < BK / 8; ++j) {
            float p = exp2f(values[j] - maximum);
            smem.prob[row * LP + slane + 8 * j] = __float2bfloat16(p);
            sum += p;
        }
        #pragma unroll
        for (int offset = 4; offset; offset /= 2)
            sum += __shfl_xor_sync(0xffffffff, sum, offset, 8);
        if (slane == 0) {
            smem.alpha[row] = alpha;
            smem.maximum[row] = maximum;
            smem.denominator[row] = smem.denominator[row] * alpha + sum;
        }
        __syncthreads();
        const float a_lo = smem.alpha[group * 16 + grp];
        const float a_hi = smem.alpha[group * 16 + grp + 8];
        #pragma unroll
        for (int j = 0; j < Shared::PV_FRAGS; ++j) {
            acc[j][0] *= a_lo; acc[j][1] *= a_lo;
            acc[j][2] *= a_hi; acc[j][3] *= a_hi;
        }
        #pragma unroll
        for (int k = 0; k < BK; k += 16) {
            uint32_t a0, a1, a2, a3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                         : "r"(smem_u32(smem.prob + group * 16 * LP + k + p_off)));
            #pragma unroll
            for (int j = 0; j < Shared::PV_FRAGS; ++j) {
                uint32_t b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                             : "=r"(b0), "=r"(b1)
                             : "r"(smem_u32(pv_row + k * LD + kw * 128 + j * 8)));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }
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
    for (int j = 0; j < Shared::PV_FRAGS; ++j) {
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


template<int D_QK, bool HAVE_TOPK_LENGTH, int BH>
void launch_small_heads(const SparseAttnFwdParams& params) {
    KU_ASSERT(params.d_v == 512 && params.h_kv == 1);
    if (params.s_q == 0) return;
    constexpr int smem_size = sizeof(SmallHeadsShared<D_QK, BH>);
    auto kernel = sparse_attn_fwd_small_heads_kernel<D_QK, HAVE_TOPK_LENGTH, BH>;
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<params.s_q * ((params.h_q + BH - 1) / BH), BH * 8, smem_size, params.stream>>>(params);
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
