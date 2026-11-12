#pragma once

#include <cuda_bf16.h>

#include "params.h"
#include "helpers.h"
#include "cuda_kernels/legacy/utils.h"

namespace legacy::sm90::small_heads {

// A BH-head CTA over a 32-key tile. Driving both matmuls with ldmatrix +
// mma.sync avoids the scattered smem operand loads that wmma expands into.
// Each head group uses four QK warps and IO::WARPS PV warps, with the value
// dimensions distributed across those PV warps as m16n8 accumulators.

template<int D_QK, int BH>
struct SmallHeadsShared {
    static_assert(D_QK == 512 || D_QK == 576);
    static_assert(BH == 16 || BH == 32);
    static constexpr int LD = D_QK + 8;
    static constexpr int BK = 32;
    static constexpr int LP = BK + 8;
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

// Load one query tile, zeroing inactive heads. The caller synchronizes before
// the first QK matmul; scalar loads also support unaligned tensor views.
template<int D_QK, int BH, int THREADS, typename Params, typename Shared>
__device__ __forceinline__ void load_query_tile(
    const Params& params, Shared& smem, int64_t q_row, int head_base) {
    constexpr int LD = Shared::LD;
    const int tid = threadIdx.x;
    const auto* q = reinterpret_cast<const __nv_bfloat16*>(params.q);
    if (q_row % 8 == 0 && params.stride_q_h_q % 8 == 0 &&
        (reinterpret_cast<uintptr_t>(q) & 15) == 0) {
        for (int i = tid; i < BH * (D_QK / 8); i += THREADS) {
            int row = i / (D_QK / 8), col = (i % (D_QK / 8)) * 8;
            int head = head_base + row;
            const auto* src = q + q_row + min(head, params.h_q - 1) * params.stride_q_h_q + col;
            unsigned dst = cute::cast_smem_ptr_to_uint(smem.q + row * LD + col);
            int bytes = head < params.h_q ? 16 : 0;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes));
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    } else {
        for (int i = tid; i < BH * D_QK; i += THREADS) {
            int row = i / D_QK, head = head_base + row;
            smem.q[row * LD + i % D_QK] = head < params.h_q
                ? q[q_row + head * params.stride_q_h_q + i % D_QK]
                : __float2bfloat16(0.f);
        }
    }
}

// Eight lanes own one score row. Update the online maximum/denominator and
// materialize BF16 probabilities for PV. The caller synchronizes before PV.
template<int BH, typename Shared>
__device__ __forceinline__ void update_online_softmax(Shared& smem, float scale_log2) {
    constexpr int BK = Shared::BK, LP = Shared::LP;
    const int tid = threadIdx.x;
    if (tid < BH * 8) {
        const int row = tid / 8, slane = tid % 8;
        float values[BK / 8];
        const float old_max = smem.maximum[row];
        float maximum = old_max;
        #pragma unroll
        for (int j = 0; j < BK / 8; ++j) {
            const int col = slane + 8 * j;
            values[j] = smem.indices[col] >= 0
                ? smem.scores[row * LP + col] * scale_log2 : -CUDART_INF_F;
            maximum = fmaxf(maximum, values[j]);
        }
        #pragma unroll
        for (int offset = 4; offset; offset /= 2)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset, 8));
        float alpha = exp2f(old_max - maximum);
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
        // Finish every lane's old_max read before lane 0 overwrites it.
        __syncwarp();
        if (slane == 0) {
            smem.alpha[row] = alpha;
            smem.maximum[row] = maximum;
            smem.denominator[row] = smem.denominator[row] * alpha + sum;
        }
    }
}

// IO is the prefill/decode adapter: it supplies params, WARPS/PV_FRAGS and
// Shared<D_QK, BH>, maps query rows with q_row(), partitions keys with
// length()/begin()/end(), stages a tile with load_kv(), and writes results with
// store(). The compute loop owns synchronization after load_kv() and before
// store(); adapters must finish any asynchronous copies before returning.
//
// Each head group uses four or eight PV warps. Retain the four-warp prefill
// register budget; eight-warp decode also needs shared memory for FP8 staging.
template<int D_QK, int BH, typename IO>
__global__ void __launch_bounds__(BH / 16 * IO::WARPS * 32, (IO::WARPS == 8 ? 2 : (BH == 16 ? (D_QK == 576 ? 3 : 4) : 2)))
attention_kernel(const IO io) {
    const auto& params = io.params;
    constexpr int BK = 32;
    using Shared = typename IO::template Shared<D_QK, BH>;
    constexpr int THREADS = BH / 16 * IO::WARPS * 32;
    constexpr int LD = Shared::LD, LP = Shared::LP;
    extern __shared__ __align__(32) unsigned char storage[];
    auto& smem = *reinterpret_cast<Shared*>(storage);
    const int tid = threadIdx.x, warp = tid / 32;
    const int group = warp / IO::WARPS, kw = warp % IO::WARPS;
    const int lane = tid & 31, grp = lane >> 2, tig = lane & 3;
    const int head_tiles = (params.h_q + BH - 1) / BH;
    const int query = blockIdx.x / head_tiles;
    const int head_base = (blockIdx.x % head_tiles) * BH;
    load_query_tile<D_QK, BH, THREADS>(params, smem, io.q_row(query), head_base);
    if (tid < BH) { smem.maximum[tid] = -1e30f; smem.denominator[tid] = 0.f; }
    __syncthreads();

    // Keep PV accumulators across key tiles.
    float acc[IO::PV_FRAGS][4];
    #pragma unroll
    for (int j = 0; j < IO::PV_FRAGS; ++j) { acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f; }
    __syncthreads();

    // ldmatrix.x4 A-tile address: thread lane -> sub-tile (m=lane/8) and row (lane%8).
    const int a_mat = lane >> 3, a_row = (lane & 7) + ((a_mat & 1) << 3);
    const int a_off = a_row * LD + ((a_mat & 2) << 2);
    const int p_off = a_row * LP + ((a_mat & 2) << 2);
    const __nv_bfloat16* q_tile = smem.q + group * 16 * LD + a_off;
    const __nv_bfloat16* kv_row = smem.buffer.kv + ((kw % 4) * 8 + grp) * LD;
    // Value tile for the PV: lanes 0..15 address keys 0..15 of one 8-dim slice.
    const __nv_bfloat16* pv_row = smem.buffer.kv + (lane & 15) * LD;

    // For each key tile: stage KV, compute QK, update softmax, accumulate PV.
    const int length = io.length(query);
    for (int start = io.begin(length); start < io.end(length); start += BK) {
        io.template load_kv<D_QK, BH>(smem, query, start);
        __syncthreads();
        // Four warps compute QK for each 16-head group.
        if (kw < 4) {
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            {
                const uint32_t b_off0 = cute::cast_smem_ptr_to_uint(kv_row + 2 * tig);
                const uint32_t b_off1 = cute::cast_smem_ptr_to_uint(kv_row + 2 * tig + 8);
                #pragma unroll 4
                for (int d = 0; d < D_QK; d += 16) {
                    uint32_t a0, a1, a2, a3, b0, b1;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                                 : "r"(cute::cast_smem_ptr_to_uint(q_tile + d)));
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
                float* s = smem.scores + (group * 16 + grp) * LP + (kw % 4) * 8 + 2 * tig;
                s[0] = d0; s[1] = d1;
                s[8 * LP] = d2; s[8 * LP + 1] = d3;
            }
        }
        __syncthreads();
        update_online_softmax<BH>(smem, params.sm_scale_div_log2);
        __syncthreads();
        // Rescale the previous PV accumulator before adding this tile.
        const float a_lo = smem.alpha[group * 16 + grp];
        const float a_hi = smem.alpha[group * 16 + grp + 8];
        #pragma unroll
        for (int j = 0; j < IO::PV_FRAGS; ++j) {
            acc[j][0] *= a_lo; acc[j][1] *= a_lo;
            acc[j][2] *= a_hi; acc[j][3] *= a_hi;
        }
        #pragma unroll
        for (int k = 0; k < BK; k += 16) {
            uint32_t a0, a1, a2, a3;
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                         : "r"(cute::cast_smem_ptr_to_uint(smem.prob + group * 16 * LP + k + p_off)));
            #pragma unroll
            for (int j = 0; j < IO::PV_FRAGS; ++j) {
                uint32_t b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                             : "=r"(b0), "=r"(b1)
                             : "r"(cute::cast_smem_ptr_to_uint(pv_row + k * LD + kw * (512 / IO::WARPS) + j * 8)));
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[j][0]), "+f"(acc[j][1]), "+f"(acc[j][2]), "+f"(acc[j][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
            }
        }
        __syncthreads();
    }
    io.template store<D_QK, BH>(smem, acc, query, head_base);
}

}
