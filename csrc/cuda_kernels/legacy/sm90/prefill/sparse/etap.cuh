#pragma once

#include "config.h"

#include "cuda_kernels/legacy/utils.h"
#include "../../helpers.h"

namespace legacy::sm90::prefill::sparse_fwd {

using namespace cute;

// ETAP (transposed attention) for h_q = 8 / 16 / 32 and D_QK = 512 / 576. Keys
// sit on the WGMMA M dim and heads on the N dim, so small head counts need no
// padding:
//   S^T [64 keys x H] = K Q^T,    O^T [512 x H] = V^T P^T
// V is the first 512 dims of every KV row, so for D_QK = 576 the trailing RoPE
// tile only enters the QK GEMM.
// A persistent CTA walks over queries. WG2 gathers Q and the 64-key KV blocks;
// block b of a query goes to buffer b%2, which belongs to math warpgroup b%2.
// Each math warpgroup runs its own online softmax over its blocks into a full
// O^T, so the two warpgroups do not wait for each other inside the loop; their
// partial results are merged in the epilogue. Softmax statistics are per head,
// i.e. per column of S^T: the column max is reduced across the 4 warps of a
// warpgroup for every block, while the column sums are kept per thread and
// reduced only in the epilogue.
template<int D_QK_, int H, bool HAVE_TOPK_LENGTH>
struct EtapKernel {

static_assert(H == 8 || H == 16 || H == 32);
static_assert(D_QK_ == 512 || D_QK_ == 576);
static constexpr int D_QK = D_QK_;
static constexpr int D_V = 512;
static constexpr int NUM_KTILES = D_QK/64;         // 64-element tiles along d_qk (8 or 9)
static constexpr int HALF_KTILES = NUM_KTILES/2;   // Tiles in the first half of a KV buffer; the
                                                   // second half also holds the trailing RoPE tile.
static constexpr int B_TOPK = 64;
static constexpr int NUM_THREADS = 128*3;
static constexpr float MAX_INIT_VAL = -1e30;
static constexpr int NUM_COLS = H/4;      // Head columns held by each thread of a 64xH accumulator
static constexpr int O_LD = D_V + 8;      // Padded row stride of the output staging tile

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQ = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<H>, Int<D_QK>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));
using SmemLayoutK = SmemLayoutKTiles<D_QK/64>;
// V^T over half of the value dims, i.e. an MN-major view of 4 K tiles
using SmemLayoutHalfVt = decltype(composition(
    SmemLayoutKTiles<D_V/64/2>{},
    Layout<Shape<Int<D_V/2>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
));
// P^T is the K-major B operand of the PV GEMM: [head][key]
using SmemLayoutP = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<H>, Int<B_TOPK>>{}
), Shape<_1, _1>{}));
using SmemLayoutPTransposed = decltype(composition(
    SmemLayoutP{},
    Layout<Shape<Int<B_TOPK>, Int<H>>, Stride<Int<H>, _1>>{}
));
// O [head][dim] with a padded row stride, viewed as (dim, head)
using SmemLayoutOTransposed = Layout<Shape<Int<D_V/2>, Int<H>>, Stride<_1, Int<O_LD>>>;

// fp32 elements of half of a warpgroup's O^T accumulator
static constexpr int HALF_O = D_V/2*H/128;

struct SharedMemoryPlan {
    array_aligned<bf16, cosize_v<SmemLayoutQ>, 1024> q;
    array_aligned<bf16, cosize_v<SmemLayoutK>, 1024> k[2];
    array_aligned<bf16, cosize_v<SmemLayoutP>, 1024> p[2];
    union {
        float exchange[HALF_O*128];     // Half of a partial O^T, indexed [elem][thread]
        bf16 o[H*O_LD];
    } epilogue;
    float col_max[2][4][H];             // Per-warp column max of the current block, [WG][warp][head]
    float col_sum[2][4][H], final_max[2][H];
    bool is_kv_valid[2][B_TOPK];
    // [buffer][half]: half 0 / 1 holds value dims [0, 256) / [256, 512)
    transac_bar_t bar_q_ready, bar_q_free, bar_k_ready[2][2], bar_k_free[2][2], bar_is_kv_valid_ready[2];
};
static_assert(sizeof(SharedMemoryPlan::k[0]) == B_TOPK*D_QK*sizeof(bf16));    // The producer addresses k[1] from k[0]

// Tiles of the first ([0, HALF_KTILES)) and second ([HALF_KTILES, NUM_KTILES)) half of a
// KV buffer. The math warpgroup releases each half after the matching half of the PV,
// so the producer can refill it while the other half is still in use.
static constexpr int k_half_begin(int half) { return half ? HALF_KTILES : 0; }
static constexpr int k_half_end(int half) { return half ? NUM_KTILES : HALF_KTILES; }

using TiledMMA_QK = decltype(make_tiled_mma(
    SM90::GMMA::ss_op_selector<bf16, bf16, float, Shape<_64, Int<H>, _16>, GMMA::Major::K, GMMA::Major::K>(),
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMMA_PV = decltype(make_tiled_mma(
    SM90::GMMA::ss_op_selector<bf16, bf16, float, Shape<_64, Int<H>, _16>, GMMA::Major::MN, GMMA::Major::K>(),
    Layout<Shape<_1, _1, _1>>{}
));

// Transposing stmatrix for a 64xH accumulator; a warp holds 16 x H of it.
using STSM_T = std::conditional_t<H == 8, SM90_U16x4_STSM_T, SM90_U16x8_STSM_T>;

enum NamedBarriers : uint32_t {
    wg0_sync = 0,
    wg1_sync = 1,
    epilogue_sync = 2
};

// Head column of the s-th column slot of this thread in a 64xH accumulator
static __device__ __forceinline__ int get_col(int s, int idx_in_warpgroup) {
    return 8*(s/2) + (idx_in_warpgroup%4)*2 + (s&1);
}

// Column slot of the i-th element of this thread in an mxH accumulator
static __device__ __forceinline__ constexpr int get_col_slot(int i) {
    return (i % (H/2))/4*2 + (i&1);
}

static __device__ __forceinline__ int get_topk_length(const SparseAttnFwdParams &params, int s_q_idx) {
    int topk_length = params.topk;
    if constexpr (HAVE_TOPK_LENGTH) topk_length = min(topk_length, max(0, __ldg(params.topk_length + s_q_idx)));
    return topk_length;
}

// Issued by the producer warpgroup; arrives on `bar` once Q is in shared memory
static __device__ __forceinline__ void load_q(const SparseAttnFwdParams &params, bf16* sq, int s_q_idx, int tid, transac_bar_t &bar) {
    Tensor sQ = make_tensor(make_smem_ptr(sq), SmemLayoutQ{});
    const bf16* q = params.q + (int64_t)s_q_idx*params.stride_q_s_q;
    if (params.stride_q_s_q % 8 == 0 && params.stride_q_h_q % 8 == 0 && (reinterpret_cast<uintptr_t>(params.q) & 15) == 0) {
        CUTE_UNROLL
        for (int i = tid; i < H*(D_QK/8); i += 128) {
            int row = i / (D_QK/8), col = (i % (D_QK/8))*8;
            uint32_t dst = cute::cast_smem_ptr_to_uint(&sQ(row, col));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(q + row*params.stride_q_h_q + col));
        }
        cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&bar));
    } else {
        for (int i = tid; i < H*D_QK; i += 128) {
            sQ(i / D_QK, i % D_QK) = q[(i / D_QK)*params.stride_q_h_q + i % D_QK];
        }
        fence_view_async_shared();
        bar.arrive();
    }
}

static __device__ __forceinline__ void devfunc(const SparseAttnFwdParams &params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int idx_in_warpgroup = threadIdx.x % 128;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);

    if (warp_idx == 0 && elect_one_sync()) {
        CUTE_UNROLL
        for (int buf = 0; buf < 2; ++buf) {
            CUTE_UNROLL
            for (int half = 0; half < 2; ++half) {
                plan.bar_k_ready[buf][half].init(128);
                plan.bar_k_free[buf][half].init(128);
            }
            plan.bar_is_kv_valid_ready[buf].init(16);
        }
        plan.bar_q_ready.init(128);
        plan.bar_q_free.init(256);
        fence_barrier_init();
    }
    __syncthreads();

    if (warpgroup_idx == 0 || warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_alloc<216>();
        const int lane_idx = threadIdx.x % 32;
        const int warp_in_wg = idx_in_warpgroup / 32;
        const int row0 = get_AorC_row_idx(0, idx_in_warpgroup);
        const int row1 = get_AorC_row_idx(1, idx_in_warpgroup);
        const uint32_t wg_sync = warpgroup_idx ? NamedBarriers::wg1_sync : NamedBarriers::wg0_sync;
        Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(plan.k[warpgroup_idx].data()), SmemLayoutK{});
        Tensor sP = make_tensor(make_smem_ptr(plan.p[warpgroup_idx].data()), SmemLayoutP{});
        Tensor sPt = make_tensor(make_smem_ptr(plan.p[warpgroup_idx].data()), SmemLayoutPTransposed{});
        Tensor rO = partition_fragment_C(TiledMMA_PV{}, Shape<Int<D_V>, Int<H>>{});
        Tensor rS = partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_TOPK>, Int<H>>{});
        Tensor rP = make_fragment_like<bf16>(rS);

        int k_phase = 0, q_phase = 0;
        CUTE_NO_UNROLL
        for (int s_q_idx = blockIdx.x; s_q_idx < params.s_q; s_q_idx += gridDim.x) {
            const int num_blocks = ku::ceil_div(get_topk_length(params, s_q_idx), (int)B_TOPK);
            float rM[NUM_COLS], rL[NUM_COLS];
            CUTE_UNROLL
            for (int s = 0; s < NUM_COLS; ++s) {
                rM[s] = MAX_INIT_VAL;
                rL[s] = 0.0f;
            }
            cute::fill(rO, 0.0f);

            plan.bar_q_ready.wait(q_phase);
            q_phase ^= 1;
            if (warpgroup_idx >= num_blocks)
                plan.bar_q_free.arrive();

            CUTE_NO_UNROLL
            for (int block_idx = warpgroup_idx; block_idx < num_blocks; block_idx += 2) {
                // S^T = K Q^T
                CUTE_UNROLL
                for (int half = 0; half < 2; ++half) {
                    plan.bar_k_ready[warpgroup_idx][half].wait(k_phase);
                    CUTE_UNROLL
                    for (int tile_idx = k_half_begin(half); tile_idx < k_half_end(half); ++tile_idx) {
                        Tensor sK_tile = flat_divide(sK, Tile<Int<B_TOPK>, _64>{})(_, _, _0{}, tile_idx);
                        Tensor sQ_tile = flat_divide(sQ, Tile<Int<H>, _64>{})(_, _, _0{}, tile_idx);
                        gemm_ss(tile_idx == 0, TiledMMA_QK{}, sK_tile, sQ_tile, rS, idx_in_warpgroup);
                    }
                }
                warpgroup_commit_batch();
                warpgroup_wait<0>();
                if (block_idx + 2 >= num_blocks)
                    plan.bar_q_free.arrive();

                // Mask invalid keys (rows), scale, and reduce the column max over the 4 warps
                plan.bar_is_kv_valid_ready[warpgroup_idx].wait(k_phase);
                const bool is_valid0 = plan.is_kv_valid[warpgroup_idx][row0];
                const bool is_valid1 = plan.is_kv_valid[warpgroup_idx][row1];
                const float scale = params.sm_scale_div_log2;
                CUTE_UNROLL
                for (int s = 0; s < NUM_COLS; ++s) {
                    int i = (s/2)*4 + (s&1);
                    rS(i) = is_valid0 ? rS(i)*scale : -INFINITY;
                    rS(i+2) = is_valid1 ? rS(i+2)*scale : -INFINITY;
                    float cur_max = max(rS(i), rS(i+2));
                    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 4));
                    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 8));
                    cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 16));
                    if (lane_idx < 4)
                        plan.col_max[warpgroup_idx][warp_in_wg][get_col(s, idx_in_warpgroup)] = cur_max;
                }
                NamedBarrier::arrive_and_wait(128, wg_sync);

                float scale_for_o[NUM_COLS];
                CUTE_UNROLL
                for (int s = 0; s < NUM_COLS; ++s) {
                    const int col = get_col(s, idx_in_warpgroup);
                    float new_max = rM[s];
                    CUTE_UNROLL
                    for (int w = 0; w < 4; ++w)
                        new_max = max(new_max, plan.col_max[warpgroup_idx][w][col]);
                    scale_for_o[s] = exp2f(rM[s] - new_max);
                    rM[s] = new_max;

                    int i = (s/2)*4 + (s&1);
                    float p0 = exp2f(rS(i) - new_max), p1 = exp2f(rS(i+2) - new_max);
                    rL[s] = rL[s]*scale_for_o[s] + (p0 + p1);
                    rP(i) = (bf16)p0;
                    rP(i+2) = (bf16)p1;
                }

                // Save P^T (the previous PV has finished reading it), rescale O^T
                {
                    auto r2s_copy = make_tiled_copy_C(Copy_Atom<STSM_T, bf16>{}, TiledMMA_QK{});
                    ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
                    cute::copy(r2s_copy, thr_copy.retile_S(rP), thr_copy.partition_D(sPt));
                }
                fence_view_async_shared();
                CUTE_UNROLL
                for (int i = 0; i < size(rO); ++i)
                    rO(i) *= scale_for_o[get_col_slot(i)];
                // Also guarantees that all warps have read col_max before it is rewritten
                NamedBarrier::arrive_and_wait(128, wg_sync);

                // O^T += V^T P^T, one half of the value dims at a time so that the
                // first half of the buffer is released early
                CUTE_UNROLL
                for (int half = 0; half < 2; ++half) {
                    Tensor sVt = make_tensor(make_smem_ptr(plan.k[warpgroup_idx].data() + half*(B_TOPK*D_V/2)), SmemLayoutHalfVt{});
                    Tensor rO_half = make_tensor(rO.data() + half*HALF_O, partition_fragment_C(TiledMMA_PV{}, Shape<Int<D_V/2>, Int<H>>{}).layout());
                    gemm_ss(false, TiledMMA_PV{}, sVt, sP, rO_half, idx_in_warpgroup);
                    warpgroup_commit_batch();
                }
                warpgroup_wait<1>();
                plan.bar_k_free[warpgroup_idx][0].arrive();
                warpgroup_wait<0>();
                plan.bar_k_free[warpgroup_idx][1].arrive();

                k_phase ^= 1;
            }

            // Reduce L over the threads of this warpgroup holding the same head column
            CUTE_UNROLL
            for (int s = 0; s < NUM_COLS; ++s) {
                rL[s] += __shfl_xor_sync(0xffffffff, rL[s], 4);
                rL[s] += __shfl_xor_sync(0xffffffff, rL[s], 8);
                rL[s] += __shfl_xor_sync(0xffffffff, rL[s], 16);
                if (lane_idx < 4) {
                    plan.col_sum[warpgroup_idx][warp_in_wg][get_col(s, idx_in_warpgroup)] = rL[s];
                    if (warp_in_wg == 0)
                        plan.final_max[warpgroup_idx][get_col(s, idx_in_warpgroup)] = rM[s];
                }
            }
            NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_sync);

            // Merge the two partial softmax states; own_factors scales this warpgroup's O^T.
            // rO is only indexed with compile-time offsets below, otherwise it would be
            // demoted to local memory.
            float own_factors[NUM_COLS];
            CUTE_UNROLL
            for (int s = 0; s < NUM_COLS; ++s) {
                const int head = get_col(s, idx_in_warpgroup);
                const float m0 = plan.final_max[0][head], m1 = plan.final_max[1][head];
                const float m = max(m0, m1);
                float l0 = 0.0f, l1 = 0.0f;
                CUTE_UNROLL
                for (int w = 0; w < 4; ++w) {
                    l0 += plan.col_sum[0][w][head];
                    l1 += plan.col_sum[1][w][head];
                }
                const float scale0 = exp2f(m0 - m), scale1 = exp2f(m1 - m);
                const float l = l0*scale0 + l1*scale1;
                float attn_sink = params.attn_sink == nullptr ? -CUDART_INF_F : params.attn_sink[head]*CUDART_L2E_F;
                float scale_factor = l == 0.0f ? 0.0f : 1.0f / (l + exp2f(attn_sink - m));    // The output should be 0 whatever attn_sink is
                own_factors[s] = (warpgroup_idx == 0 ? scale0 : scale1)*scale_factor;
                if (threadIdx.x < 4) {
                    int64_t g_offset = (int64_t)s_q_idx*params.h_q + head;
                    params.max_logits[g_offset] = l == 0.0f ? -CUDART_INF_F : m*CUDART_LN2_F;
                    params.lse[g_offset] = l == 0.0f ? CUDART_INF_F : logf(l) + m*CUDART_LN2_F;
                }
            }

            // Warpgroup w keeps the value dims [w*256, w*256+256). The other half of
            // each partial O^T is handed over through the exchange buffer, in
            // accumulator order so that the thread with the same index reads it: WG1
            // writes first, then each WG0 thread reads its slots and refills them.
            float* exchange = plan.epilogue.exchange;
            auto rOb = make_fragment_like<bf16>(partition_fragment_C(TiledMMA_PV{}, Shape<Int<D_V/2>, Int<H>>{}));
            auto put_peer_half = [&](auto wg) {
                constexpr int WG = decltype(wg)::value;
                CUTE_UNROLL
                for (int i = 0; i < HALF_O; i += 2) {
                    *reinterpret_cast<float2*>(&exchange[(i/2*128 + idx_in_warpgroup)*2]) =
                        make_float2(rO((WG^1)*HALF_O + i)*own_factors[get_col_slot(i)], rO((WG^1)*HALF_O + i + 1)*own_factors[get_col_slot(i+1)]);
                }
            };
            auto merge_own_half = [&](auto wg) {
                constexpr int WG = decltype(wg)::value;
                CUTE_UNROLL
                for (int i = 0; i < HALF_O; i += 2) {
                    float2 peer = *reinterpret_cast<const float2*>(&exchange[(i/2*128 + idx_in_warpgroup)*2]);
                    rOb(i) = (bf16)(rO(WG*HALF_O + i)*own_factors[get_col_slot(i)] + peer.x);
                    rOb(i+1) = (bf16)(rO(WG*HALF_O + i + 1)*own_factors[get_col_slot(i+1)] + peer.y);
                }
            };
            if (warpgroup_idx == 1)
                put_peer_half(Int<1>{});
            NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_sync);
            if (warpgroup_idx == 0) {
                merge_own_half(Int<0>{});
                put_peer_half(Int<0>{});
            }
            NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_sync);
            if (warpgroup_idx == 1)
                merge_own_half(Int<1>{});
            NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_sync);

            // O^T -> sO [head][dim], reusing the exchange buffer
            {
                Tensor sOt = make_tensor(make_smem_ptr(plan.epilogue.o + warpgroup_idx*(D_V/2)), SmemLayoutOTransposed{});
                auto r2s_copy = make_tiled_copy_C(Copy_Atom<STSM_T, bf16>{}, TiledMMA_PV{});
                ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
                cute::copy(r2s_copy, thr_copy.retile_S(rOb), thr_copy.partition_D(sOt));
            }
            NamedBarrier::arrive_and_wait(256, NamedBarriers::epilogue_sync);

            // `out` may be a head slice of a padded buffer, so index through the
            // output strides; the vector path additionally needs 16B-aligned rows.
            // The next query's epilogue only rewrites the buffer after its first
            // epilogue barrier, which both warpgroups reach after this store.
            const bf16* sO = plan.epilogue.o;
            const int64_t out_row = (int64_t)s_q_idx*params.stride_out_s_q;
            if ((reinterpret_cast<uintptr_t>(params.out) & 15) == 0 && params.stride_out_s_q % 8 == 0 && params.stride_out_h_q % 8 == 0) {
                for (int i = threadIdx.x; i < H*(D_V/8); i += 256) {
                    int head = i / (D_V/8), col = (i % (D_V/8))*8;
                    *reinterpret_cast<uint4*>(params.out + out_row + (int64_t)head*params.stride_out_h_q + col) =
                        *reinterpret_cast<const uint4*>(sO + head*O_LD + col);
                }
            } else {
                for (int i = threadIdx.x; i < H*D_V; i += 256) {
                    int head = i / D_V, col = i % D_V;
                    params.out[out_row + (int64_t)head*params.stride_out_h_q + col] = sO[head*O_LD + col];
                }
            }
        }
    } else {
        // Producer warpgroup
        cutlass::arch::warpgroup_reg_dealloc<72>();

        constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128/GROUP_SIZE;
        constexpr int NUM_ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
        int idx_in_group = idx_in_warpgroup % GROUP_SIZE;
        int group_idx = idx_in_warpgroup / GROUP_SIZE;

        bf16* my_sKV_base = &(make_tensor(make_smem_ptr(plan.k[0].data()), SmemLayoutKTiles<1>{})(group_idx, idx_in_group*8));
        const bf16* my_gKV_base = params.kv + idx_in_group*8;
        int64_t cache_policy = createpolicy_evict_last();
        // Rows that are not 16B aligned fall back to scalar loads
        const bool is_kv_aligned = params.stride_kv_s_kv % 8 == 0 && (reinterpret_cast<uintptr_t>(params.kv) & 15) == 0;

        int free_phases = 0b11;    // Bit b: the phase to wait on bar_k_free[b][*]
        int q_free_phase = 0;
        bool is_first_query = true;

        CUTE_NO_UNROLL
        for (int s_q_idx = blockIdx.x; s_q_idx < params.s_q; s_q_idx += gridDim.x) {
            const int topk_length = get_topk_length(params, s_q_idx);
            const int num_blocks = ku::ceil_div(topk_length, (int)B_TOPK);
            const int* gIndices = params.indices + (int64_t)s_q_idx*params.stride_indices_s_q;

            // Q is free once both math warpgroups have issued their last QK of the previous query
            if (!is_first_query) {
                plan.bar_q_free.wait(q_free_phase);
                q_free_phase ^= 1;
            }
            is_first_query = false;
            load_q(params, plan.q.data(), s_q_idx, idx_in_warpgroup, plan.bar_q_ready);

            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
                const int buf = block_idx & 1;
                int64_t token_offsets[NUM_ROWS_PER_GROUP];
                bool is_token_valid[NUM_ROWS_PER_GROUP];
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                    int offs = block_idx*B_TOPK + local_row*NUM_GROUPS + group_idx;
                    int t = offs < topk_length ? __ldg(gIndices + offs) : -1;
                    is_token_valid[local_row] = t >= 0 && t < params.s_kv;
                    token_offsets[local_row] = is_token_valid[local_row] ? t*(int64_t)params.stride_kv_s_kv : 0;
                }

                CUTE_UNROLL
                for (int half = 0; half < 2; ++half) {
                    plan.bar_k_free[buf][half].wait((free_phases >> buf) & 1);
                    if (is_kv_aligned) {
                        CUTE_UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                            CUTE_UNROLL
                            for (int tile_idx = k_half_begin(half); tile_idx < k_half_end(half); ++tile_idx) {
                                cp_async_cacheglobal_l2_prefetch_256B(
                                    my_gKV_base + token_offsets[local_row] + tile_idx*64,
                                    my_sKV_base + (buf*B_TOPK*D_QK + tile_idx*(B_TOPK*64) + local_row*NUM_GROUPS*64),
                                    is_token_valid[local_row],
                                    cache_policy
                                );
                            }
                        }
                        cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&plan.bar_k_ready[buf][half]));
                    } else {
                        CUTE_UNROLL
                        for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                            CUTE_UNROLL
                            for (int tile_idx = k_half_begin(half); tile_idx < k_half_end(half); ++tile_idx) {
                                bf16 chunk[8];
                                CUTE_UNROLL
                                for (int i = 0; i < 8; ++i)
                                    chunk[i] = is_token_valid[local_row] ? my_gKV_base[token_offsets[local_row] + tile_idx*64 + i] : bf16(0.0f);
                                *reinterpret_cast<uint4*>(my_sKV_base + (buf*B_TOPK*D_QK + tile_idx*(B_TOPK*64) + local_row*NUM_GROUPS*64)) =
                                    *reinterpret_cast<const uint4*>(chunk);
                            }
                        }
                        fence_view_async_shared();
                        plan.bar_k_ready[buf][half].arrive();
                    }
                }
                free_phases ^= 1 << buf;

                // Valid mask
                // NOTE: The buffer being free implies that the masking of its previous block has finished
                if (idx_in_group == 0) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row)
                        plan.is_kv_valid[buf][local_row*NUM_GROUPS + group_idx] = is_token_valid[local_row];
                    plan.bar_is_kv_valid_ready[buf].arrive();
                }
            }
        }
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif
}

};

template<typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1, 1)
sparse_attn_fwd_etap_kernel(__grid_constant__ const SparseAttnFwdParams params) {
    Kernel::devfunc(params);
}

template<int D_QK, int H, bool HAVE_TOPK_LENGTH>
void launch_etap(const SparseAttnFwdParams& params) {
    using Kernel = EtapKernel<D_QK, H, HAVE_TOPK_LENGTH>;
    auto kernel = &sparse_attn_fwd_etap_kernel<Kernel>;
    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlan);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    // Persistent: one CTA per SM walks over the queries
    kernel<<<std::min(params.s_q, params.num_sm), Kernel::NUM_THREADS, smem_size, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
}

template<int D_QK, bool HAVE_TOPK_LENGTH>
void run_etap(const SparseAttnFwdParams& params) {
    KU_ASSERT(params.d_qk == D_QK && params.d_v == 512 && params.h_kv == 1);
    if (params.s_q == 0) return;
    if (params.h_q == 8) {
        launch_etap<D_QK, 8, HAVE_TOPK_LENGTH>(params);
    } else if (params.h_q == 16) {
        launch_etap<D_QK, 16, HAVE_TOPK_LENGTH>(params);
    } else {
        KU_ASSERT(params.h_q == 32);
        launch_etap<D_QK, 32, HAVE_TOPK_LENGTH>(params);
    }
}

}
