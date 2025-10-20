// SM90-specific file - only compile for SM90+ architectures
#include "fwd.h"
#include <stdexcept>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && (__CUDA_ARCH__ < 1000) || !defined(__CUDA_ARCH__)

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cooperative_groups.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>

#include "utils.h"
#include "helpers.h"

namespace sm90 {

using namespace cute;

constexpr int D_Q = 576;
constexpr int D_K = 576;
constexpr int D_V = 512;

constexpr int B_H = 64;
constexpr int B_TOPK = 64;    // TopK block size
constexpr int NUM_THREADS = 128*3;
static constexpr float MAX_INIT_VAL = -1e30;    // We use this number as the initial value for mi (max logits)

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutOTiles = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    GMMA::Layout_SW128_Atom<bf16, GMMA::Major::K>{},
    Shape<Int<B_TOPK>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed = decltype(composition(
	SmemLayoutKTiles<NUM_TILES>{},
	Layout<Shape<Int<64*NUM_TILES>, Int<B_TOPK>>, Stride<Int<B_TOPK>, _1>>{}
));

using SmemLayoutQ = SmemLayoutQTiles<9>;
using SmemLayoutO = SmemLayoutOTiles<8>;
using SmemLayoutK = SmemLayoutKTiles<9>;
using SmemLayoutV = SmemLayoutKTilesTransposed<8>;
using SmemLayoutHalfV = SmemLayoutKTilesTransposed<4>;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_TOPK>>{}
), Shape<_1, _1>{}));

struct SharedMemoryPlan {
    union {
        array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
        array_aligned<bf16, cosize_v<SmemLayoutO>> o;
    } q_o;
    array_aligned<bf16, cosize_v<SmemLayoutK>> k[2];
    array_aligned<bf16, cosize_v<SmemLayoutS>> s;

    bool is_kv_valid[2][B_TOPK];
    float2 sM[32];
    float2 sL[64];   // For reduction across WG0/1 in epilogue
    float final_max_logits[64], final_lse[64];
    transac_bar_t bar_q, bar_k0_free[2], bar_k0_ready[2], bar_k1_free[2], bar_k1_ready[2], bar_is_kv_valid_ready;
};

using TiledMMA_QK = decltype(make_tiled_mma(
    GMMA::MMA_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{},
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMMA_PV_LocalP = decltype(make_tiled_mma(
    GMMA::MMA_64x256x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMMA_PV_RemoteP = decltype(make_tiled_mma(
    GMMA::MMA_64x256x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{}
));

template<
    typename Shape_Q, typename TMA_Q
>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    CUtensorMap tensor_map_O;
};

enum NamedBarriers : uint32_t {
    wg0_bunch_0_ready = 0,
    wg1_bunch_0_ready = 1,
    wg0_s0_ready = 2,
    wg1_s1_ready = 3,
    sL_ready = 4,
    warpgroup0_sync = 5,
    warpgroup1_sync = 6
};

// Save rPb (64x64, bfloat16) to sP using the stmatrix instruction
template<
    typename Tensor0,
    typename Tensor1
>
__forceinline__ __device__ void save_rS_to_sS(
    Tensor0 const &rPb,
    Tensor1 const &sP,
    int idx_in_warpgroup
) {
    auto r2s_copy = make_tiled_copy_C(
        Copy_Atom<SM90_U32x4_STSM_N, bf16>{},
        TiledMMA_QK{}
    );
    ThrCopy thr_copy = r2s_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_rPb = thr_copy.retile_S(rPb);
    Tensor thr_copy_sP = thr_copy.partition_D(sP);
    cute::copy(r2s_copy, thr_copy_rPb, thr_copy_sP);
}


template<typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
sparse_attn_fwd_kernel(__grid_constant__ const SparsePrefillParams params, __grid_constant__ const TmaParams tma_params) {
    // NOTE This kernel uses a similar schedule to Flash MLA - 0422. For a detailed explanation, please refer to https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250422-new-kernel-deep-dive.md
#if IS_SM90
    const int q_h_idx = blockIdx.x % (params.h_q/B_H);
    const int s_q_idx = blockIdx.x / (params.h_q/B_H);
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int warp_idx = cutlass::canonical_warp_idx_sync();
    const int idx_in_warpgroup = threadIdx.x % 128;

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.q_o.q.data()), SmemLayoutQ{});
    Tensor sO = make_tensor(make_smem_ptr(plan.q_o.o.data()), SmemLayoutO{});
    Tensor sS0 = make_tensor(make_smem_ptr(plan.k[0].data()+64*512), SmemLayoutS{});    // Overlap with sK0's RoPE part
    Tensor sS1 = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});

    if (warp_idx == 0 && elect_one_sync()) {
        // Prefetch TMA descriptors
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(&tma_params.tensor_map_O);

        // Initialize barriers
        plan.bar_q.init(1);
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
            plan.bar_k0_free[i].init(128);
            plan.bar_k0_ready[i].init(128);
            plan.bar_k1_free[i].init(128);
            plan.bar_k1_ready[i].init(128);
        }
        plan.bar_is_kv_valid_ready.init(16);
        fence_barrier_init();
    }

    __syncthreads();
    
    const int num_topk_blocks = params.topk / B_TOPK;
    if (warpgroup_idx == 0 || warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_alloc<216>();

        if (warp_idx == 0 && elect_one_sync()) {
            // Load Q
            Tensor gQ = flat_divide(
                tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx),
                Tile<Int<B_H>, Int<D_Q>>{}
            )(_, _, q_h_idx, _0{});
            launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
            plan.bar_q.arrive_and_expect_tx(B_H*D_Q*sizeof(bf16));
        }

        float rM[2] = {MAX_INIT_VAL, MAX_INIT_VAL}; // Meaning: the `max_logits` used for O / rL calculation
        float rL[2] = {0.0f, 0.0f};
        Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<B_H>, Int<D_V/2>>{});
        Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<B_H>, Int<B_TOPK>>{});
        Tensor rS = make_tensor<bf16>(partition_shape_A(TiledMMA_PV_LocalP{}, Shape<Int<B_H>, Int<B_TOPK>>{}));
        cute::fill(rO, 0.0f);
        
        // Wait for Q
        plan.bar_q.wait(0);

        bool cur_bar_wait_phase = 0;
        
        struct Warpgroup0 {};
        struct Warpgroup1 {};

        auto qkt_gemm_one_tile = [&](auto warpgroup_idx, int tile_idx, bool clear_accum) {
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            TiledMMA tiled_mma_QK = TiledMMA_QK{};
            Tensor sQ_tile = flat_divide(sQ, Tile<Int<B_H>, Int<64>>{})(_, _, _0{}, tile_idx);
            Tensor sK_tile = make_tensor(make_smem_ptr(plan.k[(int)IS_WG1].data() + tile_idx*B_TOPK*64), SmemLayoutKTiles<1>{});
            gemm_ss(clear_accum, tiled_mma_QK, sQ_tile, sK_tile, rP, idx_in_warpgroup);
        };

        auto mask_rP = [&](auto warpgroup_idx) {
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
            CUTE_UNROLL
            for (int row_idx = 0; row_idx < 2; ++row_idx) {
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rP); i += 4) {
                    int col = 8*(i/4) + (idx_in_warpgroup%4)*2;
                    if (!plan.is_kv_valid[IS_WG1][col]) rP(i) = -INFINITY;
                    if (!plan.is_kv_valid[IS_WG1][col+1]) rP(i+1) = -INFINITY;
                }
            }
        };

        auto online_softmax_and_rescale_o = [&](auto warpgroup_idx) {
            plan.bar_is_kv_valid_ready.wait(cur_bar_wait_phase);
            constexpr bool IS_WG1 = std::is_same_v<decltype(warpgroup_idx), Warpgroup1>;
            const float scale = params.sm_scale_div_log2;
            float r_sM[2];
            if constexpr (IS_WG1) {
                *(float2*)r_sM = plan.sM[idx_in_warpgroup/4];
            }
            float new_maxs[2];
            CUTE_UNROLL
            for (int row_idx = 0; row_idx < 2; ++row_idx) {
                // Get rowwise max
                float cur_max = -INFINITY;
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rP); i += 4) {
                    cur_max = max(cur_max, max(rP(i), rP(i+1)));
                }
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
                cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));
                cur_max *= scale;

                // Get new max and scale
                // For WG1, old_max comes from sM (written by WG0); for WG0, old_max comes from rM (read by WG0 from sM in the last round)
                new_maxs[row_idx] = max(IS_WG1 ? r_sM[row_idx] : rM[row_idx], cur_max);

                // Scale O
                float scale_for_o = exp2f(rM[row_idx]-new_maxs[row_idx]);
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rO); i += 4) {
                    rO(i) *= scale_for_o;
                    rO(i+1) *= scale_for_o;
                }

                // Get rS
                float cur_sum = 0;
                CUTE_UNROLL
                for (int i = row_idx*2; i < size(rP); i += 4) {
                    rP(i) = exp2f(rP(i)*scale - new_maxs[row_idx]);
                    rP(i+1) = exp2f(rP(i+1)*scale - new_maxs[row_idx]);
                    rS(i) = (bf16)rP(i);
                    rS(i+1) = (bf16)rP(i+1);
                    cur_sum += rP(i) + rP(i+1);
                }
                rL[row_idx] = rL[row_idx]*scale_for_o + cur_sum;
            }
            __syncwarp();
            if (idx_in_warpgroup%4 == 0) {
                plan.sM[idx_in_warpgroup/4] = *(float2*)new_maxs;
            }
            rM[0] = new_maxs[0];
            rM[1] = new_maxs[1];
        };

        auto reduce_L = [&]() {
            // Reduce L
            // For example, thread 0 reduces with thread 1, 2, and 3, as well as thread 128, 129, 130, and 131
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
            if (idx_in_warpgroup%4 == 0)
                plan.sL[threadIdx.x/4] = *(float2*)(rL);
            NamedBarrier::arrive_and_wait(256, NamedBarriers::sL_ready);
            float2 peer_L = plan.sL[(threadIdx.x/4)^32];
            rL[0] += peer_L.x;
            rL[1] += peer_L.y;
        };

        auto store_O = [&]() {
            float scale_factors[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i)
                scale_factors[i] = rL[i] == 0.0f ? 1.0f : 1.0f / rL[i];

            Tensor sO = make_tensor(make_smem_ptr(plan.q_o.o.data() + warpgroup_idx*B_H*(D_V/2)), SmemLayoutOTiles<4>{});
            bf16* stsm_addrs[4];
            int stsm_row = (idx_in_warpgroup/32)*16 + (idx_in_warpgroup%16);
            CUTE_UNROLL
            for (int i = 0; i < 64/16; ++i) {
                stsm_addrs[i] = &sO(stsm_row, (idx_in_warpgroup%32/16*8) + 16*i);
            }
            bool s2g_pred = warp_idx%4 == 0 && elect_one_sync();

            warpgroup_wait<0>();
            CUTE_UNROLL
            for (int tile_idx = 0; tile_idx < (D_V/2)/64; tile_idx += 1) {
                // Convert
                constexpr int NUM_ELEMS_EACH_TILE = B_H*64 / 128;   // 64: tile size, 128: warpgroup size
                bf16 cur_rOb[NUM_ELEMS_EACH_TILE];
                CUTE_UNROLL
                for (int i = 0; i < NUM_ELEMS_EACH_TILE; ++i) {
                    cur_rOb[i] = (bf16)(rO(tile_idx*NUM_ELEMS_EACH_TILE + i) * scale_factors[i%4>=2]);
                }
                // R -> S
                CUTE_UNROLL
                for (int i = 0; i < 64/16; ++i) {
                    SM90_U32x4_STSM_N::copy(
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 0),
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 2),
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 4),
                        *reinterpret_cast<uint32_t*>(cur_rOb + i*8 + 6),
                        *reinterpret_cast<uint128_t*>(stsm_addrs[i] + tile_idx*(B_H*64))
                    );
                }
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, warpgroup_idx ? NamedBarriers::warpgroup1_sync : NamedBarriers::warpgroup0_sync);
                // S -> G
                if (s2g_pred) {
                    int g_tile_idx = warpgroup_idx*4 + tile_idx;
                    SM90_TMA_STORE_3D::copy(
                        &tma_params.tensor_map_O,
                        plan.q_o.o.data() + g_tile_idx*(B_H*64),
                        g_tile_idx*64,
                        q_h_idx*B_H,
                        s_q_idx
                    );
                }
            }
            cute::tma_store_arrive();
        };


        if (warpgroup_idx == 0) {
            // Warpgroup 0

            auto pipelined_wait_and_qkt_gemm_l = [&]() __attribute__((always_inline)) {
                plan.bar_k0_ready[0].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup0{}, 0, true);
                qkt_gemm_one_tile(Warpgroup0{}, 1, false);
                qkt_gemm_one_tile(Warpgroup0{}, 2, false);
                qkt_gemm_one_tile(Warpgroup0{}, 3, false);
                warpgroup_commit_batch();
            };

            auto pipelined_wait_and_qkt_gemm_r = [&]() __attribute__((always_inline)) {
                plan.bar_k0_ready[1].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup0{}, 4, false);
                qkt_gemm_one_tile(Warpgroup0{}, 5, false);
                qkt_gemm_one_tile(Warpgroup0{}, 6, false);
                qkt_gemm_one_tile(Warpgroup0{}, 7, false);
                qkt_gemm_one_tile(Warpgroup0{}, 8, false);
                warpgroup_commit_batch();
            };

            auto scale_rS = [&](float scales[2]) {
                CUTE_UNROLL
                for (int row = 0; row < 2; ++row) {
                    CUTE_UNROLL
                    for (int i = row*2; i < size(rP); i += 4) {
                        rS(i) = (bf16)(rP(i) * scales[row]);
                        rS(i+1) = (bf16)(rP(i+1) * scales[row]);
                    }
                }
            };

            auto rescale_rO = [&](float scales[2]) {
                CUTE_UNROLL
                for (int row = 0; row < 2; ++row) {
                    CUTE_UNROLL
                    for (int i = row*2; i < size(rO); i += 4) {
                        rO(i) *= scales[row];
                        rO(i+1) *= scales[row];
                    }
                    rL[row] *= scales[row];
                }
            };
            
            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
                Tensor sV0l = make_tensor(make_smem_ptr(plan.k[0].data()), SmemLayoutKTilesTransposed<4>{});
                Tensor sV1l = make_tensor(make_smem_ptr(plan.k[1].data()), SmemLayoutKTilesTransposed<4>{});

                if (block_idx == 0) {
                    // NOTE We put these code here to avoid register spilling
                    pipelined_wait_and_qkt_gemm_l();
                    pipelined_wait_and_qkt_gemm_r();
                    warpgroup_wait<0>();
                }
                
                // Online softmax, inform WG1
                mask_rP(Warpgroup0{});
                
                online_softmax_and_rescale_o(Warpgroup0{});
                NamedBarrier::arrive(256, NamedBarriers::wg0_bunch_0_ready);

                // Issue rO0 += rS0 @ sV0l
                gemm_rs(false, TiledMMA_PV_LocalP{}, rS, sV0l, rO, idx_in_warpgroup);
                warpgroup_commit_batch();

                // Mark V0L as free
                warpgroup_wait<0>();
                plan.bar_k0_free[0].arrive();

                // Wait for new sM, scale rS, save, inform WG1
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_bunch_0_ready);
                float new_rM[2], scale_factors[2];
                *(float2*)new_rM = plan.sM[idx_in_warpgroup/4];
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    scale_factors[i] = exp2f(rM[i] - new_rM[i]);
                    rM[i] = new_rM[i];
                }
                scale_rS(scale_factors);
                save_rS_to_sS(rS, sS0, idx_in_warpgroup);
                fence_view_async_shared();
                NamedBarrier::arrive(256, NamedBarriers::wg0_s0_ready);

                // Wait for sS1
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg1_s1_ready);

                // Rescale rO0, Issue rO0 += sS1 @ sV1L
                rescale_rO(scale_factors);
                gemm_ss(false, TiledMMA_PV_RemoteP{}, sS1, sV1l, rO, idx_in_warpgroup);
                warpgroup_commit_batch();

                cur_bar_wait_phase ^= 1;

                if (block_idx+2 < num_topk_blocks) {
                    // Launch the next QK^T GEMM
                    pipelined_wait_and_qkt_gemm_l();

                    // Mark V1L as free
                    warpgroup_wait<1>();
                    plan.bar_k1_free[0].arrive();
                    pipelined_wait_and_qkt_gemm_r();

                    // Wait for rP0 = sQ @ sK0
                    warpgroup_wait<0>();
                } else {
                    // Mark V1L as free
                    warpgroup_wait<0>();
                    plan.bar_k1_free[0].arrive();
                }
            }

            reduce_L();
            store_O();
        } else {
            // Warpgroup 1

            auto pipelined_wait_and_qkt_gemm = [&]() __attribute__((always_inline)) {
                plan.bar_k1_ready[1].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup1{}, 4, true);
                qkt_gemm_one_tile(Warpgroup1{}, 5, false);
                qkt_gemm_one_tile(Warpgroup1{}, 6, false);
                qkt_gemm_one_tile(Warpgroup1{}, 7, false);
                qkt_gemm_one_tile(Warpgroup1{}, 8, false);
                plan.bar_k1_ready[0].wait(cur_bar_wait_phase);
                qkt_gemm_one_tile(Warpgroup1{}, 0, false);
                qkt_gemm_one_tile(Warpgroup1{}, 1, false);
                qkt_gemm_one_tile(Warpgroup1{}, 2, false);
                qkt_gemm_one_tile(Warpgroup1{}, 3, false);
                warpgroup_commit_batch();
            };
            
            CUTE_NO_UNROLL
            for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
                Tensor sV0r = make_tensor(make_smem_ptr(plan.k[0].data()+64*256), SmemLayoutKTilesTransposed<4>{});
                Tensor sV1r = make_tensor(make_smem_ptr(plan.k[1].data()+64*256), SmemLayoutKTilesTransposed<4>{});

                // Issue rP1 = sQ @ sK1, and wait
                pipelined_wait_and_qkt_gemm();
                warpgroup_wait<0>();

                mask_rP(Warpgroup1{});

                // Wait for WG0 (for sM), online softmax, Notify WG0 (sM ready)
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_bunch_0_ready);
                online_softmax_and_rescale_o(Warpgroup1{});
                NamedBarrier::arrive(256, NamedBarriers::wg1_bunch_0_ready);


                // Issue rO1 += rS1 @ sV1R
                gemm_rs(false, TiledMMA_PV_LocalP{}, rS, sV1r, rO, idx_in_warpgroup);
                warpgroup_commit_batch();
                
                // Wait for WG0 (for sS0), Issue rO1 += rS0 @ sV0R
                save_rS_to_sS(rS, sS1, idx_in_warpgroup);   // Put it here is faster
                NamedBarrier::arrive_and_wait(256, NamedBarriers::wg0_s0_ready);
                gemm_ss(false, TiledMMA_PV_RemoteP{}, sS0, sV0r, rO, idx_in_warpgroup);
                warpgroup_commit_batch();
                
                // Save rS1, inform WG0
                fence_view_async_shared();
                NamedBarrier::arrive(256, NamedBarriers::wg1_s1_ready);

                // Wait for GEMM, and inform that sV1R is free
                warpgroup_wait<1>();
                plan.bar_k1_free[1].arrive();

                // Wait for GEMM, and inform that sV0R is free
                warpgroup_wait<0>();
                plan.bar_k0_free[1].arrive();

                cur_bar_wait_phase ^= 1;
            }

            reduce_L();
            store_O();

            // Save lse
            if (idx_in_warpgroup%4 == 0) {
                for (int row = 0; row < 2; ++row) {
                    int real_row = get_AorC_row_idx(row, idx_in_warpgroup);
                    bool is_no_valid_tokens = rL[row] == 0.0f;
                    plan.final_max_logits[real_row] = is_no_valid_tokens ? -INFINITY : rM[row];
                    plan.final_lse[real_row] = is_no_valid_tokens ? -INFINITY : log2f(rL[row]) + rM[row];
                }
                fence_view_async_shared();
            }

            NamedBarrier::arrive_and_wait(128, NamedBarriers::warpgroup1_sync);
            if (idx_in_warpgroup == 0) {
                int g_offset = s_q_idx*params.h_q + q_h_idx*B_H;
                SM90_BULK_COPY_S2G::copy(plan.final_max_logits, params.max_logits + g_offset, B_H*sizeof(float));
                SM90_BULK_COPY_S2G::copy(plan.final_lse, params.lse + g_offset, B_H*sizeof(float));
                cute::tma_store_arrive();
            }
        }
    } else {
        // Producer warpgroup
        cutlass::arch::warpgroup_reg_dealloc<72>();

        constexpr int GROUP_SIZE = 8, NUM_GROUPS = 128/GROUP_SIZE;
        constexpr int NUM_ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
        int idx_in_group = idx_in_warpgroup % GROUP_SIZE;
        int group_idx = idx_in_warpgroup / GROUP_SIZE;
        int* gIndices = params.indices + s_q_idx*params.topk;   // [topk]

        bf16* my_sKV_base = &(make_tensor(make_smem_ptr(plan.k[0].data()), SmemLayoutKTiles<1>{})(group_idx, idx_in_group*8));
        bf16* my_gKV_base = params.kv + idx_in_group*8;
        
        int64_t token_indices[2][NUM_ROWS_PER_GROUP];
        bool is_token_valid[2][NUM_ROWS_PER_GROUP];
        auto load_token_indices = [&](int block_idx) {
            CUTE_UNROLL
            for (int buf_idx = 0; buf_idx < 2; ++buf_idx) {
                CUTE_UNROLL
                for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                    int offs = (block_idx+buf_idx)*B_TOPK + local_row*NUM_GROUPS + group_idx;
                    int t = __ldg(gIndices + offs);
                    token_indices[buf_idx][local_row] = t*(int64_t)params.stride_kv_s_kv;   // We mult it with params.stride_kv_s_kv here since it's faster
                    is_token_valid[buf_idx][local_row] = t >= 0 && t < params.s_kv;
                }
            }
        };
        
        int64_t cache_policy = createpolicy_evict_last();
        auto copy_tiles = [&](int block_idx, int buf_idx, int tile_start, int tile_end) {
            // Copy some K/V tiles from global memory to shared memory
            // A tile has a shape of 64 (B_TOPK) x 64
            // `buf_idx` is the index of the shared memory buffer, 0 or 1
            // `tile_idx` is the index of the tile to load, from 0 to D_K/64-1 = 8
            CUTE_UNROLL
            for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row) {
                int64_t token_index = token_indices[buf_idx][local_row];
                CUTE_UNROLL
                for (int tile_idx = tile_start; tile_idx < tile_end; ++tile_idx) {
                    cp_async_cacheglobal_l2_prefetch_256B(
                        my_gKV_base + token_index + tile_idx*64,
                        my_sKV_base + (buf_idx*B_TOPK*D_K + tile_idx*(B_TOPK*64) + local_row*NUM_GROUPS*64),
                        is_token_valid[buf_idx][local_row],
                        cache_policy
                    );
                }
            }
        };

        auto commit_to_mbar = [&](transac_bar_t &bar) {
            cutlass::arch::cpasync_barrier_arrive_noinc((uint64_t*)(&bar));
        };

        int cur_bar_wait_phase = 1;

        CUTE_NO_UNROLL
        for (int block_idx = 0; block_idx < num_topk_blocks; block_idx += 2) {
            load_token_indices(block_idx);

            // V0L
            plan.bar_k0_free[0].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+0, 0, 0, 4);
            commit_to_mbar(plan.bar_k0_ready[0]);

            // V1R
            plan.bar_k1_free[1].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+1, 1, 4, 9);
            commit_to_mbar(plan.bar_k1_ready[1]);
            
            // V0R
            plan.bar_k0_free[1].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+0, 0, 4, 9);
            commit_to_mbar(plan.bar_k0_ready[1]);

            // V1L
            plan.bar_k1_free[0].wait(cur_bar_wait_phase);
            copy_tiles(block_idx+1, 1, 0, 4);
            commit_to_mbar(plan.bar_k1_ready[0]);

            // Valid mask
            // NOTE V1R's finish implies maskings of the last round have finished
            if (idx_in_group == 0) {
                CUTE_UNROLL
                for (int buf_idx = 0; buf_idx < 2; ++buf_idx)
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < NUM_ROWS_PER_GROUP; ++local_row)
                        plan.is_kv_valid[buf_idx][local_row*NUM_GROUPS+group_idx] = is_token_valid[buf_idx][local_row];
                plan.bar_is_kv_valid_ready.arrive();
            }

            cur_bar_wait_phase ^= 1;
        }
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif
}


void run_fwd_kernel(const SparsePrefillParams& params) {
    FLASH_ASSERT(params.h_kv == 1);
    FLASH_ASSERT(params.topk % (2*B_TOPK) == 0);   // To save some boundry checkings
    FLASH_ASSERT(params.topk > 0);
    FLASH_ASSERT(params.h_q % B_H == 0);

    auto shape_Q = make_shape(params.h_q, params.d_qk, params.s_q);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q),
            make_layout(
                shape_Q,
                make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q)
            )
        ),
        SmemLayoutQ{}
    );

    CUtensorMap tensor_map_O;
    {
        uint64_t size[3] = {D_V, (unsigned long)params.h_q, (unsigned long)params.s_q};
        uint64_t stride[2] = {D_V*sizeof(bf16), D_V*params.h_q*sizeof(bf16)};
        uint32_t box_size[3] = {64, B_H, 1};
        uint32_t elem_stride[3] = {1, 1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_O,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            3,
            params.out,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        FLASH_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TmaParams<
        decltype(shape_Q), decltype(tma_Q)
    > tma_params = {
        shape_Q, tma_Q,
        tensor_map_O
    };
    auto kernel = &sparse_attn_fwd_kernel<decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cutlass::ClusterLaunchParams launch_params = {
        dim3((params.h_q/B_H)*params.s_q, 1, 1),    // NOTE We put s_q on the first dim since it can be larger than 65536 (the maximum size of griddim.y and griddim.z)
        dim3(NUM_THREADS, 1, 1),
        dim3(1, 1, 1),
        smem_size,
        params.stream
    }; 
    cutlass::launch_kernel_on_cluster(
        launch_params, (void*)kernel, params, tma_params
    );
    CHECK_CUDA_KERNEL_LAUNCH();
}

}

#else // !SM90+ architecture

namespace sm90 {

void run_fwd_kernel(const SparsePrefillParams& params) {
    throw std::runtime_error("FlashMLA sparse prefill kernel requires SM90+ architecture. This build was compiled without SM90 support.");
}

}

#endif // SM90+ architecture check
