#include "splitkv_mla.h"

#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>

#include "utils.h"
#include "components/config.h"
#include "components/epilogue.h"
#include "components/helpers.h"
#include "components/named_barriers.h"
#include "components/dequant.h"
using namespace cute;

namespace sm90 {

static constexpr float MAX_INIT_VAL = -1e30;    // Prevent (-inf) - (-inf) = nan
using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;

// Save rPb (64x64, bfloat16) to sP using the stmatrix instruction
template<
    typename Tensor0,
    typename Tensor1
>
__forceinline__ __device__ void save_rPb_to_sP(
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


// Retrieve rPb (64x64, bfloat16) from sP using the ldmatrix instruction
template<
    typename Tensor0,
    typename Tensor1
>
__forceinline__ __device__ void retrieve_rP_from_sP(
    Tensor0 &rPb,
    Tensor1 const &sP,
    int idx_in_warpgroup
) {
    TiledCopy s2r_copy = make_tiled_copy_A(
        Copy_Atom<SM75_U32x4_LDSM_N, bf16>{},
        TiledMMA_PV_LocalP{}
    );
    ThrCopy thr_copy = s2r_copy.get_slice(idx_in_warpgroup);
    Tensor thr_copy_sP = thr_copy.partition_S(sP);
    Tensor thr_copy_rPb = thr_copy.retile_D(rPb);
    cute::copy(s2r_copy, thr_copy_sP, thr_copy_rPb);
}


template<
    typename Tensor0,
    typename Tensor1,
    typename Tensor2
>
__forceinline__ __device__ void scale_softmax(
    Tensor0 &rP,
    Tensor1 &rS,
    Tensor2 &rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    bool is_kv_valid[],
    int block_idx,
    int idx_in_warpgroup
) {
    float scale_for_olds[2];
    CUTE_UNROLL
    for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
        Tensor cur_rP = flatten(rP(make_coord(_, local_row_idx, _), _, _));
        Tensor cur_rS = flatten(rS(make_coord(_, local_row_idx, _), _, _));
        Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));

        float cur_max = -INFINITY;
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            if (!is_kv_valid[(i&1)+(i/2)*8+(idx_in_warpgroup%4)*2])
                cur_rP(i) = -INFINITY;
            cur_max = max(cur_max, cur_rP(i));
        }
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
        cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

        cur_max *= scale_softmax_log2;
        float old_max = rM[local_row_idx];
        rM[local_row_idx] = max(cur_max, old_max);
        float scale_for_old = exp2f(old_max - rM[local_row_idx]);
        scale_for_olds[local_row_idx] = scale_for_old;

        CUTE_UNROLL
        for (int i = 0; i < size(cur_rO); ++i) {
            cur_rO(i) *= scale_for_old;
        }

        float cur_sum = 0;
        CUTE_UNROLL
        for (int i = 0; i < size(cur_rP); ++i) {
            cur_rP(i) = exp2f(cur_rP(i)*scale_softmax_log2 - rM[local_row_idx]);
            cur_rS(i) = (bf16)cur_rP(i);
            cur_sum += cur_rP(i);
        }
        rL[local_row_idx] = rL[local_row_idx]*scale_for_old + cur_sum;
    }
    if (idx_in_warpgroup%4 == 0)
        *(float2*)(sScale + 2*(idx_in_warpgroup/4)) = *(float2*)(scale_for_olds);
}

template<typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 2)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const DecodingParams params, __grid_constant__ const TmaParams tma_params) {
#if IS_SM90
    const int head_block_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int idx_in_cluster = head_block_idx % 2;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
    Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
    Tensor sOAccumBuf = make_tensor(make_smem_ptr(plan.u.oAccumBuf.data()), SmemLayoutOAccumBuf{});
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
    float* sM = plan.sM;
    float* sL = plan.sL;
    float* sScale = plan.sScale;
    
    // Prefetch TMA descriptors
    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
    }
    
    // Initialize TMA barriers
    if (warp_idx == 0 && elect_one_sync()) {
        plan.bar_q.init(1);
        CUTE_UNROLL
        for (int i = 0; i < NUM_K_BUFS; ++i) {
            plan.bar_k_local_ready[i].init(128);
            plan.bar_k_remote_ready[i].init(1);
            plan.bar_k_avail[i].init(4);
        }
        fence_view_async_shared();
    }
    cute::cluster_arrive();

    bool bar_phase_q = 0;
    int bar_phase_k = 0; // Don't use array here to prevent using local memory

    // Programmatic Dependent Launch: Wait for the previous kernel to finish
    // Don't use PDL because of compiler bugs!
    // cudaGridDependencySynchronize();
    
    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int sched_begin_block_idx = tile_scheduler_metadata.y;
    int end_idx = tile_scheduler_metadata.z;
    int sched_end_block_idx = tile_scheduler_metadata.w;
    if (begin_idx >= params.b || begin_idx < 0) return;
    int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4);

    if (warp_idx == 0 && elect_one_sync()) {
        Tensor gQ = flat_divide(
            tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, begin_idx),
            Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
        )(_, _, head_block_idx, _0{});
        launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
        plan.bar_q.arrive_and_expect_tx(BLOCK_M*HEAD_DIM_K*sizeof(bf16));
    }

    cute::cluster_wait();   // Wait for barriers from the other CTA to be ready

    auto get_cur_req_info = [&](int batch_idx) -> std::tuple<int, int, bool> {
        constexpr int kBlockN = TOPK_BLOCK_SIZE;
        const int start_block_idx = batch_idx == begin_idx ? sched_begin_block_idx : 0;
        // NOTE TopK attention has nothing to do with causal mask and sliding window
        int end_block_idx = batch_idx == end_idx ? sched_end_block_idx : cute::ceil_div(params.topk, kBlockN);
        const bool is_no_split = start_block_idx == 0 && end_block_idx == cute::ceil_div(params.topk, kBlockN);
        return {start_block_idx, end_block_idx, is_no_split};
    };

    if (warpgroup_idx == 0) {
        cutlass::arch::warpgroup_reg_alloc<192>();

        TiledMMA tiled_mma_QK = TiledMMA_QK{};
        ThrMMA thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
        TiledMMA tiled_mma_PV = TiledMMA_PV_LocalP{};
        ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
        
        float rL[2], rM[2];
        Tensor rO = partition_fragment_C(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V/2>>{});
        Tensor rP = partition_fragment_C(TiledMMA_QK{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
        Tensor rS = make_tensor<bf16>(partition_shape_A(TiledMMA_PV_LocalP{}, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

        #pragma unroll 1
        for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
            auto [start_block_idx, end_block_idx, is_no_split] = get_cur_req_info(batch_idx);

            rL[0] = rL[1] = 0.0f;
            rM[0] = rM[1] = MAX_INIT_VAL;
            cute::fill(rO, 0.);

            // Wait for Q
            plan.bar_q.wait(bar_phase_q);
            bar_phase_q ^= 1;

            CUTE_NO_UNROLL
            for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx++) {
                int buf_idx = (block_idx-start_block_idx) % NUM_K_BUFS;
                Tensor sK = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutK{});
                Tensor sV = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutHalfV{});

                // Wait, issue WGMMA
                plan.bar_k_local_ready[buf_idx].wait(bar_phase_k>>buf_idx&1);
                plan.bar_k_remote_ready[buf_idx].wait(bar_phase_k>>buf_idx&1);

                gemm<true, -1>(
                    tiled_mma_QK,
                    thr_mma_QK.partition_fragment_A(sQ),
                    thr_mma_QK.partition_fragment_B(sK),
                    rP
                );

                bar_phase_k ^= 1<<buf_idx;

                cute::warpgroup_wait<0>();
                
                // Calculate S = softmax(mask(scale(P)))
                if (block_idx != start_block_idx)
                    NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_free);  // Make sure that sScale and sS is free

                // Since in our case TOPK_BLOCK_SIZE == BLOCK_M, so we only need to do OOB checking for the last 2 blocks
                scale_softmax(rP, rS, rO, params.scale_softmax_log2, sScale, rM, rL, plan.is_kv_valid[buf_idx], block_idx, idx_in_warpgroup);

                // Store S into shared, inform warpgroup 1
                save_rPb_to_sP(rS, sS, idx_in_warpgroup);
                fence_view_async_shared();

                // Issue O += S @ V
                gemm<false, -1>(
                    tiled_mma_PV,
                    rS,
                    thr_mma_PV.partition_fragment_B(sV),
                    rO
                );

                NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_ready);

                cute::warpgroup_wait<0>();

                plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
                plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
            }

            // Copy the next q
            if (warp_idx == 0 && elect_one_sync()) {
                if (batch_idx != end_idx) {
                    Tensor gQ = flat_divide(
                        tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx+1),
                        Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
                    )(_, _, head_block_idx, _0{});
                    launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
                    plan.bar_q.arrive_and_expect_tx(BLOCK_M*HEAD_DIM_K*sizeof(bf16));
                } else {
                    cudaTriggerProgrammaticLaunchCompletion();
                }
            }

            // Synchronize L and M across warpgroups
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
            rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
            rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);
            if (idx_in_warpgroup%4 == 0) {
                CUTE_UNROLL
                for (int i = 0; i < 2; ++i) {
                    int row = get_AorC_row_idx(i, idx_in_warpgroup);
                    sL[row] = rL[i];
                    sM[row] = rM[i];
                }
            }

            // This is a synchronization point for warpgroup 0/1.
            // Warpgroup 0 should wait wg 1 for oBuf/oAccumBuf (overlapped with k) to be free
            // Warpgroup 1 should wait wg 0 for sL to be ready
            NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);

            CUTE_UNROLL
            for (int i = 0; i < 2; ++i)
                rL[i] = rL[i] == 0.0f ? 1.0f : rL[i];
            
            int num_valid_seq_q = min(params.q_head_per_hk - head_block_idx*BLOCK_M, BLOCK_M);
            int start_seq_idx = s_q_idx*params.q_head_per_hk + head_block_idx*BLOCK_M;
            if (is_no_split) {
                bf16* o_ptr = (bf16*)params.o_ptr + batch_idx*params.o_batch_stride + start_seq_idx*params.o_row_stride;	// (BLOCK_M, HEAD_DIM_V) : (params.o_row_stride, 1)
                Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                    make_stride(params.o_row_stride, _1{})
                ));
                float* gSoftmaxLse = (float*)params.softmax_lse_ptr + batch_idx*params.q_seq_per_hk + start_seq_idx;	// (BLOCK_M) : (1)

                store_o<true>(rO, gO, sOBuf, sOAccumBuf, rL, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLse[i] = cur_L == 0.0f ? INFINITY : logf(cur_L) + sM[i] / (float)M_LOG2E;
                }

                cute::tma_store_wait<0>();
            } else {
                int n_split_idx = batch_idx == begin_idx ? begin_n_split_idx : 0;
                int split_idx = __ldg(params.num_splits_ptr+batch_idx) + n_split_idx;
                float* oaccum_ptr = (float*)params.oaccum_ptr + (split_idx*params.q_seq_per_hk + start_seq_idx)*HEAD_DIM_V;	// (BLOCK_M, HEAD_DIM_V) : (HEAD_DIM_V, 1)
                float* gSoftmaxLseAccum = (float*)params.softmax_lseaccum_ptr + split_idx*params.q_seq_per_hk + start_seq_idx;	// (BLOCK_M) : (1)
                Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), Layout<
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>,
                    Stride<Int<HEAD_DIM_V>, _1>
                >{});
                store_o<false>(rO, gOAccum, sOBuf, sOAccumBuf, rL, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLseAccum[i] = cur_L == 0.0f ? -INFINITY : log2f(cur_L) + sM[i];
                }

                cute::tma_store_wait<0>();
            }
            
            cute::cluster_sync();   // Must use arrive_and_wait here to prevent overwritting sL while WG1 is writing back its result
        }
    } else if (warpgroup_idx == 1) {
        cutlass::arch::warpgroup_reg_dealloc<160>();

        TiledMMA tiled_mma_PV = TiledMMA_PV_RemoteP{};
        ThrMMA thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);
        Tensor rO = partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V/2>>{});
        float rL[2];

        #pragma unroll 1
        for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
            auto [start_block_idx, end_block_idx, is_no_split] = get_cur_req_info(batch_idx);
            cute::fill(rO, 0.);

            CUTE_NO_UNROLL
            for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx++) {
                int buf_idx = (block_idx-start_block_idx) % NUM_K_BUFS;
                Tensor sV = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data() + (SmemLayoutV{})(_256{}, _0{})), SmemLayoutHalfV{});

                // Wait for S and sScale
                NamedBarrier::arrive_and_wait(256, NamedBarriers::sScale_and_sS_ready);

                // Scale O
                float cur_scales[2];
                *(float2*)cur_scales = *(float2*)(sScale + (idx_in_warpgroup/4)*2);
                CUTE_UNROLL
                for (int local_row_idx = 0; local_row_idx < 2; ++local_row_idx) {
                    Tensor cur_rO = flatten(rO(make_coord(_, local_row_idx, _), _, _));
                    CUTE_UNROLL
                    for (int i = 0; i < size(cur_rO); ++i) {
                        cur_rO(i) *= cur_scales[local_row_idx];
                    }
                }
                
                // Issue O += S @ V, and wait
                gemm<false, -1>(
                    tiled_mma_PV,
                    thr_mma_PV.partition_fragment_A(sS),
                    thr_mma_PV.partition_fragment_B(sV),
                    rO
                );
                cute::warpgroup_wait<0>();
                
                plan.bar_k_avail[buf_idx].arrive(0, idx_in_warpgroup == 32);
                plan.bar_k_avail[buf_idx].arrive(1, idx_in_warpgroup == 64);
                
                if (block_idx != end_block_idx-1)
                    NamedBarrier::arrive(256, NamedBarriers::sScale_and_sS_free);   // Tell WG0 that sScale and sS are available
            }

            NamedBarrier::arrive_and_wait(256, NamedBarriers::oBuf_free_and_sL_ready);
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                int row = get_AorC_row_idx(i, idx_in_warpgroup);
                rL[i] = sL[row];
            }

            CUTE_UNROLL
            for (int i = 0; i < 2; ++i)
                rL[i] = rL[i] == 0.0f ? 1.0f : rL[i];
                
            int num_valid_seq_q = min(params.q_head_per_hk - head_block_idx*BLOCK_M, BLOCK_M);
            int start_seq_idx = s_q_idx*params.q_head_per_hk+head_block_idx*BLOCK_M;
            if (is_no_split) {
                bf16* o_ptr = (bf16*)params.o_ptr + batch_idx*params.o_batch_stride + start_seq_idx*params.o_row_stride;	// (BLOCK_M, HEAD_DIM_V) : (params.o_row_stride, 1)
                Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                    make_stride(params.o_row_stride, _1{})
                ));

                store_o<true>(rO, gO, sOBuf, sOAccumBuf, rL, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                cute::tma_store_wait<0>();
            } else {
                int n_split_idx = batch_idx == begin_idx ? begin_n_split_idx : 0;
                int split_idx = __ldg(params.num_splits_ptr+batch_idx) + n_split_idx;
                float* oaccum_ptr = (float*)params.oaccum_ptr + (split_idx*params.q_seq_per_hk + start_seq_idx)*HEAD_DIM_V;	// (BLOCK_M, HEAD_DIM_V) : (HEAD_DIM_V, 1)
                Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), Layout<
                    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>,
                    Stride<Int<HEAD_DIM_V>, _1>
                >{});
                store_o<false>(rO, gOAccum, sOBuf, sOAccumBuf, rL, tma_params, batch_idx, s_q_idx, head_block_idx, num_valid_seq_q, warpgroup_idx, idx_in_warpgroup);

                cute::tma_store_wait<0>();
            }

            cute::cluster_sync();   // We must use arrive_and_wait instead of arrive here to create an order between "forall warp in WG1, warp has done written back O" and "warp 2 signals `bar_k_avail`"
        }
    } else {
        // Producer warpgroup
        cutlass::arch::warpgroup_reg_dealloc<152>();

        int warp_idx = __shfl_sync(0xffffffff, idx_in_warpgroup / 32, 0);   // NOTE TPBNO
        int lane_idx = idx_in_warpgroup % 32;
        int my_token_idx = warp_idx*8 + lane_idx%8;
        
        CUTE_NO_UNROLL
        for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
            auto [start_block_idx, end_block_idx, is_no_split] = get_cur_req_info(batch_idx);
            int* gIndices = params.indices_ptr + batch_idx*params.indices_batch_stride + s_q_idx*params.indices_row_stride; // (topk) : (1)
            
            #define GET_TOKEN_INDEX(block_idx) __ldg(gIndices + (block_idx)*TOPK_BLOCK_SIZE + idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)
            int nxt_token_index = GET_TOKEN_INDEX(start_block_idx);

            CUTE_NO_UNROLL
            for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx++) {
                int buf_idx = (block_idx-start_block_idx) % NUM_K_BUFS;

                // Define shared and global tensors
                bf16* sK_nope_base = plan.u.k[buf_idx].data() + (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)*8 + ((lane_idx/8)*16)*TOPK_BLOCK_SIZE;
                bf16* sK_nope_peer_base = get_peer_addr(sK_nope_base);
                
                transac_bar_t* peer_bar_k_remote_ready = get_peer_addr(&(plan.bar_k_remote_ready[buf_idx]));
                int token_index = nxt_token_index;
                if (block_idx+1 != end_block_idx)
                    nxt_token_index = GET_TOKEN_INDEX(block_idx+1);
                int block_index = token_index/PAGE_BLOCK_SIZE;
                int rel_idx_in_block = (token_index+PAGE_BLOCK_SIZE) % PAGE_BLOCK_SIZE;   // NOTE When token_index is -1, -1/PAGE_BLOCK_SIZE = 0 and (-1+PAGE_BLOCK_SIZE)%PAGE_BLOCK_SIZE = 63, so there will be no illegal-memory-access error
                fp8* gK_base = (fp8*)params.k_ptr + block_index*params.k_batch_stride + rel_idx_in_block*params.k_row_stride;
                float4 scales = load_128b_from_gmem<float4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>((float*)(gK_base+HEAD_DIM_NOPE));

                // Wait for the nope buffer to be available
                plan.bar_k_avail[buf_idx].wait((bar_phase_k>>buf_idx&1)^1);
                bar_phase_k ^= 1 << buf_idx;
                
                // Copy block #block_index
                if (idx_in_warpgroup == 0) {
                    plan.bar_k_remote_ready[buf_idx].arrive_and_expect_tx((TOPK_BLOCK_SIZE/2)*(HEAD_DIM_NOPE+HEAD_DIM_ROPE)*sizeof(bf16));
                }

                // Collectively copy from global memory and dequant
                // For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
                
                fp8* gK_nope = gK_base + (lane_idx/8)*16;
                if (token_index == -1) {
                    scales = {0.0f, 0.0f, 0.0f, 0.0f};
                }
                CUTE_UNROLL
                for (int dim_idx = 0; dim_idx < HEAD_DIM_NOPE/64; dim_idx += 1) {
                    fp8x16 cur_fp8x16 = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(gK_nope + dim_idx*64);   // We use EVICT_LAST here since gK_base may not be aligned to 32B
                    float scale = dim_idx < 4 ? (dim_idx < 2 ? scales.x : scales.y) : (dim_idx < 6 ? scales.z : scales.w);
                    auto dequant_and_save_bf16x8 = [&](const fp8x8 &data, int offset) {
                        int smem_offset = (dim_idx*64 + offset) * TOPK_BLOCK_SIZE;
                        bf16x8 cur_bf16x8 = cvt_fp8x8_bf16x8(data, scale);
                        *(__int128_t*)(sK_nope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
                        st_async_128b(sK_nope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
                    };
                    if (token_index == -1)
                        *(uint128_t*)(&cur_fp8x16) = uint128_t();
                    dequant_and_save_bf16x8(cur_fp8x16.lo, 0);
                    dequant_and_save_bf16x8(cur_fp8x16.hi, 8);
                }

                bf16* gK_rope = (bf16*)(gK_base+HEAD_DIM_NOPE+NUM_SCALES*sizeof(float)) + (lane_idx/8)*8;
                bf16* sK_rope_base = plan.u.k[buf_idx].data() + (idx_in_cluster*(TOPK_BLOCK_SIZE/2) + my_token_idx)*8 + ((lane_idx/8)*8)*TOPK_BLOCK_SIZE;
                bf16* sK_rope_peer_base = get_peer_addr(sK_rope_base);

                CUTE_UNROLL
                for (int dim_idx = 0; dim_idx < HEAD_DIM_ROPE/32; dim_idx += 1) {
                    bf16x8 cur_bf16x8 = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_rope + dim_idx*32);
                    if (token_index == -1)
                        *(uint128_t*)(&cur_bf16x8) = uint128_t();
                    int smem_offset = (HEAD_DIM_NOPE + dim_idx*32) * TOPK_BLOCK_SIZE;
                    *(__int128_t*)(sK_rope_base + smem_offset) = *(__int128_t*)&cur_bf16x8;
                    st_async_128b(sK_rope_peer_base + smem_offset, cur_bf16x8, peer_bar_k_remote_ready);
                }

                fence_view_async_shared();

                if (idx_in_warpgroup < 32) {
                    // We put this after fence_view_async_shared() since this won't be read by async proxy
                    int2 indices = __ldg((int2*)(gIndices + block_idx*TOPK_BLOCK_SIZE + lane_idx*2));
                    *(char2*)(&plan.is_kv_valid[buf_idx][lane_idx*2]) = {indices.x != -1, indices.y != -1};
                }

                // Signal the barrier
                plan.bar_k_local_ready[buf_idx].arrive();
            }

            cute::cluster_sync();
        }
    }

    if (begin_idx > end_idx) {
        cute::cluster_sync();    // Don't need a cluster_sync() when begin_idx <= end_idx, since the loop will execute at least once and the final statement is cluster_sync()
    }
#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm90");
    }
#endif

}


void run_flash_splitkv_mla_fp8_sparse_kernel(DecodingParams &params, cudaStream_t stream) {
    FLASH_ASSERT(params.h_k == 1);
    FLASH_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);

    auto shape_Q = make_shape(params.q_head_per_hk, params.d, params.s_q, params.b);
    auto tma_Q = cute::make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            make_gmem_ptr((bf16*)params.q_ptr),
            make_layout(
                shape_Q,
                make_stride(params.q_row_stride, _1{}, params.q_head_per_hk*params.q_row_stride, params.q_batch_stride)
            )
        ),
        SmemLayoutQ{}
    );

    auto shape_O = make_shape(params.q_head_per_hk, params.d_v, params.s_q, params.b);
    auto tma_O = cute::make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(
            make_gmem_ptr((bf16*)params.o_ptr),
            make_layout(
                shape_O,
                make_stride(params.o_row_stride, _1{}, params.q_head_per_hk*params.o_row_stride, params.o_batch_stride)
            )
        ),
        SmemLayoutOBuf{}
    );

    TmaParams<
        decltype(shape_Q), decltype(tma_Q),
        decltype(shape_O), decltype(tma_O)
    > tma_params = {
        shape_Q, tma_Q,
        shape_O, tma_O
    };
    auto mla_kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<decltype(tma_params)>;

    constexpr size_t smem_size = sizeof(SharedMemoryPlan);
    CHECK_CUDA(cudaFuncSetAttribute(mla_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    const int num_m_block = cute::ceil_div(params.q_head_per_hk, 2*BLOCK_M) * 2;
    // NOTE Don't use PDL because of potential compiler bugs!
    // cudaLaunchAttribute mla_kernel_attributes[1];
    // mla_kernel_attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    // mla_kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
    // cudaLaunchConfig_t mla_kernel_config = {
    //     dim3(num_m_block, params.h_k, params.num_sm_parts),
    //     dim3(NUM_THREADS, 1, 1),
    //     smem_size,
    //     stream,
    //     mla_kernel_attributes,
    //     1
    // };
    // cudaLaunchKernelEx(&mla_kernel_config, mla_kernel, params, tma_params);
    cutlass::ClusterLaunchParams launch_params = {
        dim3(num_m_block, params.s_q, params.num_sm_parts),
        dim3(NUM_THREADS, 1, 1),
        dim3(2, 1, 1),
        smem_size,
        stream
    };
    cutlass::launch_kernel_on_cluster(
        launch_params, (void*)mla_kernel, params, tma_params
    );
    CHECK_CUDA_KERNEL_LAUNCH();
}

}
