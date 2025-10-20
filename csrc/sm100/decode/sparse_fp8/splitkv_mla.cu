// SM100-specific file - only compile for SM100+ architectures
#include "splitkv_mla.h"
#include <stdexcept>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) || !defined(__CUDA_ARCH__)

#include <cutlass/barrier.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/tensor.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#include "utils.h"
#include "dequant.h"
#include "sm100/defines.h"
#include "sm100/helpers.h"
#include "sm100/intrinsics.h"
#include "sm100/ws_gemm.h"

namespace sm100 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using namespace cute;

constexpr int B_H = 64;
constexpr int B_TOPK = 64;
constexpr int D_K = 576;
constexpr int D_V = 512;
constexpr int NUM_BUFS = 2;
constexpr int NUM_THREADS = 128*3;
constexpr int NUM_WORKING_THREADS = 128 + 128 + 32;
constexpr float MAX_INIT_VAL = -1e30f;  // To avoid (-inf) - (-inf) = NaN

template<
    typename Shape_Q, typename TMA_Q,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q shape_Q; TMA_Q tma_Q;
    Shape_O shape_O; TMA_O tma_O;
};

namespace tmem_addr {
    constexpr int o = 0;    // o: [0, 256]
    constexpr int p = 256;  // p: [256, 288]
};

using SmemLayoutQ = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_K>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutOBuf = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},  // TODO This may lead to TMA double traffic
    Shape<Int<B_H>, Int<D_V>>{}
));

using SmemLayoutOAccumBuf = Layout<
    Shape<Int<B_H>, Int<D_V>>,
    Stride<Int<520>, _1>	// We use stride = 520 here to avoid bank conflict
>;

using SmemLayoutS = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_TOPK>>{},
    Step<_1, _2>{}
));

template<int NUM_TILES>
using SmemLayoutKTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed = decltype(composition(
    SmemLayoutKTiles<NUM_TILES>{},
    Layout<
        Shape<Int<64*NUM_TILES>, Int<B_TOPK>>,
        Stride<Int<B_TOPK>, _1>
    >{}
));

using SmemLayoutK = SmemLayoutKTiles<9>;
using SmemLayoutV = SmemLayoutKTilesTransposed<8>;

struct SharedMemoryPlan {
    array_aligned<bf16, cosize_v<SmemLayoutQ>> q;
    union {
        array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
        array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> o_accum_buf;
        array_aligned<bf16, cosize_v<SmemLayoutK>> k[NUM_BUFS];
    } u;
    array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    transac_bar_t bar_q;
    transac_bar_t bar_k_ready[NUM_BUFS], bar_k_free[NUM_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS];
    float rowwise_max_buf[128], rowwise_li_buf[128];
    bool is_token_valid[NUM_BUFS][B_TOPK];
    array_aligned<uint32_t, 1> tmem_start_addr;
};

using TiledMMA_QK = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, B_TOPK, UMMA::Major::K, UMMA::Major::K>{},
    Layout<Shape<_1, _1, _1>>{}
)); // TODO Use TS?

using TiledMMA_SV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{},
    Layout<Shape<_1, _1, _1>>{},
    Tile<Int<B_H>, Int<D_V>>{}
));

template<typename T>
CUTE_DEVICE
void store_128b(void* smem_ptr, const T &data) {
    static_assert(sizeof(T) == 16);
    *(__int128*)smem_ptr = *(__int128*)&data;
}

template<typename TmaParams>
__global__ void __launch_bounds__(NUM_THREADS, 1, 1)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const DecodingParams params, __grid_constant__ const TmaParams tma_params) {
#if IS_SM100
    const int head_block_idx = blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    // Define shared tensors
    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});

    if (warp_idx == 0 && elect_one_sync()) {
        cute::prefetch_tma_descriptor(tma_params.tma_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(tma_params.tma_O.get_tma_descriptor());
    }

    if (warp_idx == 0) {
        if (elect_one_sync()) {
            plan.bar_q.init(1);
            for (int i = 0; i < NUM_BUFS; ++i) {
                plan.bar_k_ready[i].init(128);
                plan.bar_k_free[i].init(1);
                plan.bar_qk_done[i].init(1);
                plan.bar_so_ready[i].init(128);
            }
            cutlass::arch::fence_barrier_init();
        }
        cute::TMEM::Allocator1Sm().allocate(512, plan.tmem_start_addr.data());
        TRAP_ONLY_DEVICE_ASSERT(plan.tmem_start_addr.data()[0] == 0);
        cute::TMEM::Allocator1Sm().release_allocation_lock();
    }
    __syncthreads();

    int bar_phase_k = 0;

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x;
    int sched_begin_block_idx = tile_scheduler_metadata.y;
    int end_idx = tile_scheduler_metadata.z;
    int sched_end_block_idx = tile_scheduler_metadata.w;
    if (begin_idx >= params.b || begin_idx < 0) {
        if (warp_idx == 0) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
        return;
    }

    auto get_cur_req_info = [&](int batch_idx) -> std::tuple<int, int, bool> {
        int start_block_idx = batch_idx == begin_idx ? sched_begin_block_idx : 0;
        int end_block_idx = batch_idx == end_idx ? sched_end_block_idx : params.topk / B_TOPK;
        bool is_no_split = start_block_idx == 0 && end_block_idx == params.topk / B_TOPK;
        return {start_block_idx, end_block_idx, is_no_split};
    };

    if (warpgroup_idx == 0) {
        // Producer warpgroup

        #pragma unroll 1
        for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
            auto [start_block_idx, end_block_idx, is_no_split] = get_cur_req_info(batch_idx);
            int* gIndices = params.indices_ptr + batch_idx*params.indices_batch_stride + s_q_idx*params.indices_row_stride; // (topk) : (1)

            constexpr int GROUP_SIZE = 4, NUM_GROUPS = 128 / GROUP_SIZE;
            constexpr int ROWS_PER_GROUP = B_TOPK / NUM_GROUPS;
            int group_idx = idx_in_warpgroup / GROUP_SIZE;
            int idx_in_group = idx_in_warpgroup % GROUP_SIZE;

            NamedBarrier::arrive_and_wait(NUM_WORKING_THREADS, 1);

            CUTE_NO_UNROLL
            for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx++) {
                int buf_idx = block_idx % NUM_BUFS;

                // Wait for buffer to be available
                plan.bar_k_free[buf_idx].wait(bar_phase_k>>buf_idx&1^1);

                // Load
                Tensor sK = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutK{});

                CUTE_UNROLL
                for (int local_row = 0; local_row < ROWS_PER_GROUP; ++local_row) {
                    int smem_row = group_idx + local_row*NUM_GROUPS;
                    int token_index = __ldg(gIndices + block_idx*B_TOPK + smem_row);
                    bool is_token_invalid = token_index == -1;
                    if (idx_in_group == 0)
                        plan.is_token_valid[buf_idx][smem_row] = !is_token_invalid;
                    if (is_token_invalid) {
                        uint128_t zeros = uint128_t{};
                        CUTE_UNROLL
                        for (int local_col = 0; local_col < D_V / (GROUP_SIZE*16); ++local_col) {
                            int col_base = local_col*(GROUP_SIZE*16) + idx_in_group*16;
                            store_128b(&sK(smem_row, col_base  ), zeros);
                            store_128b(&sK(smem_row, col_base+8), zeros);
                        }
                        CUTE_UNROLL
                        for (int local_col = 0; local_col < (D_K-D_V) / (GROUP_SIZE*8); ++local_col) {
                            int col_base = local_col*(GROUP_SIZE*8) + idx_in_group*8;
                            store_128b(&sK(smem_row, D_V+col_base), zeros);
                        }
                    } else {
                        int block_index = token_index/B_TOPK;
                        int rel_idx_in_block = (token_index+B_TOPK) % B_TOPK;   // NOTE When token_index is -1, -1/B_TOPK = 0 and (-1+B_TOPK)%B_TOPK = 63, so there will be no illegal-memory-access error. However, masking is necessary to prevent NaN (TODO Skip some rows instead?) TODO Masking
                        fp8* gK_base = (fp8*)params.k_ptr + block_index*params.k_batch_stride + rel_idx_in_block*params.k_row_stride;
                        float4 scales = __ldg((float4*)(gK_base + D_V));

                        CUTE_UNROLL
                        for (int local_col = 0; local_col < D_V / (GROUP_SIZE*16); ++local_col) {
                            int col_base = local_col*(GROUP_SIZE*16) + idx_in_group*16;
                            fp8x16 cur_fp8s = ldg_128_fp8x16(gK_base + col_base);
                            float cur_scale = local_col < (256/(GROUP_SIZE*16)) ?
                                (local_col < (128/(GROUP_SIZE*16)) ? scales.x : scales.y) :
                                (local_col < (384/(GROUP_SIZE*16)) ? scales.z : scales.w);
                            store_128b(&sK(smem_row, col_base  ), cvt_fp8x8_bf16x8(cur_fp8s.a0, cur_scale));
                            store_128b(&sK(smem_row, col_base+8), cvt_fp8x8_bf16x8(cur_fp8s.a1, cur_scale));
                        }

                        CUTE_UNROLL
                        for (int local_col = 0; local_col < (D_K-D_V) / (GROUP_SIZE*8); ++local_col) {
                            int col_base = local_col*(GROUP_SIZE*8) + idx_in_group*8;
                            fp8x16 cur_k_rope_fp8s = ldg_128_fp8x16(gK_base + D_V + 4*sizeof(float) + col_base*sizeof(bf16));
                            bf16x8 cur_k_rope = *reinterpret_cast<bf16x8*>(&cur_k_rope_fp8s);
                            store_128b(&sK(smem_row, D_V+col_base), cur_k_rope);
                        }
                    }
                }

                fence_view_async_shared();

                // Signal
                plan.bar_k_ready[buf_idx].arrive();

                bar_phase_k ^= 1<<buf_idx;
            }
        }
    } else if (warpgroup_idx == 1) {
        // Scale & Exp warpgroup
        cutlass::arch::warpgroup_reg_alloc<240>();

        int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4);

        #pragma unroll 1
        for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
            auto [start_block_idx, end_block_idx, is_no_split] = get_cur_req_info(batch_idx);

            NamedBarrier::arrive_and_wait(NUM_WORKING_THREADS, 1);

            float li = 0.0f;
            float mi = MAX_INIT_VAL;

            CUTE_NO_UNROLL
            for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx++) {
                int buf_idx = block_idx % NUM_BUFS;

                // Wait for P
                plan.bar_qk_done[buf_idx].wait(bar_phase_k>>buf_idx&1);
                tcgen05_after_thread_sync();

                // Load P from TMEM
                float p[B_TOPK/2];
                float2* p_float2 = reinterpret_cast<float2*>(p);
                tmem_ld_32dp32bNx<B_TOPK/2>(tmem_addr::p, p);
                cutlass::arch::fence_view_async_tmem_load();

                // Get rowwise max
                float cur_max = -INFINITY;
                CUTE_UNROLL
                for (int i = 0; i < B_TOPK/2; ++i) {
                    if (!plan.is_token_valid[buf_idx][(idx_in_warpgroup/64)*(B_TOPK/2)+i]) p[i] = -INFINITY;
                    cur_max = max(cur_max, p[i]);
                }
                cur_max *= params.scale_softmax_log2;
                
                NamedBarrier::arrive_and_wait(128, 0);  // TODO Name these barriers
                plan.rowwise_max_buf[idx_in_warpgroup] = cur_max;
                NamedBarrier::arrive_and_wait(128, 0);
                cur_max = max(cur_max, plan.rowwise_max_buf[idx_in_warpgroup ^ 64]);

                float new_max = max(mi, cur_max);
                float scale_for_old = exp2f(mi - new_max);
                float2 scale_for_old_float2 = {scale_for_old, scale_for_old};

                // Get S
                float2 scale_softmax_log2_float2 = {params.scale_softmax_log2, params.scale_softmax_log2};
                float2 neg_new_max_float2 = {-new_max, -new_max};
                bf16 s[B_TOPK/2];
                float2 cur_sum = {0.0f, 0.0f};
                CUTE_UNROLL
                for (int i = 0; i < (B_TOPK/2)/2; ++i) {
                    float2 t = float2_fma(p_float2[i], scale_softmax_log2_float2, neg_new_max_float2);
                    t.x = exp2(t.x);
                    t.y = exp2(t.y);
                    *(__nv_bfloat162*)&s[i*2] = __float22bfloat162_rn(t);
                    cur_sum = float2_add(cur_sum, t);
                }

                // Save S
                // NOTE We don't need a barrier here, since the current QK^T has finished implies that the previous SV has finished
                bf16* sS_base = plan.s.data() + (idx_in_warpgroup/64)*(B_H*B_TOPK/2) + (idx_in_warpgroup%64) * 8;
                CUTE_UNROLL
                for (int i = 0; i < (B_TOPK/2)/8; i += 1) {
                    store_128b(sS_base + i*8*B_H, *((bf16x8*)s + i));
                }
                fence_view_async_shared();

                // Rescale O
                if (block_idx != start_block_idx) {
                    constexpr int B_SCALE_O = 64;
                    float2 o[B_SCALE_O/2];
                    CUTE_UNROLL
                    for (int b = 0; b < (D_V/2)/B_SCALE_O; ++b) {
                        tmem_ld_32dp32bNx<B_SCALE_O>(tmem_addr::o + b*B_SCALE_O, o);
                        cutlass::arch::fence_view_async_tmem_load();
                        CUTE_UNROLL
                        for (int i = 0; i < B_SCALE_O/2; ++i)
                            o[i] = float2_mul(o[i], scale_for_old_float2);
                        tmem_st_32dp32bNx<B_SCALE_O>(tmem_addr::o + b*B_SCALE_O, o);
                        cutlass::arch::fence_view_async_tmem_store();
                    }
                }
                plan.bar_so_ready[buf_idx].arrive();

                // Update mi and li
                mi = new_max;
                li = li * scale_for_old + cur_sum.x + cur_sum.y;

                bar_phase_k ^= 1<<buf_idx;
            }

            // Epilogue

            // Deal with no valid token cases
            if (mi == MAX_INIT_VAL) {
                mi = -INFINITY;
                li = 0.0f;
            }

            // Reduce li
            plan.rowwise_li_buf[idx_in_warpgroup] = li;
            NamedBarrier::arrive_and_wait(128, 0);
            li += plan.rowwise_li_buf[idx_in_warpgroup ^ 64];

            // Save li
            int num_valid_heads = min(B_H, params.q_head_per_hk - head_block_idx*B_H);
            int start_seq_idx = s_q_idx*params.q_head_per_hk + head_block_idx*B_H;
            int n_split_idx = batch_idx == begin_idx ? begin_n_split_idx : 0;
            int split_idx = is_no_split ? 0 : (__ldg(params.num_splits_ptr+batch_idx) + n_split_idx);
            if (idx_in_warpgroup < num_valid_heads) {
                if (is_no_split) {
                    float* gSoftmaxLse = (float*)params.softmax_lse_ptr + batch_idx*params.q_seq_per_hk + start_seq_idx + idx_in_warpgroup;
                    *gSoftmaxLse = li == 0.0f ? INFINITY : logf(li) + mi / (float)M_LOG2E; // NOTE Follows Flash MLA's approach, which returns +inf when there are no valid indices
                } else {
                    float* gSoftmaxLseAccum = (float*)params.softmax_lseaccum_ptr + split_idx*params.q_seq_per_hk + start_seq_idx + idx_in_warpgroup;
                    *gSoftmaxLseAccum = li == 0.0f ? -INFINITY : log2f(li) + mi;
                }
            }

            // Wait for the last SV gemm
            plan.bar_k_free[(end_block_idx-1)%NUM_BUFS].wait(bar_phase_k>>((end_block_idx-1)%NUM_BUFS)&1^1);
            tcgen05_after_thread_sync();

            // Save O
            float o_scale = li == 0.0f ? 0.0f : 1.0f / li;
            float2 o_scale_float2 = {o_scale, o_scale};
            if (is_no_split) {
                constexpr int B_EPI = 32;
                float2 o[B_EPI/2];
                __nv_bfloat162 o_bf16[B_EPI/2];
                Tensor sO = make_tensor(make_smem_ptr(plan.u.o_buf.data()), SmemLayoutOBuf{});
                bf16* sO_base = plan.u.o_buf.data() + ((idx_in_warpgroup/64)*128)*B_H + (idx_in_warpgroup%64)*8;
                CUTE_UNROLL
                for (int i = 0; i < (D_V/2) / B_EPI; ++i) {
                    // Load
                    tmem_ld_32dp32bNx<B_EPI>(tmem_addr::o + i*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    // Scale & Convert
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI/2; ++j) {
                        o[j] = float2_mul(o[j], o_scale_float2);
                        o_bf16[j] = __float22bfloat162_rn(o[j]);
                    }
                    // Store
                    int col_base = (i*B_EPI>=D_V/4 ? D_V/2 : 0) + (i*B_EPI%(D_V/4));
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI / 8; ++j)
                        store_128b(sO_base + (col_base+j*8)*B_H, *reinterpret_cast<bf16x8*>(&o_bf16[j*4]));
                }
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, 0);
                if (warp_idx == 4 && elect_one_sync()) {
                    Tensor tma_gO = tma_params.tma_O.get_tma_tensor(tma_params.shape_O)(_, _, s_q_idx, batch_idx);
                    auto thr_tma = tma_params.tma_O.get_slice(_0{});
                    Tensor my_tma_gO = flat_divide(tma_gO, Shape<Int<B_H>, Int<D_V>>{})(_, _, head_block_idx, _0{});
                    cute::copy(
                        tma_params.tma_O,
                        thr_tma.partition_S(sO),
                        thr_tma.partition_D(my_tma_gO)
                    );
                    cute::tma_store_arrive();
                }
            } else {
                constexpr int B_EPI = 64;
                float2 o[B_EPI/2];
                Tensor sO = make_tensor(make_smem_ptr(plan.u.o_accum_buf.data()), SmemLayoutOAccumBuf{});
                CUTE_UNROLL
                for (int i = 0; i < (D_V/2) / B_EPI; ++i) {
                    // Load
                    tmem_ld_32dp32bNx<B_EPI>(tmem_addr::o + i*B_EPI, o);
                    cutlass::arch::fence_view_async_tmem_load();
                    // Scale & Convert
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI/2; ++j)
                        o[j] = float2_mul(o[j], o_scale_float2);
                    // Store
                    int col_base = (idx_in_warpgroup/64)*128 + (i*B_EPI >= D_V/4 ? D_V/2 : 0) + (i*B_EPI%(D_V/4));
                    CUTE_UNROLL
                    for (int j = 0; j < B_EPI / 4; ++j)
                        store_128b(&sO(idx_in_warpgroup%64, col_base + j*4), *reinterpret_cast<float4*>(&o[j*2]));
                }
                fence_view_async_shared();
                NamedBarrier::arrive_and_wait(128, 0);
                if (elect_one_sync()) {
                    CUTE_UNROLL
                    for (int local_row = 0; local_row < B_H/4; ++local_row) {
                        int smem_row = local_row*4 + (warp_idx-4);
                        if (smem_row < num_valid_heads) {
                            SM90_BULK_COPY_S2G::copy(
                                &sO(smem_row, _0{}),
                                (float*)params.oaccum_ptr + (split_idx*params.q_seq_per_hk + start_seq_idx + smem_row)*D_V,
                                D_V*sizeof(float)
                            );
                        }
                    }
                    cute::tma_store_arrive();
                }
            }

            cute::tma_store_wait<0>();
        }

        if (warp_idx == 4) {
            cute::TMEM::Allocator1Sm().free(0, 512);
        }
    } else {
        cutlass::arch::warpgroup_reg_dealloc<96>();
        if (warp_idx == 8) {
            // UTCMMA warp

            bool bar_phase_q = 0;
            TiledMMA tiled_mma_qk = TiledMMA_QK{};
            TiledMMA tiled_mma_sv = TiledMMA_SV{};
            Tensor tP = partition_fragment_C(tiled_mma_qk, Shape<Int<B_H>, Int<B_TOPK>>{});
            Tensor tO = partition_fragment_C(tiled_mma_sv, Shape<Int<B_H>, Int<D_V>>{});
            tO.data().get() = tmem_addr::o;
            tP.data().get() = tmem_addr::p;
            Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
            
            #pragma unroll 1
            for (int batch_idx = begin_idx; batch_idx <= end_idx; ++batch_idx) {
                auto [start_block_idx, end_block_idx, is_no_split] = get_cur_req_info(batch_idx);

                if (elect_one_sync()) {
                    // Copy Q
                    Tensor gQ = flat_divide(
                        tma_params.tma_Q.get_tma_tensor(tma_params.shape_Q)(_, _, s_q_idx, batch_idx),
                        Tile<Int<B_H>, Int<D_K>>{}
                    )(_, _, head_block_idx, _0{});
                    launch_tma_copy(tma_params.tma_Q, gQ, sQ, plan.bar_q, TMA::CacheHintSm90::EVICT_FIRST);
                    plan.bar_q.arrive_and_expect_tx(B_H*D_K*sizeof(bf16));
                }

                NamedBarrier::arrive_and_wait(NUM_WORKING_THREADS, 1);

                if (elect_one_sync()) {
                    // Wait for Q
                    plan.bar_q.wait(bar_phase_q);
                    bar_phase_q ^= 1;
                    tcgen05_after_thread_sync();
                    
                    CUTE_NO_UNROLL
                    for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx++) {
                        int buf_idx = block_idx % NUM_BUFS;
                        
                        // Wait for K
                        plan.bar_k_ready[buf_idx].wait(bar_phase_k>>buf_idx&1);
                        tcgen05_after_thread_sync();
                        Tensor sK = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutK{});
                        
                        // Issue P = Q @ K^T
                        utcmma_ss(tiled_mma_qk, sQ, sK, tP, true);
                        umma_arrive_noelect(plan.bar_qk_done[buf_idx]);

                        // Wait for S
                        plan.bar_so_ready[buf_idx].wait(bar_phase_k>>buf_idx&1);
                        tcgen05_after_thread_sync();
                        Tensor sV = make_tensor(make_smem_ptr(plan.u.k[buf_idx].data()), SmemLayoutV{});

                        // Issue O += S @ V
                        utcmma_ss(tiled_mma_sv, sS, sV, tO, block_idx == start_block_idx);
                        umma_arrive_noelect(plan.bar_k_free[buf_idx]);

                        bar_phase_k ^= 1<<buf_idx;
                    }
                }
                __syncwarp();

                // NOTE If we reach this point, we must have done the QK gemm (since we've waited for bar_so_ready)
                // So we can launch the copy of the next Q block immediately
            }
        }
    }

#else
    if (cute::thread0()) {
        CUTE_INVALID_CONTROL_PATH("This kernel only supports sm100 ~ sm119");
    }
#endif
}

void run_flash_splitkv_mla_fp8_sparse_kernel(DecodingParams &params, cudaStream_t stream) {
    FLASH_ASSERT(params.h_k == 1);
    FLASH_ASSERT(params.topk % B_TOPK == 0);

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

    const int num_m_blocks = cute::ceil_div(params.q_head_per_hk, B_H);
    // NOTE Don't use PDL because of potential compiler bugs!
    mla_kernel<<<dim3(num_m_blocks, params.s_q, params.num_sm_parts), dim3(NUM_THREADS, 1, 1), smem_size, stream>>>(params, tma_params);
    CHECK_CUDA_KERNEL_LAUNCH();
}
    
}

#else // !SM100+ architecture

namespace sm100 {

void run_flash_splitkv_mla_fp8_sparse_kernel(DecodingParams &params, cudaStream_t stream) {
    throw std::runtime_error("FlashMLA sparse FP8 kernel requires SM100+ architecture. This build was compiled without SM100 support.");
}

}

#endif // SM100+ architecture check
