#pragma once

#include "splitkv_mla.h"

#include <cuda_fp8.h>
#include <cstdio>
#include <math_constants.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>

#include <kerutils/kerutils.cuh>

#include "cuda_kernels/utils.h"
#include "components/dequant.h"
#include "config.h"

using namespace cute;

namespace sm120::decode::sparse {

static constexpr float MAX_INIT_VAL = -1e30;    // Prevent (-inf) - (-inf) = nan
using cutlass::arch::fence_view_async_shared;
using fp8_e8m0 = __nv_fp8_e8m0;

// Host-side parameters handed to the kernel through __grid_constant__: only the
// O store descriptor is needed (Q is read directly by the compute threads).
struct TMAParams { CUtensorMap tensor_map_O; };

struct MainloopArgs {
    int start_block_idx, end_block_idx;
    bool is_no_split;

    // The following fields are only valid for DeepSeek-V4
    int topk_length, extra_topk_length, num_orig_kv_blocks;
};

// ---------------------------------------------------------------------------
// Dequantize one topk block (64 tokens) of K from global memory into the
// shared K buffer, in K-major (dim-contiguous) format. All 256 threads take part;
// each thread owns one token and a 16-dim strip of that token's dims.
//
// V is never dequantized: in MLA decode the value operand of the PV gemm is the
// dequantized nope part of K (d_v == nope dims), read straight from the same
// buffer (token stride HEAD_DIM_K, dim stride 1) -- no copy, no extra smem.
// ---------------------------------------------------------------------------
template<ModelType MODEL_TYPE, int NUM_HEADS>
__device__ __forceinline__ void dequant_k_block(
    const SparseAttnDecodeParams &params,
    typename KernelTemplate<MODEL_TYPE, NUM_HEADS>::SharedMemoryPlan &plan,
    int batch_idx,
    int s_q_idx,
    int block_idx,
    const MainloopArgs &args,
    bool use_extra,           // V4 only: block lives in the extra KV cache
    [[maybe_unused]] bool is_first_extra_block
) {
    using Kernel = KernelTemplate<MODEL_TYPE, NUM_HEADS>;
    constexpr int HEAD_DIM_K = Kernel::HEAD_DIM_K;
    constexpr int HEAD_DIM_NOPE = Kernel::HEAD_DIM_NOPE;
    constexpr int HEAD_DIM_ROPE = Kernel::HEAD_DIM_ROPE;
    constexpr int NUM_SCALES = Kernel::NUM_SCALES;
    constexpr int QUANT_TILE_SIZE = Kernel::QUANT_TILE_SIZE;
    constexpr int TOPK_BLOCK_SIZE = Kernel::TOPK_BLOCK_SIZE;
    constexpr int D_QK_SEG = Kernel::D_QK_SEG;

    const int token = threadIdx.x & 63;
    const int tg = threadIdx.x >> 6;    // 0..3 -> one of the 4 dim strips handled per token

    int* gIndices = params.indices + batch_idx*params.stride_indices_b + s_q_idx*params.stride_indices_s_q;
    int* gExtraIndices = params.extra_indices + batch_idx*params.stride_extra_indices_b + s_q_idx*params.stride_extra_indices_s_q;

    int token_index;
    if constexpr (MODEL_TYPE == ModelType::V32) {
        token_index = __ldg(gIndices + block_idx*TOPK_BLOCK_SIZE + token);
    } else {
        if (!use_extra)
            token_index = __ldg(gIndices + block_idx*TOPK_BLOCK_SIZE + token);
        else
            token_index = __ldg(gExtraIndices + (block_idx-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE + token);
    }

    // For DeepSeek-V4, tokens beyond topk_length are invalid (and may be garbage)
    bool is_valid = token_index != -1;
    if constexpr (MODEL_TYPE == ModelType::V4) {
        int topk_length = use_extra ? args.extra_topk_length : args.topk_length;
        int rel_block_idx = use_extra ? block_idx - args.num_orig_kv_blocks : block_idx;
        is_valid = is_valid && (rel_block_idx*TOPK_BLOCK_SIZE + token < topk_length);
    }

    int page_block_size = use_extra ? params.extra_page_block_size : params.page_block_size;
    int64_t k_block_stride = use_extra ? params.stride_extra_kv_block : params.stride_kv_block;
    int64_t k_row_stride = use_extra ? params.stride_extra_kv_row : params.stride_kv_row;
    fp8* k_ptr = use_extra ? (fp8*)params.extra_kv : (fp8*)params.kv;

    // uint32 div/mod: when token_index == -1 (UINT_MAX), UINT_MAX % page_block_size
    // stays < page_block_size, so no illegal memory access can occur.
    int block_index = token_index == -1 ? 0 : (int)((uint32_t)token_index/(uint32_t)page_block_size);
    int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;

    // V4 note: a token's full region is 584B (asserted below), but the fp8-nope +
    // bf16-rope DATA row is only 576B wide; the 8B e8m0 scales for ALL tokens of
    // a block are clustered in a separate region right after the block's rows
    // (see gK_scales_base). params.stride_kv_row is the 584B token-region stride,
    // so for V4 the nope/rope row step is hardcoded (matches the DeepSeek-V4
    // format and the SM90 kernel).
    fp8* gK_base;
    if constexpr (MODEL_TYPE == ModelType::V4) {
        gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*(HEAD_DIM_NOPE + HEAD_DIM_ROPE*sizeof(bf16));
    } else {
        gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*k_row_stride;
    }

    bf16 scales[NUM_SCALES];
    if constexpr (MODEL_TYPE == ModelType::V32) {
        static_assert(NUM_SCALES == 4);
        float scales_float[NUM_SCALES];
        *(float4*)(scales_float) = load_128b_from_gmem<float4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>((float*)(gK_base+HEAD_DIM_NOPE));
        CUTE_UNROLL
        for (int i = 0; i < NUM_SCALES; ++i)
            scales[i] = (bf16)scales_float[i];
    } else {
        static_assert(NUM_SCALES == 8);
        fp8_e8m0* gK_scales_base = (fp8_e8m0*)(k_ptr + block_index*k_block_stride + page_block_size*(HEAD_DIM_NOPE+HEAD_DIM_ROPE*sizeof(bf16)) + rel_idx_in_block*NUM_SCALES*sizeof(fp8_e8m0));
        fp8_e8m0 scales_e8m0[NUM_SCALES];
        *(int64_t*)scales_e8m0 = __ldg((int64_t*)gK_scales_base);
        CUTE_UNROLL
        for (int i = 0; i < NUM_SCALES; i += 2) {
            *(__nv_bfloat162_raw*)(scales+i) = __nv_cvt_e8m0x2_to_bf162raw(*(__nv_fp8x2_storage_t*)(scales_e8m0+i));
        }
    }

    plan.is_kv_valid[token] = is_valid;
    if (!is_valid) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_SCALES; ++i)
            scales[i] = (bf16)0.0f;
    }

    // (SEG, TOKEN, COL64) layout: dim d -> seg = d/64, col-in-seg = d%64, element
    // address = seg*SEG_STRIDE + token*64 + col. 8-col int128 stores are 16B aligned.
    constexpr int SEG_STRIDE = TOPK_BLOCK_SIZE * D_QK_SEG;
    auto sk_addr = [&](int dim) -> __int128_t* {
        return (__int128_t*)(plan.u.k.data() + (dim / D_QK_SEG)*SEG_STRIDE + token*D_QK_SEG + dim % D_QK_SEG);
    };

    // ----- nope dims (fp8, dequant): 8 x (16 dims) per thread -----
    // Alignment: V32 rows are 656B and V4 data rows 576B, both 16B-aligned (and
    // the V4 block stride is a multiple of 576), so 16B loads are always safe.
    fp8* gK_nope = gK_base;
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
        int dim16 = (tg*8 + i)*16;
        if (dim16 >= HEAD_DIM_NOPE) break;
        fp8x16 cur = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(gK_nope + dim16);
        if (!is_valid) *(uint128_t*)(&cur) = uint128_t();
        bf16 scale = scales[dim16 / QUANT_TILE_SIZE];
        __nv_bfloat162 scale2 = __bfloat162bfloat162(*(__nv_bfloat16*)(&scale));
        bf16x8 lo = cvt_fp8x8_bf16x8(cur.lo, scale2);
        bf16x8 hi = cvt_fp8x8_bf16x8(cur.hi, scale2);
        *sk_addr(dim16)     = *(__int128_t*)&lo;
        *sk_addr(dim16 + 8) = *(__int128_t*)&hi;
    }

    // ----- rope dims (bf16, straight copy): 16 dims per thread -----
    {
        int dim16 = tg*16;   // 0..48 (HEAD_DIM_ROPE == 64, 4 threads x 16)
        bf16* gK_rope;
        if constexpr (MODEL_TYPE == ModelType::V32) {
            // layout: [512 nope fp8][16B fp32 scales][64 rope bf16]
            gK_rope = (bf16*)(gK_base + HEAD_DIM_NOPE + NUM_SCALES*sizeof(float));
        } else {
            // layout: [448 nope fp8][64 rope bf16], scales are NOT in the row
            gK_rope = (bf16*)(gK_base + HEAD_DIM_NOPE);
        }
        CUTE_UNROLL
        for (int j = 0; j < 2; ++j) {
            bf16x8 cur = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_rope + dim16 + j*8);
            if (!is_valid) *(uint128_t*)(&cur) = uint128_t();
            *sk_addr(HEAD_DIM_NOPE + dim16 + j*8) = *(__int128_t*)&cur;
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse MLA decode kernel, register mma.sync (no GMMA/tcgen05 on sm_120a).
// CTA = 2 warpgroups, 256 threads:
//   WG0: QK (Q re-read from gmem per 64-dim segment) -> mask/softmax -> rS/sS;
//        PV over V dims [0,256) with rS staying in registers (gemm_rs).
//   WG1: PV over V dims [256,512) with rS re-read from sS (gemm_ss).
// Single KV buffer (K dequantized dim-contiguous; V is K's nope part, no copy).
// v0 sync model: plain __syncthreads() between phases (correctness first; the
// gmem<->smem pipeline overlap / dual buffering comes with the perf pass).
// ---------------------------------------------------------------------------
template<ModelType MODEL_TYPE, int NUM_HEADS>
template<typename TMAParams>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnDecodeParams &params, const TMAParams &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int head_block_idx = NUM_M_BLOCKS == 1 ? 0 : blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sK = make_tensor(make_smem_ptr(plan.u.k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(plan.u.k.data()), SmemLayoutV{});   // K's nope dims, (8 SEG,TOPK,64)
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
    Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
    float* sM = plan.sM;
    float* sL = plan.sL;
    float* sScale = plan.sScale;

    // split-KV is optional (enable_split_kv only when topk+extra_topk>640,
    // like sm100; sm90 always split). Without it num_sm_parts stays 0 and no
    // tile_scheduler_metadata exists: each (head_block, s_q) CTA covers the full KV
    // range, so synthesize a full-range schedule instead of dereferencing nullptr.
    bool is_split = params.tile_scheduler_metadata_ptr != nullptr;
    DecodingSchedMeta sched_meta;
    if (is_split) {
        sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
        if (sched_meta.begin_req_idx >= params.b) return;
    } else {
        sched_meta = DecodingSchedMeta{};
        sched_meta.begin_req_idx = 0;
        sched_meta.end_req_idx = params.b - 1;
        sched_meta.begin_block_idx = 0;
        sched_meta.is_first_req_splitted = false;
        sched_meta.is_last_req_splitted = false;
    }

    auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
        MainloopArgs args;
        int total_topk_padded;
        if constexpr (MODEL_TYPE == ModelType::V32) {
            total_topk_padded = params.topk;
        } else {
            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            int orig_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
            int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
            total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
            args.topk_length = topk_length;
            args.extra_topk_length = extra_topk_length;
            args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
        }

        const int total_num_blocks = total_topk_padded / TOPK_BLOCK_SIZE;
        args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        args.end_block_idx = batch_idx == sched_meta.end_req_idx ? (is_split ? sched_meta.end_block_idx : total_num_blocks) : total_num_blocks;
        args.is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

        return args;
    };

    // Q tile of this CTA from global memory (re-read per block/segment; no smem Q)
    auto gQ_full = make_tensor(
        make_gmem_ptr((bf16*)params.q),
        make_layout(
            make_shape(params.h_q, params.d_qk, params.s_q, params.b),
            make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b)
        )
    );

    TiledMMA_QK tiled_mma_QK{};
    auto thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
    TiledMMA_PV_LocalP tiled_mma_PV{};
    auto thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

    float rL[2], rM[2];
    // PV output: 4 independent 64-col C fragments (one 64x64
    // C per V N-segment; a single (64,256) C cannot be N-sliced by local_tile).
    using ROType = decltype(partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}));
    ROType rO[4] = {
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
    };
    Tensor rP = partition_fragment_C(tiled_mma_QK, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
    Tensor rS = make_tensor<bf16>(partition_shape_A(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

    float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
    if (params.attn_sink != nullptr) {
        for (int i = 0; i < 2; ++i) {
            int head_idx = head_block_idx*BLOCK_M + get_AorC_row_idx(i, idx_in_warpgroup);
            rAttn_sink[i] = __ldg((float*)params.attn_sink + head_idx) * CUDART_L2E_F;
        }
    }

    #pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        MainloopArgs args = get_cur_req_info(batch_idx);

        rL[0] = rL[1] = 0.0f;
        rM[0] = rM[1] = MAX_INIT_VAL;
        CUTE_UNROLL
        for (int s = 0; s < 4; ++s)
            cute::fill(rO[s], 0.f);

        Tensor gQ = flat_divide(
            gQ_full(_, _, s_q_idx, batch_idx),
            Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
        )(_, _, head_block_idx, _0{});   // (BLOCK_M, HEAD_DIM_K)

        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
            // -------- phase 1: dequant K (all 256 threads) --------
            if constexpr (MODEL_TYPE == ModelType::V32) {
                dequant_k_block<MODEL_TYPE, NUM_HEADS>(params, plan, batch_idx, s_q_idx, block_idx, args, false, false);
            } else {
                if (block_idx < min(args.num_orig_kv_blocks, args.end_block_idx))
                    dequant_k_block<MODEL_TYPE, NUM_HEADS>(params, plan, batch_idx, s_q_idx, block_idx, args, false, false);
                else
                    dequant_k_block<MODEL_TYPE, NUM_HEADS>(params, plan, batch_idx, s_q_idx, block_idx, args, true, true);
            }
            __syncthreads();

            // -------- phase 2 (WG0): QK + mask/softmax -> rS/sS --------
            if (warpgroup_idx == 0) {
                cute::clear(rP);
                CUTE_NO_UNROLL
                for (int seg = 0; seg < NUM_QK_SEG; ++seg) {
                    Tensor gQseg = local_tile(gQ, Shape<Int<BLOCK_M>, _64>{}, make_coord(_0{}, seg));
                    // (SEG,TOKEN,COL64) -> standalone (64,64) NKContig stride (64,1) per segment
                    Tensor sKseg = make_tensor(
                        sK.data() + seg*D_QK_SEG*TOPK_BLOCK_SIZE,
                        Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>, Stride<Int<D_QK_SEG>, _1>>{});
                    Tensor rQa = thr_mma_QK.partition_fragment_A(gQseg);
                    Tensor rKb = thr_mma_QK.partition_fragment_B(sKseg);
                    cute::copy(thr_mma_QK.partition_A(gQseg), rQa);
                    cute::copy(thr_mma_QK.partition_B(sKseg), rKb);
                    cute::gemm(tiled_mma_QK, rQa, rKb, rP);
                }

                // softmax on segment 0; every segment rescales its own 64 cols
                float scale_for_olds[2];
                CUTE_UNROLL
                for (int s = 0; s < 4; ++s)
                    scale_softmax(rP, rS, rO[s], params.sm_scale_div_log2, sScale, rM, rL, plan.is_kv_valid, block_idx, idx_in_warpgroup, scale_for_olds, s == 0);

                save_rS_to_sS(rS, sS, idx_in_warpgroup);
                fence_view_async_shared();
                // Transpose V segs 0..7 to DIM-major (dim*64+token) so the PV B-partition's
                // K-axis (token) is the fast stride-1 axis. cute::partition_fragment_B
                // requires K-major B operands (verified: PV rO was garbage until transposed;
                // QK's B already has K=dim fast and works). In-place pairwise swap per seg.
                // Parallelized across the 128 WG0 threads. Visibility to both warpgroups
                // is established by the __syncthreads() right after this branch. (No barrier
                // inside this branch: that would be a divergent bar.sync -- WG1 never
                // reaches it. The full-block barrier after the branch is sufficient.)
                {
                    bf16* svbase = plan.u.k.data();
                    const int seg_elems = D_QK_SEG * TOPK_BLOCK_SIZE;
                    const int total = 8 * seg_elems;
                    for (int X = threadIdx.x; X < total; X += NUM_THREADS / 2) {
                        const int sg = X / seg_elems;
                        const int x = X - sg * seg_elems;
                        const int t = x / D_QK_SEG, d = x - t * D_QK_SEG;
                        const int y = d*TOPK_BLOCK_SIZE + t;
                        if (x < y) {
                            bf16 tmp = svbase[sg*seg_elems + x];
                            svbase[sg*seg_elems + x] = svbase[sg*seg_elems + y];
                            svbase[sg*seg_elems + y] = tmp;
                        }
                    }
                }
                fence_view_async_shared();
            }
            __syncthreads();

            // -------- phase 3: PV (each WG its half of V; V = K's nope, no dequant) --------
            if (warpgroup_idx == 0) {
                // gemm_ss: rS re-read from sS (layout-matched A fragment),
                // same as WG1. rS-in-registers (gemm_rs) cannot be used: scale_softmax
                // fills rS with C-fragment indices but cute::gemm decodes the A operand
                // with the MMA A-fragment layout, which pairs a different (row,token)
                // set; both warpgroups must go through sS.
                Tensor rSv = thr_mma_PV.partition_fragment_A(sS);
                cute::copy(thr_mma_PV.partition_A(sS), rSv);
                CUTE_NO_UNROLL
                for (int seg = 0; seg < 4; ++seg) {
                    Tensor sVseg = make_tensor(
                        sV.data() + seg*D_QK_SEG*TOPK_BLOCK_SIZE,
                        Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>, Stride<Int<D_QK_SEG>, _1>>{});
                    Tensor rVb = thr_mma_PV.partition_fragment_B(sVseg);
                    cute::copy(thr_mma_PV.partition_B(sVseg), rVb);
                    cute::gemm(tiled_mma_PV, rSv, rVb, rO[seg]);
                }
            } else {
                // scale historical O by this block's scale_for_old, then PV over V dims [256,512)
                float cur_scales[2];
                *(float2*)cur_scales = *(float2*)(sScale + (idx_in_warpgroup/4)*2);
                CUTE_UNROLL
                for (int s = 0; s < 4; ++s)
                    CUTE_UNROLL
                    for (int g = 0; g < (int)size(rO[s]); ++g)
                        rO[s](g) *= cur_scales[(g%4)/2];

                Tensor rSv = thr_mma_PV.partition_fragment_A(sS);
                cute::copy(thr_mma_PV.partition_A(sS), rSv);
                CUTE_NO_UNROLL
                for (int seg = 0; seg < 4; ++seg) {
                    Tensor sVseg = make_tensor(
                        sV.data() + (4 + seg)*D_QK_SEG*TOPK_BLOCK_SIZE,
                        Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>, Stride<Int<D_QK_SEG>, _1>>{});
                    Tensor rVb = thr_mma_PV.partition_fragment_B(sVseg);
                    cute::copy(thr_mma_PV.partition_B(sVseg), rVb);
                    cute::gemm(tiled_mma_PV, rSv, rVb, rO[seg]);
                }
            }
            __syncthreads();
        }

        // -------- epilogue (per batch) --------
        if (warpgroup_idx == 0) {
            // combine L/M across the two rows this thread owns
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

            // final o_scales for THIS wg (and published to WG1 through sOScale)
            int row0 = get_AorC_row_idx(0, idx_in_warpgroup);
            int row1 = get_AorC_row_idx(1, idx_in_warpgroup);
            float o_scales[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                int row = i == 0 ? row0 : row1;
                float cur_L = (i == 0 ? rL[0] : rL[1]);
                float cur_M = (i == 0 ? rM[0] : rM[1]);
                if (args.is_no_split) {
                    o_scales[i] = cur_L == 0.0f ? 0.0f : __fdividef(1.0f, cur_L + exp2f(rAttn_sink[i] - cur_M));
                } else {
                    o_scales[i] = cur_L == 0.0f ? 0.0f : __fdividef(1.0f, cur_L);
                }
                if (idx_in_warpgroup%4 == 0)
                    plan.sOScale[row] = o_scales[i];
            }
        }
        __syncthreads();

        // each wg finishes its own rO half with the final o_scales
        float o_scales[2];
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i)
            o_scales[i] = plan.sOScale[get_AorC_row_idx(i, idx_in_warpgroup)];

        int start_head_idx = head_block_idx*BLOCK_M;
        int num_valid_seq_q = min(params.h_q - start_head_idx, BLOCK_M);
        if (args.is_no_split) {
            bf16* o_ptr = (bf16*)params.out + batch_idx*params.stride_o_b + s_q_idx*params.stride_o_s_q + start_head_idx*params.stride_o_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_h_q, 1)
            Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                make_stride(params.stride_o_h_q, _1{})
            ));
            // stage own O half into oBuf (unions with k; K/V already dead)
            // cute::copy(rO, partition_fragment_C(sMySeg)) is broken here:
            // the target fragment lands in LOCAL (%STL) instead of shared memory,
            // so the copy silently writes nowhere and TMA would ship K remnants.
            // Write each (row,col) explicitly through the SW128 tensor instead.
            // Fragment map: 2 rows {base, base+8} per thread,
            // base = 16*(t/32)+(t%32)/4; col = 8*(g/4)+2*(t%4)+(g%4)%2; lut=(g%4)/2.
            const int t = idx_in_warpgroup;
            const int base = 16*(t/32) + (t%32)/4;
            CUTE_UNROLL
            for (int s = 0; s < 4; ++s) {
                Tensor sMySeg = local_tile(sOBuf, Shape<_64, _64>{}, make_coord(_0{}, warpgroup_idx*4 + s));
                CUTE_UNROLL
                for (int g = 0; g < 32; ++g) {
                    int lut = (g%4)/2, cp = (g%4)%2;
                    int row = base + 8*lut;
                    int col = 8*(g/4) + 2*(t%4) + cp;
                    sMySeg(row, col) = (bf16)(rO[s](g) * o_scales[lut]);
                }
            }
            fence_view_async_shared();
            __syncthreads();

            if (threadIdx.x == 0) {
                SM90_TMA_STORE_5D::copy(
                    &tma_params.tensor_map_O,
                    plan.u.oBuf.data(),
                    0, head_block_idx*64, 0,
                    s_q_idx, batch_idx
                );
                cute::tma_store_arrive();
            }

            if (warpgroup_idx == 0) {
                float* gSoftmaxLse = (float*)params.lse + batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q + start_head_idx;	// (BLOCK_M) : (1)
                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLse[i] = cur_L == 0.0f ? INFINITY : logf(cur_L) + sM[i] / (float)M_LOG2E;
                }
            }
            cute::tma_store_wait<0>();
        } else {
            int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
            int split_idx = __ldg(params.num_splits_ptr+batch_idx) + n_split_idx;
            float* oaccum_ptr = (float*)params.o_accum + split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + start_head_idx*params.stride_o_accum_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
            Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), make_layout(
                Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                make_stride(params.stride_o_accum_h_q, _1{})
            ));
            // direct scatter to the fp32 global accumulator (no oAccumBuf on sm120);
            // 4 N-segments: each C fragment is a 64-col block (row/col map per segment)
            CUTE_UNROLL
            for (int s = 0; s < 4; ++s) {
                CUTE_UNROLL
                for (int idx = 0; idx < size(rO[s]); idx += 2) {
                    int row = (idx_in_warpgroup/32)*16 + (idx_in_warpgroup%32/4) + (idx%4 >= 2 ? 8 : 0);
                    int col = warpgroup_idx*256 + s*64 + (idx_in_warpgroup%4)*2 + idx/4*8;
                    *(float2*)(&gOAccum(row, col)) = float2 {
                        rO[s](idx) * o_scales[idx%4>=2],
                        rO[s](idx+1) * o_scales[idx%4>=2],
                    };
                }
            }

            if (warpgroup_idx == 0) {
                float* gSoftmaxLseAccum = (float*)params.lse_accum + split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + start_head_idx;	// (BLOCK_M) : (1)
                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLseAccum[i] = cur_L == 0.0f ? -INFINITY : log2f(cur_L) + sM[i];
                }
            }
        }
        __syncthreads();
    }
#endif
}

template<typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TMAParams tma_params) {
    Kernel::devfunc(params, tma_params);
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
    KU_ASSERT(params.d_qk == HEAD_DIM_K);
    KU_ASSERT(params.d_v == HEAD_DIM_V);
    KU_ASSERT(params.h_q % BLOCK_M == 0);
    if constexpr (MODEL_TYPE == ModelType::V4) {
        constexpr int BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE + 8;
        KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN, "Each page block in KV cache must be contiguous for head64 sparse fp8 decoding attention in DeepSeek-V4");  // Each block must be contiguous
        if (params.extra_kv != nullptr) {
            KU_ASSERT(params.stride_extra_kv_row == BYTES_PER_TOKEN, "Each page block in extra KV cache must be contiguous for head64 sparse fp8 decoding attention in DeepSeek-V4");  // Each block must be contiguous
        }
    } else {
        KU_ASSERT(params.extra_kv == nullptr, "V3.2 does not support extra KV cache");
        KU_ASSERT(params.topk_length == nullptr, "V3.2 does not support dynamic topk length");
        KU_ASSERT(params.stride_kv_row == 656);  // number of bytes per token (512 fp8 + 4 float32 + 64 bfloat16)
    }

    // Only the O store descriptor is needed (no Q TMA on sm120: Q is read directly
    // from global memory by the compute threads).
    CUtensorMap tensor_map_O;
    {
        // 5D TMA to store O
        uint64_t size[5] = {OBUF_SW, (unsigned long)params.h_q, HEAD_DIM_V/OBUF_SW, (unsigned long)params.s_q, (unsigned long)params.b};
        uint64_t stride[4] = {params.stride_o_h_q*sizeof(bf16), OBUF_SW*sizeof(bf16), params.stride_o_s_q*sizeof(bf16), params.stride_o_b*sizeof(bf16)};
        uint32_t box_size[5] = {OBUF_SW, BLOCK_M, HEAD_DIM_V/OBUF_SW, 1, 1};
        uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_O,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            5,
            params.out,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            OBUF_SW == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B :
                OBUF_SW == 32 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B :
                OBUF_SW == 16 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B :
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TMAParams tma_params{tensor_map_O};

    auto kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<MODEL_TYPE, NUM_HEADS>, TMAParams>;

    // opt-in the maximally allowed dynamic shared memory (101376B);
    // our plan (~83.5KB) fits within it.
    {
        cudaError_t sattr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);
        if (sattr != cudaSuccess) {
            fprintf(stderr, "[sm120 run] setattr failed: %s\n", cudaGetErrorString(sattr));
        }
    }

    // Non-split-KV leaves num_sm_parts == 0; clamp to 1 (one CTA covers the full
    // KV range, see devfunc's synthesized full-range schedule).
    int num_sm_parts = params.num_sm_parts > 0 ? params.num_sm_parts : 1;
    dim3 grid_dim(NUM_M_BLOCKS, params.s_q, num_sm_parts);
    kernel<<<grid_dim, NUM_THREADS, sizeof(SharedMemoryPlan), params.stream>>>(params, tma_params);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[sm120 run] launch failed: %s (grid=(%d,%d,%d) smem=%zu num_threads=%d)\n",
                cudaGetErrorString(err), grid_dim.x, grid_dim.y, grid_dim.z,
                (size_t)sizeof(SharedMemoryPlan), NUM_THREADS);
    }
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(params);
}

}
