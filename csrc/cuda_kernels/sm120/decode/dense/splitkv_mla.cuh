// Dense (raw bf16/fp16) MLA decode: register mma.sync, segmented dataflow.
// Ported from the sm120 sparse DECODE kernel: same 256-thread / 2-warpgroup
// structure, same register fragment math (verified on sm_120a), same
// in-place K->V transpose for the PV B operand. Differences vs sparse decode:
//   - no top-k: every request iterates its KV blocks [block_start, block_end)
//     from the paged block_table (raw KV, 64 tokens per page);
//   - the query tile is per-(batch,h_k): grid covers (m_block, h_k, sm_part),
//     and each CTA serially walks the batches in its partition, reloading Q per
//     batch (q_seq_per_hk is per-batch);
//   - causal masking via rRightBorder for the last KV block instead of a
//     per-token validity mask;
//   - output is bf16 (no_split) or fp32 oaccum (split), with ln-domain lse /
//     log2-domain lse_accum, no attn_sink, no scale_for_olds asymmetry.
// v0 sync model: plain __syncthreads() between phases (correctness first).
#include "splitkv_mla.h"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cluster_launch.hpp>

#include <kerutils/kerutils.cuh>

#include "cuda_kernels/legacy/defines.h"
#include "params.h"
#include "config.h"

#include "cuda_kernels/legacy/defines.h"
#include "params.h"
#include "config.h"

namespace sm120::decode::dense {

using namespace cute;

static constexpr float MAX_INIT_VAL = -1e30;    // Prevent (-inf) - (-inf) = nan

typedef unsigned int uint;

__forceinline__ __device__ int get_mask_len(
    const DenseAttnDecodeParams &params,
    int m_block_idx, int local_seq_q_idx,
    int block_m
) {
    // m_block_idx is per-batch (q_seq_per_hk rows are per (b, h_k)), so the
    // global q row within this batch is m_block_idx*BLOCK_M + local.
    int global_seq_q_idx = m_block_idx*block_m + local_seq_q_idx;
    if (global_seq_q_idx < params.q_seq_per_hk) {
        int s_q_idx = global_seq_q_idx / params.q_head_per_hk;
        return params.s_q - s_q_idx - 1;
    } else {
        // Out-of-bound request, regard as no masks
        return 0;
    }
}

template<typename InputT, int HEAD_DIM_K>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
__device__ __forceinline__
#endif
void KernelTemplate<InputT, HEAD_DIM_K>::devfunc(const DenseAttnDecodeParams &params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    using K = KernelTemplate<InputT, HEAD_DIM_K>;
    // [nvcc] class-template static constexpr members can't be odr-used in device
    // code; alias into local constexpr so runtime arithmetic below is plain ints
    // (sparse decode parity).
    constexpr int BLOCK_M = K::BLOCK_M;
    constexpr int PAGE_BLOCK_SIZE = K::PAGE_BLOCK_SIZE;
    constexpr int HEAD_DIM_V = K::HEAD_DIM_V;
    constexpr int NUM_QK_SEG_ = K::NUM_QK_SEG;
    constexpr int D_QK_SEG_ = K::D_QK_SEG;

    const int m_block_idx = blockIdx.x;
    const int k_head_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sK = make_tensor(make_smem_ptr(plan.k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(plan.k.data()), SmemLayoutV{});   // K's first 512 dims, (8 SEG,TOKEN,64)
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});

    // Q tile from global memory (re-read per segment; no smem Q).
    // gQ_full: [b, h_k, q_seq_per_hk, d] (per-batch rows); strides in elements.
    auto gQ_full = make_tensor(
        make_gmem_ptr((InputT*)params.q_ptr),
        make_layout(
            make_shape(params.q_seq_per_hk, params.d, params.h_k, params.b),
            make_stride(params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride)
        )
    );
    // Q tiles are built per-batch inside the request loop (q_seq_per_hk is
    // per-batch); see gQ_b below.

    auto tiled_mma_QK = make_input_tiled_mma<InputT>();
    auto thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
    auto tiled_mma_PV = make_input_tiled_mma<InputT>();   // WG0 gemm_rs + WG1 gemm_ss share the atom
    auto thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
    if (sched_meta.begin_req_idx >= params.b) return;
    // Zero-fill pad columns [64,72) of plan.k (K_ROW_PAD stride): any stray
    // PV/C fragment read of the padding then sees deterministic zeros instead
    // of uninitialized smem. Defensive; the dense path never writes these
    // columns, so this is a determinism guarantee, not a semantic one.
    {
        constexpr int PW = K_ROW_PAD - D_QK_SEG_;
        const int npad = NUM_QK_SEG_ * PAGE_BLOCK_SIZE * PW;
        unsigned short* kdu = (unsigned short*)(const void*)plan.k.data();
        for (int i = threadIdx.x; i < npad; i += (int)NUM_THREADS) {
            const int s = i / (PAGE_BLOCK_SIZE * PW);
            const int rem = i % (PAGE_BLOCK_SIZE * PW);
            const int t = rem / PW;
            const int c = D_QK_SEG_ + rem % PW;
            kdu[s*PAGE_BLOCK_SIZE*K_ROW_PAD + t*K_ROW_PAD + c] = 0;
        }
    }
    __syncthreads();

    const int* block_table = params.block_table + 0;   // + batch*block_table_batch_stride inside loop

    float rL[2], rM[2];
    using ROType = decltype(partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}));
    ROType rO[4] = {
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
    };
    Tensor rP = partition_fragment_C(tiled_mma_QK, Shape<Int<BLOCK_M>, Int<PAGE_BLOCK_SIZE>>{});
    Tensor rS = make_tensor<InputT>(partition_shape_A(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<PAGE_BLOCK_SIZE>>{}));

    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        int seqlen_k = __ldg(params.seqlens_k_ptr + batch_idx);
        int start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        int end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : cute::ceil_div(seqlen_k, PAGE_BLOCK_SIZE);
        bool is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

        // reload Q for this batch (per-batch rows)
        Tensor gQ_b = flat_divide(
            gQ_full(_, _, k_head_idx, batch_idx),
            Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
        )(_, _, m_block_idx, _0{});

        // Per-row right border of the visible KV prefix. Rows differ only in the
        // causal mask, so only the 2 rows each thread owns are tracked. We do NOT
        // implement the sm90 common-mask end_block shortening; the
        // full block range is always iterated and the mask trims the last block.
        int rRightBorder[2];
        if (params.is_causal) {
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                int row = get_AorC_row_idx(i, idx_in_warpgroup);
                int mask_len = get_mask_len(params, m_block_idx, row, BLOCK_M);
                rRightBorder[i] = min(seqlen_k - mask_len, end_block_idx*(int)PAGE_BLOCK_SIZE);
            }
        } else {
            rRightBorder[0] = rRightBorder[1] = seqlen_k;
        }

        rL[0] = rL[1] = 0.0f;
        rM[0] = rM[1] = MAX_INIT_VAL;
        CUTE_UNROLL
        for (int s = 0; s < 4; ++s)
            cute::fill(rO[s], 0.f);

        const int* block_table_b = block_table + batch_idx*params.block_table_batch_stride;

        CUTE_NO_UNROLL
        for (int block_idx = start_block_idx; block_idx < end_block_idx; ++block_idx) {
            // -------- phase 1: copy K page (raw InputT, all 256 threads) --------
            // Gmem-coalesced reload: each warp owns 8 contiguous in-page tokens
            // and streams their full rows (HEAD_DIM_K bf16) as 32x16B lanes ->
            // 512B/iteration, fully-coalesced 128B sectors. Smem destination
            // unchanged: (SEG,TOKEN,COL64), 16B stores at 8-bf16-aligned cols.
            {
                const int warp = threadIdx.x >> 5;          // 0..7
                const int lane = threadIdx.x & 31;
                const int page = __ldg(block_table_b + block_idx);
                // Page is OOB-guarded by the scheduler; kcache rows are [page][token][d]
                // with token the IN-PAGE position (0..63); the page covers one 64-token
                // block of the request. Validity is per GLOBAL token: the last page of
                // a request is partial (seqlen_k % 64 != 0), and rows beyond seqlen_k
                // must be zero-filled (their QK then has no effect and V contributes 0).
                // (Rows are read anyway -- the page's 64 slots are always allocated.)
                const int gtok_base = block_idx*PAGE_BLOCK_SIZE + warp*8;   // GLOBAL token (validity only)
                constexpr int GPR = (NUM_QK_SEG_ * D_QK_SEG_) / 8;          // 16B groups per row (72/64)
                constexpr int PASSES = GPR / 4;                             // 8 rows * GPR / 32 lanes (18/16)
                const InputT* gK_row_base = (const InputT*)params.k_ptr
                    + (int64_t)page*params.k_batch_stride + (int64_t)k_head_idx*params.k_head_stride;
                constexpr int SEG_STRIDE = PAGE_BLOCK_SIZE * K_ROW_PAD;
                CUTE_UNROLL
                for (int k = 0; k < PASSES; ++k) {
                    const int g = lane + k*32;              // contiguous group -> warp reads 512B/iter
                    const int tok = g / GPR;                // in-warp token 0..7 (PAGE-LOCAL row)
                    const int dim8 = (g % GPR) * 8;         // dim, 8-bf16 aligned
                    // gK_row_base already points at this page, so the row index is
                    // the PAGE-LOCAL token (warp*8+tok), same as the pre-COPY1 code;
                    // gtok (global) is used for row validity only.
                    const uint4 cur = *reinterpret_cast<const uint4*>(
                        gK_row_base + (int64_t)(warp*8 + tok)*params.k_row_stride + dim8);
                    const uint4 v = (gtok_base + tok < seqlen_k) ? cur : make_uint4(0, 0, 0, 0);
                    *(uint4*)(plan.k.data() + (dim8 / D_QK_SEG_)*SEG_STRIDE + (warp*8 + tok)*K_ROW_PAD + dim8 % D_QK_SEG_) = v;
                }
            }
            __syncthreads();

            // -------- phase 2 (WG0): QK + mask/softmax -> rS/sS --------
            if (warpgroup_idx == 0) {
                cute::clear(rP);
                CUTE_NO_UNROLL
                for (int seg = 0; seg < NUM_QK_SEG_; ++seg) {
                    Tensor gQseg = local_tile(gQ_b, Shape<Int<BLOCK_M>, _64>{}, make_coord(_0{}, seg));
                    // (SEG,TOKEN,COL64) -> standalone (64,64) NKContig stride (K_ROW_PAD,1) per segment
                    Tensor sKseg = make_tensor(
                        sK.data() + seg*K_ROW_PAD*PAGE_BLOCK_SIZE,
                        Layout<Shape<Int<PAGE_BLOCK_SIZE>, Int<D_QK_SEG_>>, Stride<Int<K_ROW_PAD>, _1>>{});
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
                    scale_softmax(rP, rS, rO[s], params.scale_softmax_log2, plan.sScale, rM, rL, rRightBorder, block_idx, idx_in_warpgroup, scale_for_olds, s == 0);

                save_rS_to_sS(rS, sS, idx_in_warpgroup);
                // In-place K->V DIM-major transpose (kept in the WG0 branch so it
                // cannot race this warpgroup's QK reads of plan.k; WG1 only reads
                // plan.k after the __syncthreads() below). Under the 72-stride
                // addressing, logical element (t,d) sits at t*72+d and its
                // transpose partner (d,t) at d*72+t in the same segment; after
                // the swap, sVs[tok*72+dim] == the original (dim,tok) element.
                {
                    InputT* svbase = plan.k.data();
                    constexpr int SEG_LOGICAL = D_QK_SEG_ * PAGE_BLOCK_SIZE;  // 4096 bf16 per seg
                    constexpr int TOTAL = NUM_QK_SEG_ * SEG_LOGICAL;          // all 8 segs
                    constexpr int SEG_BF16 = PAGE_BLOCK_SIZE * K_ROW_PAD;     // 4608 physical bf16 per seg
                    CUTE_NO_UNROLL
                    for (int X = threadIdx.x; X < TOTAL; X += NUM_THREADS / 2) {   // 128 WG0 threads, stride 128
                        const int sg = X / SEG_LOGICAL;
                        const int x = X - sg * SEG_LOGICAL;
                        const int t = x / D_QK_SEG_;
                        const int d = x - t * D_QK_SEG_;
                        const int base = sg * SEG_BF16;
                        const int a = base + t * K_ROW_PAD + d;
                        const int b = base + d * K_ROW_PAD + t;
                        if (a < b) {
                            InputT tmp = svbase[a];
                            svbase[a] = svbase[b];
                            svbase[b] = tmp;
                        }
                    }
                }
            }
            __syncthreads();

            // -------- phase 3: PV (each WG its half of V; V = K's first 512) --------
            if (warpgroup_idx == 0) {
                // gemm_ss: rS re-read from sS (both warpgroups go through sS, see decode).
                Tensor rSv = thr_mma_PV.partition_fragment_A(sS);
                cute::copy(thr_mma_PV.partition_A(sS), rSv);
                CUTE_NO_UNROLL
                for (int seg = 0; seg < 4; ++seg) {
                    Tensor sVseg = make_tensor(
                        sV.data() + seg*K_ROW_PAD*PAGE_BLOCK_SIZE,
                        Layout<Shape<Int<PAGE_BLOCK_SIZE>, Int<D_QK_SEG_>>, Stride<Int<K_ROW_PAD>, _1>>{});
                    Tensor rVb = thr_mma_PV.partition_fragment_B(sVseg);
                    // Same read as the QK B path (cute::copy).
                    cute::copy(thr_mma_PV.partition_B(sVseg), rVb);
                    cute::gemm(tiled_mma_PV, rSv, rVb, rO[seg]);
                }
            } else {
                // scale historical O by this block's scale_for_old, then PV over V dims [256,512)
                float cur_scales[2];
                *(float2*)cur_scales = *(float2*)(plan.sScale + (idx_in_warpgroup/4)*2);
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
                        sV.data() + (4 + seg)*K_ROW_PAD*PAGE_BLOCK_SIZE,
                        Layout<Shape<Int<PAGE_BLOCK_SIZE>, Int<D_QK_SEG_>>, Stride<Int<K_ROW_PAD>, _1>>{});
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
                    plan.sL[row] = rL[i];
                    plan.sM[row] = rM[i];
                }
            }

            // final o_scales for THIS wg (and published to WG1 through sOScale).
            // No attn_sink on dense: o_scale = 1/L. rL was pruned to 1.0 by the
            // caller-facing rule below (exp-underflow guard, sm90 parity).
            int row0 = get_AorC_row_idx(0, idx_in_warpgroup);
            int row1 = get_AorC_row_idx(1, idx_in_warpgroup);
            float o_scales_wg0[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                int row = i == 0 ? row0 : row1;
                float cur_L = (i == 0 ? rL[0] : rL[1]);
                // prune: rL may be 0.0f / NaN when large QK^T values underflow
                // exp2f; align with the old MLA kernel (sm90 parity). Must happen
                // HERE (WG0 only): WG1's rL is never touched by the softmax and
                // stays 0, so any per-wg recompute would break WG1's o_scales -
                // it gets the (pruned) 1/L through sOScale instead.
                float pruned_L = (cur_L == 0.0f || cur_L != cur_L) ? 1.0f : cur_L;
                o_scales_wg0[i] = __fdividef(1.0f, pruned_L);
                if (idx_in_warpgroup%4 == 0)
                    plan.sOScale[row] = o_scales_wg0[i];
            }
        }
        __syncthreads();

        // each wg finishes its own rO half with the final o_scales (already
        // pruned by WG0 before publishing through sOScale).
        float o_scales[2];
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i)
            o_scales[i] = plan.sOScale[get_AorC_row_idx(i, idx_in_warpgroup)];

        int num_valid_seq_q = min(params.q_seq_per_hk - m_block_idx*BLOCK_M, (int)BLOCK_M);
        if (is_no_split) {
            // out: [b, h_k, q_seq_per_hk, d_v] (per-batch rows; o_head_stride spans q_seq_per_hk).
            InputT* o_ptr = (InputT*)params.o_ptr
                + (int64_t)batch_idx*params.o_batch_stride
                + (int64_t)k_head_idx*params.o_head_stride
                + (int64_t)m_block_idx*BLOCK_M*params.o_row_stride;	// (BLOCK_M, HEAD_DIM_V) : (o_row_stride, 1)
            Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                make_stride(params.o_row_stride, _1{})
            ));
            // direct bf16/fp16 scatter; fragment map: 2 rows {base, base+8},
            // base = 16*(t/32)+(t%32)/4; col = 8*(g/4)+2*(t%4)+(g%4)%2 within a 64-col
            // segment; s = warpgroup half [0,256) or [256,512).
            const int t = idx_in_warpgroup;
            const int base = 16*(t/32) + (t%32)/4;
            if (num_valid_seq_q == BLOCK_M || (base < num_valid_seq_q)) {
                CUTE_UNROLL
                for (int s = 0; s < 4; ++s) {
                    CUTE_UNROLL
                    for (int g = 0; g < 32; ++g) {
                        int lut = (g%4)/2, cp = (g%4)%2;
                        int row = base + 8*lut;
                        if (row >= num_valid_seq_q) continue;   // tail (q_seq_per_hk % 64 != 0) guard
                        int col = warpgroup_idx*256 + s*64 + 8*(g/4) + 2*(t%4) + cp;
                        gO(row, col) = (InputT)(rO[s](g) * o_scales[lut]);
                    }
                }
            }

            if (warpgroup_idx == 0) {
                // softmax_lse: [b, h_k, q_seq_per_hk], ln domain
                float* gSoftmaxLse = (float*)params.softmax_lse_ptr
                    + ((int64_t)batch_idx*params.h_k + k_head_idx)*params.q_seq_per_hk
                    + m_block_idx*BLOCK_M;
                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = plan.sL[i];
                    gSoftmaxLse[i] = (cur_L == 0.0f || cur_L != cur_L) ? INFINITY
                        : (float)(log((double)cur_L) + (double)plan.sM[i] / M_LOG2E);
                }
            }
        } else {
            int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
            int split_idx = __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
            // oaccum: [total_num_splits, h_k, q_seq_per_hk, d_v] contiguous fp32
            float* oaccum_ptr = (float*)params.oaccum_ptr
                + ((int64_t)split_idx*params.h_k + k_head_idx)*params.q_seq_per_hk*HEAD_DIM_V
                + (int64_t)m_block_idx*BLOCK_M*HEAD_DIM_V;
            Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), Layout<
                Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>,
                Stride<Int<HEAD_DIM_V>, _1>
            >{});
            const int t = idx_in_warpgroup;
            const int base = 16*(t/32) + (t%32)/4;
            if (num_valid_seq_q == BLOCK_M || (base < num_valid_seq_q)) {
                CUTE_UNROLL
                for (int s = 0; s < 4; ++s) {
                    CUTE_UNROLL
                    for (int g = 0; g < 32; ++g) {
                        int lut = (g%4)/2, cp = (g%4)%2;
                        int row = base + 8*lut;
                        if (row >= num_valid_seq_q) continue;
                        int col = warpgroup_idx*256 + s*64 + 8*(g/4) + 2*(t%4) + cp;
                        gOAccum(row, col) = rO[s](g) * o_scales[lut];
                    }
                }
            }

            if (warpgroup_idx == 0) {
                // softmax_lseaccum: [total_num_splits, h_k, q_seq_per_hk], log2 domain
                float* gSoftmaxLseAccum = (float*)params.softmax_lseaccum_ptr
                    + ((int64_t)split_idx*params.h_k + k_head_idx)*params.q_seq_per_hk
                    + m_block_idx*BLOCK_M;
                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = plan.sL[i];
                    // [fast-math] log2f -> __log2f approx; keep split-combine lse exact
                    // (double) for the same reason as the no_split lse above.
                    gSoftmaxLseAccum[i] = (cur_L == 0.0f || cur_L != cur_L) ? -INFINITY
                        : (float)(log2((double)cur_L) + (double)plan.sM[i]);
                }
            }
        }
        __syncthreads();
    }
#endif
}

template<typename InputT, int HEAD_DIM_K>
__global__ void __launch_bounds__(KernelTemplate<InputT, HEAD_DIM_K>::NUM_THREADS, 1)
flash_fwd_splitkv_mla_kernel(__grid_constant__ const DenseAttnDecodeParams params) {
    KernelTemplate<InputT, HEAD_DIM_K>::devfunc(params);
}

template<typename InputT, int HEAD_DIM_K>
void KernelTemplate<InputT, HEAD_DIM_K>::run(const DenseAttnDecodeParams &params) {
    const int num_m_block = cute::ceil_div(params.q_seq_per_hk, BLOCK_M);
    dim3 grid_dim((long)num_m_block, params.h_k, params.num_sm_parts);
    auto kernel = &flash_fwd_splitkv_mla_kernel<InputT, HEAD_DIM_K>;
    {
        cudaError_t sattr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedMemoryPlan));
        if (sattr != cudaSuccess) {
            fprintf(stderr, "[sm120 dense decode run] setattr failed: %s\n", cudaGetErrorString(sattr));
        }
    }
    // Use cudaLaunchKernelEx to enable PDL (Programmatic Dependent Launch)
    // so the combine kernel's cudaGridDependencySynchronize() actually waits for
    // us. Plain <<<>>> launch leaves the dependency unregistered: combine then
    // races the split-KV accumulator writes, producing nondeterministic lse/out
    // errors on split requests (observed on varlen seqlen=1 cases). Matches the
    // sm90 dense decode launch (upstream PDL semantics).
    cudaLaunchAttribute kernel_attributes[1];
    kernel_attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    kernel_attributes[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t kernel_config = {
        grid_dim,
        dim3(NUM_THREADS, 1, 1),
        sizeof(SharedMemoryPlan),
        params.stream,
        kernel_attributes,
        1
    };
    cudaLaunchKernelEx(&kernel_config, kernel, params);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[sm120 dense decode run] launch failed: %s (grid=(%d,%d,%d) smem=%zu num_threads=%d)\n",
                cudaGetErrorString(err), grid_dim.x, grid_dim.y, grid_dim.z,
                (size_t)sizeof(SharedMemoryPlan), NUM_THREADS);
    }
}

template<typename InputT, int HEAD_DIM_K>
void run_flash_splitkv_mla_kernel(const DenseAttnDecodeParams &params) {
    KernelTemplate<InputT, HEAD_DIM_K>::run(params);
}

}
