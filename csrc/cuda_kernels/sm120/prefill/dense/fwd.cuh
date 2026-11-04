// Dense prefill (FMHA): register mma.sync, segmented dataflow (see
// config.h). Reuses the sm120 sparse-prefill skeleton; differences:
//   - Q tile = 64 GLOBAL query rows (across several sequences) x 1 query head
//     (grid = (h_q, ceil(total_q/64))), K/V = the kv head of that head (GQA).
//   - k and v are independent tensors, so phase1 copies V STRAIGHT into the
//     dim-major (token-fast) mma-B layout -- no in-place K->V transpose needed
//     (but the layout must still be dim-major: the PV B operand's mma-K axis is
//     the token, opposite of QK where it is the dim).
//   - per-row masking: each row's valid kv window [beg,end) comes from its
//     sequence (binary search in cu_seqlens_q), plus a per-row causal upper
//     bound (causal = tril(diagonal = s_kv - s_q), like the torch reference).
//   - output o is stored directly to gmem per (row, head), lse as [total_q,h_q]
//     with head-major physical layout (element (row,head) at head*total_q+row,
//     i.e. the memory layout of torch.empty(h,total_q).T). No max_logits (not
//     in the sm100 FMHA return set).
// v0 sync model: plain __syncthreads() between phases (correctness first).
#include "fwd.h"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <cutlass/cluster_launch.hpp>

#include <kerutils/kerutils.cuh>

#include "cuda_kernels/legacy/defines.h"
#include "../../decode/sparse/components/dequant.h"   // load_128b_from_gmem
#include "config.h"

#include <climits>

namespace sm120::prefill::dense {

using namespace sm120::decode::sparse;    // L1CacheHint / L2PrefetchHint / load_128b_from_gmem

static constexpr float MAX_INIT_VAL = -1e30;    // Prevent (-inf) - (-inf) = nan

template<int D_QK>
__device__ __forceinline__ void KernelTemplate<D_QK>::devfunc(const DensePrefillParams &params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    constexpr int NUM_QK_SEG_ = KernelTemplate<D_QK>::NUM_QK_SEG;
    constexpr int HEAD_DIM_V_ = KernelTemplate<D_QK>::HEAD_DIM_V;

    const int head_idx = blockIdx.x;
    const int q_tile = blockIdx.y;
    const int q_global0 = q_tile * BLOCK_M;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sQ = make_tensor(make_smem_ptr(plan.q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(plan.k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(plan.v.data()), SmemLayoutV{});
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});

    // GQA: query head head_idx -> kv head.
    const int kv_head = head_idx * params.h_kv / params.h_q;
    // 64-bit row offsets: total_q * stride_q_row can exceed int32 for large tensors.
    const int64_t STRIDE_Q_ROW = (int64_t)params.h_q * params.d_qk;
    const int64_t STRIDE_KV_ROW = (int64_t)params.h_kv * params.d_qk;
    const int64_t STRIDE_V_ROW = (int64_t)params.h_kv * HEAD_DIM_V_;

    // No row-base clamp: q_base = q_global0 (grid is exactly ceil(total_q/64),
    // so q_base always names allocated rows). A tail tile's rows past total_q
    // have an all-false mask (sSeq = -1) and are never written; their Q read is
    // made safe by phase1's per-row clamp to sQClamp[r] = total_q-1.
    const int q_base = q_global0;

    TiledMMA_QK tiled_mma_QK{};
    auto thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
    TiledMMA_PV tiled_mma_PV{};
    auto thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

    float rL[2], rM[2];
    // PV output: 2 independent 64-col C fragments (HEAD_DIM_V = 128).
    using ROType = decltype(partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}));
    ROType rO[2] = {
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
    };
    Tensor rP = partition_fragment_C(tiled_mma_QK, Shape<Int<BLOCK_M>, Int<KV_BLOCK_SIZE>>{});
    Tensor rS = make_tensor<bf16>(partition_shape_A(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<KV_BLOCK_SIZE>>{}));

    rL[0] = rL[1] = 0.0f;
    rM[0] = rM[1] = MAX_INIT_VAL;
    CUTE_UNROLL
    for (int s = 0; s < 2; ++s)
        cute::fill(rO[s], 0.f);

    // -------- precompute: per-row sequence / kv window / causal bound --------
    // cu_seqlens_q/kv are sorted; seq = largest i with cu_q[i] <= qg.
    {
        const int b = params.b;
        const int32_t* cu_q = params.cu_seqlens_q;
        const int32_t* cu_kv = params.cu_seqlens_kv;
        if (threadIdx.x < BLOCK_M) {
            const int r = threadIdx.x;
            const int qg = q_base + r;
            if (qg < params.total_q) {
                int lo = 0, hi = b;
                while (lo < hi) {
                    const int mid = (lo + hi + 1) >> 1;
                    if (__ldg(cu_q + mid) <= qg) lo = mid; else hi = mid - 1;
                }
                const int seq = lo;
                const int qb = __ldg(cu_q + seq);
                const int qe = __ldg(cu_q + seq + 1);
                const int kvb = __ldg(cu_kv + seq);
                const int kve = __ldg(cu_kv + seq + 1);
                plan.sSeq[r] = seq;
                plan.sKVBeg[r] = kvb;
                plan.sKVEnd[r] = kve;
                plan.sCausalMax[r] = kvb + (kve - kvb) - (qe - qb) + (qg - qb);
                plan.sQClamp[r] = qg;
            } else {
                plan.sSeq[r] = -1;
                plan.sKVBeg[r] = 0;
                plan.sKVEnd[r] = 0;
                plan.sCausalMax[r] = -1;
                plan.sQClamp[r] = 0;         // safe read row; mask all-false
            }
        }
        // union kv range of the valid rows (two-warp reduction), then 64-align
        if (threadIdx.x < BLOCK_M) {
            const int r = threadIdx.x;
            int lo = plan.sKVBeg[r], hi = plan.sKVEnd[r];
            if (plan.sSeq[r] < 0) { lo = INT_MAX; hi = -1; }
            const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
            CUTE_UNROLL
            for (int off = 16; off; off >>= 1) {
                lo = min(lo, __shfl_xor_sync(0xffffffff, lo, off));
                hi = max(hi, __shfl_xor_sync(0xffffffff, hi, off));
            }
            if (lane == 0) { plan.sUnionLoW[w] = lo; plan.sUnionHiW[w] = hi; }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            plan.sUnionLo = min(plan.sUnionLoW[0], plan.sUnionLoW[1]);
            plan.sUnionHi = max(plan.sUnionHiW[0], plan.sUnionHiW[1]);
        }
        __syncthreads();
    }

    const int block_begin = (plan.sUnionLo / KV_BLOCK_SIZE) * KV_BLOCK_SIZE;
    const int block_end = ((plan.sUnionHi + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE) * KV_BLOCK_SIZE;
    const int num_kv_blocks = max(0, (block_end - block_begin) / KV_BLOCK_SIZE);

    const float scale_softmax_log2 = params.softmax_scale * CUDART_L2E_F;
    const bool is_causal = params.mask_mode == 1;

    // -------- phase 1a: copy the Q tile once (BLOCK_M rows x D_QK, dim-fast) --------
    // Per-row clamp: a tail tile's rows past total_q read row total_q-1 instead
    // (their mask is all-false, so the value is never used). All 256 threads.
    {
        const int job = threadIdx.x;                 // 0..255
        constexpr int QJOBS = BLOCK_M * NUM_QK_SEG_; // 128 (d=128) / 192 (d=192)
        if (job < QJOBS) {
            const int row = job % BLOCK_M;
            const int seg = job / BLOCK_M;           // 64-dim segment
            const bf16* gQ_r = (const bf16*)params.q
                + (int64_t)plan.sQClamp[row]*STRIDE_Q_ROW
                + (int64_t)head_idx*params.d_qk + seg*D_QK_SEG;
            bf16* dst = plan.q.data() + row*HEAD_DIM_K + seg*D_QK_SEG;
            CUTE_UNROLL
            for (int i = 0; i < D_QK_SEG/8; ++i) {   // 64 dims -> 8 x 16B
                bf16x8 cur = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gQ_r + i*8);
                *(uint128_t*)(dst + i*8) = *(uint128_t*)&cur;
            }
        }
    }
    __syncthreads();

    CUTE_NO_UNROLL
    for (int block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
        const int kv_global0 = block_begin + block_idx*KV_BLOCK_SIZE;

        // -------- phase 1: copy K + V blocks (all 256 threads) --------
        {
            const int token = threadIdx.x & 63;
            const int tg = threadIdx.x >> 6;    // 0..3 -> one of the 4 dim strips per token
            const int KV_GLOBAL = kv_global0 + token;
            const bool oob = KV_GLOBAL >= params.total_kv;
            const int row = oob ? 0 : KV_GLOBAL;

            const bf16* gK_row = (const bf16*)params.k + (int64_t)row*STRIDE_KV_ROW + (int64_t)kv_head*params.d_qk;
            const bf16* gV_row = (const bf16*)params.v + (int64_t)row*STRIDE_V_ROW + (int64_t)kv_head*HEAD_DIM_V_;

            // K layout (SEG, TOKEN, COL64): element address = seg*SEG_STRIDE +
            // token*64 + col. The QK B operand's mma-K axis is the dim, so dim
            // must be the fast axis -> (token,dim) stride (64,1). 16B stores.
            constexpr int SEG_STRIDE = KV_BLOCK_SIZE * D_QK_SEG;
            auto sk_addr = [&](int dim) -> __int128_t* {
                return (__int128_t*)(plan.k.data() + (dim / D_QK_SEG)*SEG_STRIDE + token*D_QK_SEG + dim % D_QK_SEG);
            };

            // K: NUM_QK_SEG 64-dim segments (192 -> 6 16B loads/thread, 128 -> 4).
            CUTE_UNROLL
            for (int i = 0; i < NUM_QK_SEG_; ++i) {
                const int dim16 = (tg*NUM_QK_SEG_ + i)*16;
                CUTE_UNROLL
                for (int j = 0; j < 2; ++j) {
                    bf16x8 cur = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_row + dim16 + j*8);
                    if (oob) *(uint128_t*)(&cur) = uint128_t();
                    *sk_addr(dim16 + j*8) = *(__int128_t*)&cur;
                }
            }

            // V layout (SEG, COL64, TOKEN): element address = seg*SEG_STRIDE +
            // col*64 + token, i.e. token-fast. Unlike K, the PV B operand's
            // mma-K axis is the TOKEN (O = P @ V), so token must be the fast
            // axis (the dim-major layout decode/sparse produce via the in-place
            // K->V transpose). Each thread owns a fixed token and 4 dims here;
            // the 8-token run of one dim is a 16B token-fast smem store fed by
            // an 8-way (token-strided) gmem gather.
            {
                const int tg8 = threadIdx.x & 7;        // token octet 0..7
                const int token0 = tg8 * 8;             // first token of this 8-run
                const int dim_slot = threadIdx.x >> 3;  // 0..31
                const bf16* gV_base = (const bf16*)params.v
                    + (int64_t)(kv_global0 + token0)*STRIDE_V_ROW + (int64_t)kv_head*HEAD_DIM_V_;
                CUTE_UNROLL
                for (int q = 0; q < 4; ++q) {
                    const int dim = dim_slot + q*32;    // 0..127
                    // 8 2B gathers at token stride STRIDE_V_ROW; zero across the
                    // block tail (tokens past total_kv) like K's oob handling.
                    uint16_t vals[8];
                    CUTE_UNROLL
                    for (int t = 0; t < 8; ++t) {
                        const int g = kv_global0 + token0 + t;
                        vals[t] = g < params.total_kv
                            ? *(const uint16_t*)(gV_base + (int64_t)t*STRIDE_V_ROW + dim)
                            : 0;
                    }
                    bf16* dst = plan.v.data() + (dim / D_QK_SEG)*SEG_STRIDE + (dim % D_QK_SEG)*KV_BLOCK_SIZE + token0;
                    *(uint128_t*)dst = *(uint128_t*)vals;
                }
            }
        }
        __syncthreads();

        // -------- phase 2 (WG0): QK + mask/softmax -> rS/sS --------
        if (warpgroup_idx == 0) {
            cute::clear(rP);
            CUTE_NO_UNROLL
            for (int seg = 0; seg < NUM_QK_SEG_; ++seg) {
                // Q comes from smem (phase1 copied it, per-row clamped); (64,64)
                // per-segment local tile of SmemLayoutQ (STRIDE (D_QK,1)).
                Tensor sQseg = local_tile(sQ, Shape<Int<BLOCK_M>, Int<D_QK_SEG>>{}, make_coord(_0{}, seg));
                // (SEG,TOKEN,COL64) -> standalone (64,64) NKContig stride (64,1) per segment
                Tensor sKseg = make_tensor(
                    sK.data() + seg*D_QK_SEG*KV_BLOCK_SIZE,
                    Layout<Shape<Int<KV_BLOCK_SIZE>, Int<D_QK_SEG>>, Stride<Int<D_QK_SEG>, _1>>{});
                Tensor rQa = thr_mma_QK.partition_fragment_A(sQseg);
                Tensor rKb = thr_mma_QK.partition_fragment_B(sKseg);
                cute::copy(thr_mma_QK.partition_A(sQseg), rQa);
                cute::copy(thr_mma_QK.partition_B(sKseg), rKb);
                cute::gemm(tiled_mma_QK, rQa, rKb, rP);
            }

            // softmax on segment 0 (per-row kv window + causal); every segment
            // rescales its own 64 output cols.
            float scale_for_olds[2];
            CUTE_UNROLL
            for (int s = 0; s < 2; ++s)
                scale_softmax(rP, rS, rO[s], scale_softmax_log2, plan.sScale, rM, rL,
                              kv_global0, is_causal, plan.sKVBeg, plan.sKVEnd, plan.sCausalMax,
                              idx_in_warpgroup, scale_for_olds, s == 0);

            save_rS_to_sS(rS, sS, idx_in_warpgroup);
            fence_view_async_shared();
        }
        __syncthreads();

        // -------- phase 3: PV (WG0 -> V dims [0,64), WG1 -> [64,128)) --------
        if (warpgroup_idx == 0) {
            // gemm_ss: rS re-read from sS (both warpgroups go through sS, see decode).
            Tensor rSv = thr_mma_PV.partition_fragment_A(sS);
            cute::copy(thr_mma_PV.partition_A(sS), rSv);
            {
                // dim-major segment: (n-mode stride 64 = dim, k-mode stride 1 = token)
                Tensor sVseg = make_tensor(
                    sV.data() + 0*D_QK_SEG*KV_BLOCK_SIZE,
                    Layout<Shape<Int<KV_BLOCK_SIZE>, Int<D_QK_SEG>>, Stride<Int<D_QK_SEG>, _1>>{});
                Tensor rVb = thr_mma_PV.partition_fragment_B(sVseg);
                cute::copy(thr_mma_PV.partition_B(sVseg), rVb);
                cute::gemm(tiled_mma_PV, rSv, rVb, rO[0]);
            }
        } else {
            // scale historical O by this block's scale_for_old, then PV on seg1.
            float cur_scales[2];
            *(float2*)cur_scales = *(float2*)(plan.sScale + (idx_in_warpgroup/4)*2);
            CUTE_UNROLL
            for (int s = 0; s < 2; ++s)
                CUTE_UNROLL
                for (int g = 0; g < (int)size(rO[s]); ++g)
                    rO[s](g) *= cur_scales[(g%4)/2];

            Tensor rSv = thr_mma_PV.partition_fragment_A(sS);
            cute::copy(thr_mma_PV.partition_A(sS), rSv);
            {
                Tensor sVseg = make_tensor(
                    sV.data() + 1*D_QK_SEG*KV_BLOCK_SIZE,
                    Layout<Shape<Int<KV_BLOCK_SIZE>, Int<D_QK_SEG>>, Stride<Int<D_QK_SEG>, _1>>{});
                Tensor rVb = thr_mma_PV.partition_fragment_B(sVseg);
                cute::copy(thr_mma_PV.partition_B(sVseg), rVb);
                cute::gemm(tiled_mma_PV, rSv, rVb, rO[1]);
            }
        }
        __syncthreads();
    }

    // -------- epilogue --------
    if (warpgroup_idx == 0) {
        // combine L/M across the four threads owning the same row
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

        // final o_scales for THIS wg (published to WG1 through sOScale); no
        // attn_sink, so o_scale = 1/L (0 when the row attended nothing).
        int row0 = get_AorC_row_idx(0, idx_in_warpgroup);
        int row1 = get_AorC_row_idx(1, idx_in_warpgroup);
        float o_scales_wg0[2];
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
            int row = i == 0 ? row0 : row1;
            float cur_L = (i == 0 ? rL[0] : rL[1]);
            o_scales_wg0[i] = cur_L == 0.0f ? 0.0f : __fdividef(1.0f, cur_L);
            if (idx_in_warpgroup%4 == 0)
                plan.sOScale[row] = o_scales_wg0[i];
        }
    }
    __syncthreads();

    float o_scales[2];
    CUTE_UNROLL
    for (int i = 0; i < 2; ++i)
        o_scales[i] = plan.sOScale[get_AorC_row_idx(i, idx_in_warpgroup)];

    // direct bf16 scatter of O: [total_q, h_q, d_v] with row stride h_q*d_v;
    // this CTA owns (q_base..+64, head_idx). Only writes rows < total_q.
    {
        Tensor gO = make_tensor(
            make_gmem_ptr((bf16*)params.o + (int64_t)q_base*(int64_t)params.h_q*HEAD_DIM_V_ + (int64_t)head_idx*HEAD_DIM_V_),
            make_layout(
                Shape<Int<BLOCK_M>, Int<HEAD_DIM_V_>>{},
                make_stride(int64_t((int64_t)params.h_q*HEAD_DIM_V_), _1{})
            )
        );
        const int t = idx_in_warpgroup;
        const int base = 16*(t/32) + (t%32)/4;
        CUTE_UNROLL
        for (int g = 0; g < 32; ++g) {
            int lut = (g%4)/2, cp = (g%4)%2;
            int row = base + 8*lut;
            if (plan.sSeq[row] < 0) continue;   // row past total_q
            int col = warpgroup_idx*64 + 8*(g/4) + 2*(t%4) + cp;
            gO(row, col) = (bf16)(rO[warpgroup_idx](g) * o_scales[lut]);
        }
    }

    // lse (natural ln domain) as [total_q, h_q] with head-major physical layout:
    // element (row, head) lives at head*total_q + row, which is the memory
    // layout of torch.empty(h_q, total_q).T (the Python side builds lse that way
    // and sm100's CUTLASS FMHA writes the same head-major layout). rM/rL live in
    // log2 domain. WG0 only.
    // A row that attended nothing (empty causal window, e.g. mask diagonal<0)
    // gets L == 0; torch's logsumexp of an all -inf row is -inf, so write
    // -INFINITY (out is left at 0 for such rows).
    if (warpgroup_idx == 0) {
        float* gLse = (float*)params.lse + (int64_t)head_idx*params.total_q + q_base;
        int i = threadIdx.x;
        if (i < BLOCK_M && plan.sSeq[i] >= 0) {
            float cur_L = plan.sL[i];
            float cur_M = plan.sM[i];
            gLse[i] = cur_L == 0.0f ? -INFINITY : logf(cur_L) + cur_M*CUDART_LN2_F;
        }
    }
#endif
}

template<int D_QK>
__global__ void __launch_bounds__(KernelTemplate<D_QK>::NUM_THREADS, 1)
dense_prefill_fwd_kernel(__grid_constant__ const DensePrefillParams params) {
    KernelTemplate<D_QK>::devfunc(params);
}

template<int D_QK>
void KernelTemplate<D_QK>::run(const DensePrefillParams &params) {
    KU_ASSERT(params.h_q % params.h_kv == 0);
    KU_ASSERT(params.d_qk == HEAD_DIM_K);
    KU_ASSERT(params.d_v == HEAD_DIM_V);
    KU_ASSERT(params.total_q > 0 && params.total_kv >= 0);
    KU_ASSERT(params.b > 0);

    const int q_tiles = (params.total_q + BLOCK_M - 1) / BLOCK_M;
    dim3 grid_dim((unsigned)params.h_q, (unsigned)q_tiles, 1);
    auto kernel = &dense_prefill_fwd_kernel<D_QK>;
    {
        cudaError_t sattr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);
        if (sattr != cudaSuccess) {
            fprintf(stderr, "[sm120 dense prefill run] setattr failed: %s\n", cudaGetErrorString(sattr));
        }
    }
    kernel<<<grid_dim, NUM_THREADS, sizeof(SharedMemoryPlan), params.stream>>>(params);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[sm120 dense prefill run] launch failed: %s (grid=(%d,%d,%d) smem=%zu num_threads=%d)\n",
                cudaGetErrorString(err), grid_dim.x, grid_dim.y, grid_dim.z,
                (size_t)sizeof(SharedMemoryPlan), NUM_THREADS);
    }
}

template<int D_QK>
void run_dense_fwd_kernel(const DensePrefillParams &params) {
    KernelTemplate<D_QK>::run(params);
}

}
