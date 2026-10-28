// Sparse prefill: register mma.sync, segmented dataflow (see config.h).
// Ported from the sm120 sparse DECODE kernel: same 256-thread / 2-warpgroup
// structure, same register fragment math (verified on sm_120a), same
// in-place K->V transpose for the PV B operand. Differences vs decode:
//   - the query tile is B_H=64 DISTINCT (s_q, head) rows (grid covers h_q/s_q),
//   - the KV cache is raw bf16 (sparse prefill takes unquantized KV), straight
//     per-64-dim-segment copy into smem -- no fp8 dequant, no scales,
//   - output O (bf16) is stored directly to gmem (no TMA staging), plus
//     max_logits / lse in the natural (ln) domain.
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
#include "params.h"
#include "../../decode/sparse/components/dequant.h"   // load_128b_from_gmem
#include "config.h"

namespace sm120::prefill::sparse {

using namespace sm120::decode::sparse;    // L1CacheHint / L2PrefetchHint / load_128b_from_gmem

static constexpr float MAX_INIT_VAL = -1e30;    // Prevent (-inf) - (-inf) = nan

template<int D_QK>
__device__ __forceinline__ void KernelTemplate<D_QK>::devfunc(const SparseAttnFwdParams &params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    constexpr int HEAD_DIM_NOPE = KernelTemplate<D_QK>::HEAD_DIM_NOPE;
    constexpr int HEAD_DIM_ROPE = KernelTemplate<D_QK>::HEAD_DIM_ROPE;
    constexpr int NUM_QK_SEG_ = KernelTemplate<D_QK>::NUM_QK_SEG;

    const int s_q_idx = blockIdx.x / (params.h_q / BLOCK_M);
    const int head_block_idx = blockIdx.x % (params.h_q / BLOCK_M);
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sK = make_tensor(make_smem_ptr(plan.k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(plan.k.data()), SmemLayoutV{});   // K's nope dims, (8 SEG,TOPK,64)
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});

    // Q tile of this CTA from global memory (re-read per segment; no smem Q).
    // q: [s_q, h_q, d_qk] with strides (stride_q_s_q, stride_q_h_q, 1).
    // s_q can be huge (70000 in tests); the q tensor then spans ~9GB, so the
    // s_q-row offset (s_q_idx*stride_q_s_q) overflows int32. Strides must be int64.
    auto gQ_full = make_tensor(
        make_gmem_ptr((bf16*)params.q),
        make_layout(
            make_shape(params.s_q, params.h_q, params.d_qk),
            make_stride((int64_t)params.stride_q_s_q, (int64_t)params.stride_q_h_q, int64_t(1))
        )
    );
    // flat_divide((h_q,d_qk), Tile(64,576)) lays modes out as (64, 576, h_q/64, 1):
    // tiler modes FIRST, block counts LAST. So the head
    // block must be selected at coord 2 (the h_q/64 mode), exactly like decode.
    Tensor gQ = flat_divide(
        gQ_full(s_q_idx, _, _),
        Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
    )(_, _, head_block_idx, _0{});   // (BLOCK_M, HEAD_DIM_K)

    TiledMMA_QK tiled_mma_QK{};
    auto thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
    TiledMMA_PV_LocalP tiled_mma_PV{};
    auto thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

    // PV reads V DIM-major through ldmatrix.trans instead of transposing in
    // smem; U16x4 (x2) covers exactly one 8n x 16k mma B atom.
    auto ldsm_PV = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bf16>{}, tiled_mma_PV);
    auto thr_ldsm_PV = ldsm_PV.get_slice(idx_in_warpgroup);
    static constexpr auto kTransposedView =
        Layout<Shape<Int<D_QK_SEG>, Int<TOPK_BLOCK_SIZE>>, Stride<_1, Int<K_ROW_PAD>>>{};

    float rL[2], rM[2];
    // PV output: 4 independent 64-col C fragments (same as decode).
    using ROType = decltype(partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}));
    ROType rO[4] = {
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
    };
    Tensor rP = partition_fragment_C(tiled_mma_QK, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
    Tensor rS = make_tensor<bf16>(partition_shape_A(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

    // attn_sink in log2 domain, per global head index, else -inf (no sink).
    float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
    if (params.attn_sink != nullptr) {
        for (int i = 0; i < 2; ++i) {
            int head_idx = head_block_idx*BLOCK_M + get_AorC_row_idx(i, idx_in_warpgroup);
            rAttn_sink[i] = __ldg((float*)params.attn_sink + head_idx) * CUDART_L2E_F;
        }
    }

    int topk_cur = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    // sm90 semantics: ALWAYS iterate the full topk/64 blocks even with topk_length;
    // the per-token validity mask below (offs < topk_cur) is what shortens each row.
    // Truncating the loop to topk_cur/64 would drop the tail tokens (topk_length is
    // rarely a multiple of 64) and mismatch the reference.
    int num_kv_blocks = params.topk / TOPK_BLOCK_SIZE;
    const int* gIndices = params.indices + s_q_idx*params.stride_indices_s_q;

    rL[0] = rL[1] = 0.0f;
    rM[0] = rM[1] = MAX_INIT_VAL;
    CUTE_UNROLL
    for (int s = 0; s < 4; ++s)
        cute::fill(rO[s], 0.f);

    CUTE_NO_UNROLL
    for (int block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
        // -------- phase 1: copy K block (bf16, all 256 threads) --------
        {
            const int token = threadIdx.x & 63;
            const int tg = threadIdx.x >> 6;    // 0..3 -> one of the 4 dim strips per token
            int token_index = __ldg(gIndices + block_idx*TOPK_BLOCK_SIZE + token);
            // Mirrors sm90 phase1: valid iff 0 <= t < s_kv. OOB positions carry
            // sentinels (s_kv itself, huge ints, negatives) that must NOT become
            // gmem pointers -- clamp the read row to 0 for any invalid token.
            bool is_valid = token_index >= 0 && token_index < params.s_kv;
            if (params.topk_length != nullptr)
                is_valid = is_valid && (block_idx*TOPK_BLOCK_SIZE + token < topk_cur);
            plan.is_kv_valid[token] = is_valid;

            int row = is_valid ? token_index : 0;   // safe (masked) read
            const bf16* gK_row = (const bf16*)params.kv + (int64_t)row*params.stride_kv_s_kv;

            // (SEG, TOKEN, COL64) layout: dim d -> seg = d/64, col-in-seg = d%64,
            // element address = seg*SEG_STRIDE + token*64 + col. 16B stores.
            constexpr int SEG_STRIDE = TOPK_BLOCK_SIZE * K_ROW_PAD;
            auto sk_addr = [&](int dim) -> __int128_t* {
                return (__int128_t*)(plan.k.data() + (dim / D_QK_SEG)*SEG_STRIDE + token*K_ROW_PAD + dim % D_QK_SEG);
            };

            // bf16 straight copy: 16 dims per (i,j) step; tg covers every (NUM_QK_SEG*16)th dim.
            // Row stride is d_qk bf16 (contiguous, 16B aligned for 512/576), so 16B loads are safe.
            CUTE_UNROLL
            for (int i = 0; i < NUM_QK_SEG_; ++i) {
                int dim16 = (tg*NUM_QK_SEG_ + i)*16;
                CUTE_UNROLL
                for (int j = 0; j < 2; ++j) {
                    bf16x8 cur = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_row + dim16 + j*8);
                    if (!is_valid) *(uint128_t*)(&cur) = uint128_t();
                    *sk_addr(dim16 + j*8) = *(__int128_t*)&cur;
                }
            }
        }
        __syncthreads();

        // -------- phase 2 (WG0): QK + mask/softmax -> rS/sS --------
        if (warpgroup_idx == 0) {
            cute::clear(rP);
            CUTE_NO_UNROLL
            for (int seg = 0; seg < NUM_QK_SEG_; ++seg) {
                Tensor gQseg = local_tile(gQ, Shape<Int<BLOCK_M>, _64>{}, make_coord(_0{}, seg));
                // (SEG,TOKEN,COL64) -> standalone (64,64) NKContig stride (64,1) per segment
                Tensor sKseg = make_tensor(
                    sK.data() + seg*K_ROW_PAD*TOPK_BLOCK_SIZE,
                    Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>, Stride<Int<K_ROW_PAD>, _1>>{});
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
                scale_softmax(rP, rS, rO[s], params.sm_scale_div_log2, plan.sScale, rM, rL, plan.is_kv_valid, block_idx, idx_in_warpgroup, scale_for_olds, s == 0);

            save_rS_to_sS(rS, sS, idx_in_warpgroup);
            fence_view_async_shared();
        }
        __syncthreads();

        // -------- phase 3: PV (each WG its half of V; V = K's nope) --------
        if (warpgroup_idx == 0) {
            // gemm_ss: rS re-read from sS (both warpgroups go through sS, see decode).
            Tensor rSv = thr_mma_PV.partition_fragment_A(sS);
            cute::copy(thr_mma_PV.partition_A(sS), rSv);
            CUTE_NO_UNROLL
            for (int seg = 0; seg < 4; ++seg) {
                Tensor sVseg = make_tensor(
                    sV.data() + seg*K_ROW_PAD*TOPK_BLOCK_SIZE, kTransposedView);
                Tensor rVb = thr_mma_PV.partition_fragment_B(sVseg);
                cute::copy(ldsm_PV, thr_ldsm_PV.partition_S(sVseg),
                           thr_ldsm_PV.retile_D(rVb));
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
                    sV.data() + (4 + seg)*K_ROW_PAD*TOPK_BLOCK_SIZE, kTransposedView);
                Tensor rVb = thr_mma_PV.partition_fragment_B(sVseg);
                cute::copy(ldsm_PV, thr_ldsm_PV.partition_S(sVseg),
                           thr_ldsm_PV.retile_D(rVb));
                cute::gemm(tiled_mma_PV, rSv, rVb, rO[seg]);
            }
        }
        __syncthreads();
    }

    // -------- epilogue --------
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

        // final o_scales for THIS wg (and published to WG1 through sOScale)
        int row0 = get_AorC_row_idx(0, idx_in_warpgroup);
        int row1 = get_AorC_row_idx(1, idx_in_warpgroup);
        float o_scales_wg0[2];
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
            int row = i == 0 ? row0 : row1;
            float cur_L = (i == 0 ? rL[0] : rL[1]);
            float cur_M = (i == 0 ? rM[0] : rM[1]);
            o_scales_wg0[i] = cur_L == 0.0f ? 0.0f : __fdividef(1.0f, cur_L + exp2f(rAttn_sink[i] - cur_M));
            if (idx_in_warpgroup%4 == 0)
                plan.sOScale[row] = o_scales_wg0[i];
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

    // direct bf16 scatter of O (fragment map):
    // 2 rows {base, base+8} per thread, base = 16*(t/32)+(t%32)/4;
    // col = 8*(g/4)+2*(t%4)+(g%4)%2 within a 64-col segment.
    {
        // out: [s_q, h_q, d_v] contiguous (no per-tensor strides in SparseAttnFwdParams).
        // s_q stride is h_q*d_v -- NOT BLOCK_M*d_v (that would only match h_q==64).
        constexpr int STRIDE_O_H_Q = HEAD_DIM_V;
        const int STRIDE_O_S_Q = params.h_q * HEAD_DIM_V;
        // 64-bit offsets: s_q_idx*STRIDE_O_S_Q overflows int32 for large s_q (see gQ_note above).
        bf16* o_ptr = (bf16*)params.out + (int64_t)s_q_idx*STRIDE_O_S_Q + (int64_t)start_head_idx*STRIDE_O_H_Q;	// (BLOCK_M, HEAD_DIM_V) : (STRIDE_O_H_Q, 1)
        Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
            Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
            make_stride(STRIDE_O_H_Q, _1{})
        ));
        const int t = idx_in_warpgroup;
        const int base = 16*(t/32) + (t%32)/4;
        if (num_valid_seq_q == BLOCK_M || (base < num_valid_seq_q)) {
            CUTE_UNROLL
            for (int s = 0; s < 4; ++s) {
                CUTE_UNROLL
                for (int g = 0; g < 32; ++g) {
                    int lut = (g%4)/2, cp = (g%4)%2;
                    int row = base + 8*lut;
                    if (row >= num_valid_seq_q) continue;   // tail (h_q % 64 != 0) guard
                    int col = warpgroup_idx*256 + s*64 + 8*(g/4) + 2*(t%4) + cp;
                    gO(row, col) = (bf16)(rO[s](g) * o_scales[lut]);
                }
            }
        }
    }

    // max_logits & lse (natural ln domain); rM/rL live in log2 domain.
    if (warpgroup_idx == 0) {
        float* gMaxLogits = (float*)params.max_logits + (int64_t)s_q_idx*params.h_q + start_head_idx;   // (BLOCK_M) : (1)
        float* gSoftmaxLse = (float*)params.lse + (int64_t)s_q_idx*params.h_q + start_head_idx;         // (BLOCK_M) : (1)
        int i = threadIdx.x;
        if (i < num_valid_seq_q) {
            float cur_L = plan.sL[i];
            float cur_M = plan.sM[i];
            if (cur_L == 0.0f) {
                gMaxLogits[i] = -INFINITY;
                gSoftmaxLse[i] = INFINITY;
            } else {
                gMaxLogits[i] = cur_M * CUDART_LN2_F;
                gSoftmaxLse[i] = logf(cur_L) + cur_M * CUDART_LN2_F;
            }
        }
    }
#endif
}

template<int D_QK>
__global__ void __launch_bounds__(KernelTemplate<D_QK>::NUM_THREADS, 1)
sparse_attn_fwd_kernel(__grid_constant__ const SparseAttnFwdParams params) {
    KernelTemplate<D_QK>::devfunc(params);
}

template<int D_QK>
void KernelTemplate<D_QK>::run(const SparseAttnFwdParams &params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
    KU_ASSERT(params.d_qk == HEAD_DIM_K);
    KU_ASSERT(params.d_v == HEAD_DIM_V);
    KU_ASSERT(params.h_q % BLOCK_M == 0);

    const int num_head_blocks = params.h_q / BLOCK_M;
    dim3 grid_dim((long)num_head_blocks*params.s_q, 1, 1);
    auto kernel = &sparse_attn_fwd_kernel<D_QK>;
    {
        cudaError_t sattr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);
        if (sattr != cudaSuccess) {
            fprintf(stderr, "[sm120 prefill run] setattr failed: %s\n", cudaGetErrorString(sattr));
        }
    }
    kernel<<<grid_dim, NUM_THREADS, sizeof(SharedMemoryPlan), params.stream>>>(params);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[sm120 prefill run] launch failed: %s (grid=(%d,%d,%d) smem=%zu num_threads=%d)\n",
                cudaGetErrorString(err), grid_dim.x, grid_dim.y, grid_dim.z,
                (size_t)sizeof(SharedMemoryPlan), NUM_THREADS);
    }
}

template<int D_QK>
void run_sparse_fwd_kernel(const SparseAttnFwdParams &params) {
    KernelTemplate<D_QK>::run(params);
}

}
