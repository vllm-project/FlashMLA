#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/legacy/defines.h"

namespace sm120::prefill::dense {

using namespace cute;

// Dense prefill (FMHA) kernel, register mma.sync (no GMMA/tcgen05 on
// sm_120a). Same segmented dataflow as sm120 prefill/sparse & decode/dense:
//   CTA = 256 threads / 2 warpgroups.
//   One CTA = 1 query head x 64 global query rows (possibly across several
//   sequences) x the union kv-block range of those rows.
//   grid = (h_q, ceil(total_q / BLOCK_M)).
//   WG0: per kv block QK (Q re-read from gmem per 64-dim segment, K from smem)
//        -> mask/softmax (per-row kv range + causal) -> rS/sS.
//   WG0/WG1: PV over V dims [0,64) / [64,128).
// Unlike decode/sparse, k and v are INDEPENDENT tensors (generic FMHA, not MLA
// merged cache), so V is copied straight into the mma-friendly (token,dim)
// stride-1 layout and there is NO in-place K->V transpose at all.
//   O accumulates in registers, stored directly to gmem (no TMA staging);
//   lse is stored as [total_q, h_q] with head-major physical layout (element
//   (row, head) at head*total_q + row) -- the memory layout of the .T view the
//   Python side produces (same head-major convention as the sm100 FMHA path).
template<int D_QK>
class KernelTemplate {
public:

static_assert(D_QK == 128 || D_QK == 192);
static constexpr int BLOCK_M = 64;
static constexpr int KV_BLOCK_SIZE = 64;
static constexpr int NUM_THREADS = 128*2;

static constexpr int HEAD_DIM_K = D_QK;
static constexpr int HEAD_DIM_V = 128;

static constexpr int D_QK_SEG = 64;
static constexpr int NUM_QK_SEG = HEAD_DIM_K / D_QK_SEG;   // 128:2, 192:3
static constexpr int NUM_V_SEG = HEAD_DIM_V / D_QK_SEG;    // 2

// Shared layouts. QK's B operand and PV's B operand are mma K stride-1 in
// OPPOSITE directions, because QK's mma-K is the dim (K[n_tok, k_dim]) while
// PV's mma-K is the token (O = P @ V -> V[n_dim, k_tok]):
//   K: (SEG, TOKEN, DIM64) stride (TOK*64, 64, 1)   -- dim fast
//   V: (SEG, DIM64, TOKEN) stride (DIM*64, 64, 1)   -- token fast (dim-major),
//      i.e. the same dim-major layout decode/sparse reach via the K->V
//      in-place transpose; here V is an independent tensor so phase1 writes it
//      directly in that layout.
using SmemLayoutK = Layout<
    Shape<Int<NUM_QK_SEG>, Int<KV_BLOCK_SIZE>, Int<D_QK_SEG>>,
    Stride<Int<KV_BLOCK_SIZE*D_QK_SEG>, Int<D_QK_SEG>, _1>
>;
using SmemLayoutV = Layout<
    Shape<Int<NUM_V_SEG>, Int<D_QK_SEG>, Int<KV_BLOCK_SIZE>>,
    Stride<Int<KV_BLOCK_SIZE*D_QK_SEG>, Int<KV_BLOCK_SIZE>, _1>
>;

// Q copied into smem by phase1 (safe per-row clamp there), dim-fast like gmem.
// (A operand of the QK mma; no transpose needed.)
using SmemLayoutQ = Layout<
    Shape<Int<BLOCK_M>, Int<HEAD_DIM_K>>,
    Stride<Int<HEAD_DIM_K>, _1>
>;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<BLOCK_M>, Int<KV_BLOCK_SIZE>>{}
), Shape<_1, _1>{}));

struct SharedMemoryPlan {
    CUTE_ALIGNAS(1024) array_aligned<bf16, cosize_v<SmemLayoutQ>> q;   // Q tile (BLOCK_M rows x D_QK), dim-fast
    CUTE_ALIGNAS(1024) array_aligned<bf16, cosize_v<SmemLayoutK>> k;   // K tile (64 tokens x D_QK)
    CUTE_ALIGNAS(1024) array_aligned<bf16, cosize_v<SmemLayoutV>> v;   // V tile (64 tokens x HEAD_DIM_V), separate tensor
    CUTE_ALIGNAS(1024) array_aligned<bf16, cosize_v<SmemLayoutS>> s;

    // per-row precompute (row = local row 0..63 within the CTA's 64-query tile)
    int sSeq[BLOCK_M];        // sequence id, -1 if the row is past total_q
    int sKVBeg[BLOCK_M];      // first global kv token of that seq
    int sKVEnd[BLOCK_M];      // one-past-last global kv token of that seq
    int sCausalMax[BLOCK_M];  // causal upper bound (global kv token) per row,
                              // = beg + (kvlen - q_len) + row_in_seq; only used
                              // when mask_mode == causal.
    int sQClamp[BLOCK_M];     // safe gmem q row to read: min(q_base + r,
                              // total_q - 1), so a tail tile's out-of-range rows
                              // (mask all-false anyway) never read out of bounds.

    // union kv range of the 64 rows (computed at CTA start), 64-aligned
    int sUnionLoW[2], sUnionHiW[2];   // per-warp partials (rows 0..31 / 32..63)
    int sUnionLo, sUnionHi;

    float sM[BLOCK_M], sL[BLOCK_M], sScale[BLOCK_M], sOScale[BLOCK_M];
};

using TMMA_SM120 = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{},
    Layout<Shape<_4, _1, _1>>{},
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMMA_QK = TMMA_SM120;
using TiledMMA_PV = TMMA_SM120;

// Same reg->(row,col) map as the decode/prefill family (homomorphic, verified).
static __forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
    return (idx_in_warpgroup/32)*16 + local_row_idx*8 + (idx_in_warpgroup%32/4);
}

// Save rS to sS (scatter by C-fragment indices; see decode config.h for why
// cute::copy(partition_fragment_A) is not used).
template<typename Tensor0, typename Tensor1>
static __forceinline__ __device__ void save_rS_to_sS(
    Tensor0 const &rS,
    Tensor1 const &sS,
    int idx_in_warpgroup
) {
    const int base = 16*(idx_in_warpgroup/32) + (idx_in_warpgroup%32)/4;
    CUTE_UNROLL
    for (int nb = 0; nb < 8; ++nb) {
        CUTE_UNROLL
        for (int lut = 0; lut < 2; ++lut) {
            CUTE_UNROLL
            for (int cp = 0; cp < 2; ++cp) {
                const int g = 4*nb + 2*lut + cp;
                const int row = base + 8*lut;
                const int token = 8*nb + 2*(idx_in_warpgroup%4) + cp;
                sS(row, token) = rS(g);
            }
        }
    }
}

// mask(scale(P)) -> S in registers; updates rM/rL; writes per-row scale_for_old
// to sScale. WG0 only. Per-row masking: unlike sparse (single seq per CTA), the
// 64 rows of this Q tile may belong to different sequences with different kv
// ranges, so validity is evaluated inline per element:
//   valid = beg <= tok < end  &&  (!causal || tok <= causal_max)
// where tok is the GLOBAL kv token index and beg/end/causal_max come from the
// per-row precompute.
template<typename Tensor0, typename Tensor1, typename Tensor2>
static __forceinline__ __device__ void scale_softmax(
    Tensor0 &rP,
    Tensor1 &rS,
    Tensor2 &rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    int kv_global0,            // first global kv token of this block
    bool is_causal,
    const int sKVBeg[],
    const int sKVEnd[],
    const int sCausalMax[],
    int idx_in_warpgroup,
    float scale_for_olds[2],
    bool do_softmax
) {
    if (do_softmax) {
        const int base = 16*(idx_in_warpgroup/32) + (idx_in_warpgroup%32)/4;
        const int qcol = 2*(idx_in_warpgroup%4);
        constexpr int P_COL_SLOTS = decltype(size(rP))::value/4;   // KV_BLOCK_SIZE/8 == 8
        CUTE_UNROLL
        for (int lut = 0; lut < 2; ++lut) {
            const int row = base + 8*lut;
            const int beg = sKVBeg[row], end = sKVEnd[row];
            const int cmax = is_causal ? sCausalMax[row] : INT_MAX;
            float cur_max = -INFINITY;
            CUTE_UNROLL
            for (int nb = 0; nb < P_COL_SLOTS; ++nb) {
                CUTE_UNROLL
                for (int cp = 0; cp < 2; ++cp) {
                    const int g = 4*nb + 2*lut + cp;
                    const int tok = kv_global0 + 8*nb + qcol + cp;
                    if (!(tok >= beg && tok < end && tok <= cmax))
                        rP(g) = -INFINITY;
                    cur_max = max(cur_max, rP(g));
                }
            }
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 1));
            cur_max = max(cur_max, __shfl_xor_sync(0xffffffff, cur_max, 2));

            cur_max *= scale_softmax_log2;
            float old_max = rM[lut];
            rM[lut] = max(cur_max, old_max);
            scale_for_olds[lut] = exp2f(old_max - rM[lut]);

            float cur_sum = 0;
            CUTE_UNROLL
            for (int nb = 0; nb < P_COL_SLOTS; ++nb) {
                CUTE_UNROLL
                for (int cp = 0; cp < 2; ++cp) {
                    const int g = 4*nb + 2*lut + cp;
                    const float p = exp2f(rP(g)*scale_softmax_log2 - rM[lut]);
                    rP(g) = p;
                    rS(g) = (bf16)p;
                    cur_sum += p;
                }
            }

            rL[lut] = rL[lut]*scale_for_olds[lut] + cur_sum;
        }
        if (idx_in_warpgroup%4 == 0)
            *(float2*)(sScale + 2*(idx_in_warpgroup/4)) = *(float2*)(scale_for_olds);
    }
    constexpr int O_COL_SLOTS = decltype(size(rO))::value/4;   // 64-col segment /8 == 8
    CUTE_UNROLL
    for (int lut = 0; lut < 2; ++lut) {
        CUTE_UNROLL
        for (int nb = 0; nb < O_COL_SLOTS; ++nb) {
            CUTE_UNROLL
            for (int cp = 0; cp < 2; ++cp) {
                rO(4*nb + 2*lut + cp) *= scale_for_olds[lut];
            }
        }
    }
}

static __device__ __forceinline__ void
devfunc(const DensePrefillParams &params);

static void run(const DensePrefillParams &params);

};

}
