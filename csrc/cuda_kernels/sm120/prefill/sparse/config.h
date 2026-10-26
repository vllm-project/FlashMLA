#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/legacy/defines.h"
#include "params.h"

namespace sm120::prefill::sparse {

using namespace cute;

// Sparse prefill kernel, register mma.sync (no GMMA/tcgen05 on sm_120a).
// Same segmented dataflow as the sm120 decode kernel (compare decode/sparse):
//   CTA = 256 threads / 2 warpgroups.
//   WG0: per kv block QK (Q re-read from gmem per 64-dim segment, K from smem)
//        -> mask/softmax -> rS/sS, then PV over V dims [0,256).
//   WG1: PV over V dims [256,512) using sS from WG0.
//   Single KV smem buffer: the K tile (B_TOPK x D_QK bf16, straight copy from
//   the raw bf16 kv cache -- the sparse prefill takes unquantized KV) is
//   transposed in place to dim-major before PV, exactly like decode.
//   O accumulates in registers and is stored directly to gmem (no TMA staging).
template<int D_QK>
class KernelTemplate {
public:

static_assert(D_QK == 576 || D_QK == 512);
static constexpr int BLOCK_M = 64;
static constexpr int TOPK_BLOCK_SIZE = 64;
static constexpr int NUM_THREADS = 128*2;

static constexpr int HEAD_DIM_K = D_QK;
static constexpr int HEAD_DIM_V = 512;
static constexpr int HEAD_DIM_ROPE = HEAD_DIM_K - HEAD_DIM_V;
static constexpr int HEAD_DIM_NOPE = HEAD_DIM_V;

static constexpr int D_QK_SEG = 64;
static constexpr int NUM_QK_SEG = HEAD_DIM_K / D_QK_SEG;

// Same shared layouts as the decode kernel: each 64-col segment is a standalone
// (64,64) NKContig stride (64,1); the mma.sync B operand of QK needs K (dim)
// stride-1. V is K's nope part (first 8 segments), transposed to dim-major
// (dim*64+token) before PV.
using SmemLayoutK = Layout<
    Shape<Int<NUM_QK_SEG>, Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>,
    Stride<Int<TOPK_BLOCK_SIZE*D_QK_SEG>, Int<D_QK_SEG>, _1>
>;
using SmemLayoutV = Layout<
    Shape<Int<HEAD_DIM_V/D_QK_SEG>, Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>,
    Stride<Int<TOPK_BLOCK_SIZE*D_QK_SEG>, Int<D_QK_SEG>, _1>
>;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}
), Shape<_1, _1>{}));

struct SharedMemoryPlan {
    CUTE_ALIGNAS(1024) array_aligned<bf16, cosize_v<SmemLayoutK>> k;   // KV tile (64 tokens x D_QK), then V
    CUTE_ALIGNAS(1024) array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    bool is_kv_valid[TOPK_BLOCK_SIZE];

    float sM[BLOCK_M], sL[BLOCK_M], sScale[BLOCK_M], sOScale[BLOCK_M];
};

using TMMA_SM120 = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{},
    Layout<Shape<_4, _1, _1>>{},
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMMA_QK = TMMA_SM120;
using TiledMMA_PV_LocalP = TMMA_SM120;   // WG0: rS stays in registers (gemm_rs)
using TiledMMA_PV_RemoteP = TMMA_SM120;  // WG1: rS re-read from sS (gemm_ss)

// Same reg->(row,col) map as the old GMMA layout (homomorphic, verified on sm_120a).
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
// to sScale. WG0 only. Layout-agnostic: iterates even-indexed regs (see decode).
// Called once per PV C-segment (4x). do_softmax only on segment 0: computes the
// running max/softmax and fills scale_for_olds. Every call then rescales its rO
// segment by scale_for_olds, so all 256 output cols share the same running-max
// correction.
template<typename Tensor0, typename Tensor1, typename Tensor2>
static __forceinline__ __device__ void scale_softmax(
    Tensor0 &rP,
    Tensor1 &rS,
    Tensor2 &rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    bool is_kv_valid[],
    int block_idx,
    int idx_in_warpgroup,
    float scale_for_olds[2],
    bool do_softmax
) {
    if (do_softmax) {
        const int base = 16*(idx_in_warpgroup/32) + (idx_in_warpgroup%32)/4;
        const int qcol = 2*(idx_in_warpgroup%4);
        constexpr int P_COL_SLOTS = decltype(size(rP))::value/4;   // TOPK_BLOCK_SIZE/8 == 8
        CUTE_UNROLL
        for (int lut = 0; lut < 2; ++lut) {
            float cur_max = -INFINITY;
            CUTE_UNROLL
            for (int nb = 0; nb < P_COL_SLOTS; ++nb) {
                CUTE_UNROLL
                for (int cp = 0; cp < 2; ++cp) {
                    const int g = 4*nb + 2*lut + cp;
                    if (!is_kv_valid[8*nb + qcol + cp])
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
devfunc(const SparseAttnFwdParams &params);

static void run(const SparseAttnFwdParams &params);

};

}
