#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm89.hpp>
#include <cute/atom/mma_traits_sm89.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/legacy/defines.h"
#include "params.h"

namespace sm120::decode::sparse {

using namespace cute;

template<ModelType MODEL_TYPE, int NUM_HEADS>
class KernelTemplate {
public:

static_assert(NUM_HEADS == 64 || NUM_HEADS == 128);
static constexpr int NUM_M_BLOCKS = NUM_HEADS / 64;
static constexpr int BLOCK_M = 64;
static constexpr int TOPK_BLOCK_SIZE = 64;
// 256-thread CTA = 2 warpgroups (no tcgen05, no wgmma on sm_120a):
//   WG0: QK (register mma.sync, Q re-read from gmem per segment) + mask/softmax
//        -> rS/sS, then PV over V dims [0,256).
//   WG1: PV over V dims [256,512) using sS from WG0.
// A 512-wide rO would need 256 fp32 regs per thread, so the PV output is split
// in half across warpgroups.
static constexpr int NUM_THREADS = 128*2;

static constexpr int OBUF_SW = 64;   // swizzle width of the O staging buffer (matches the TMA descriptor)

static constexpr int HEAD_DIM_K = MODEL_TYPE == ModelType::V32 ? 576 : 512;
static constexpr int HEAD_DIM_V = 512;
static constexpr int HEAD_DIM_ROPE = 64;
static constexpr int HEAD_DIM_NOPE = HEAD_DIM_K - HEAD_DIM_ROPE;

static constexpr int QUANT_TILE_SIZE = MODEL_TYPE == ModelType::V32 ? 128 : 64;
static constexpr int NUM_SCALES = MODEL_TYPE == ModelType::V32 ? 4 : 8;  // For DeepSeek-V4: 7 fp8_e4m3 + 1 padding

// QK runs in HEAD_DIM_K/64 segments of 64 dims (9 for V32 / 8 for V4):
// 64x64 gemm = 32 A + 32 B + 32 C regs/thread, no register spill.
static constexpr int D_QK_SEG = 64;
static constexpr int NUM_QK_SEG = HEAD_DIM_K / D_QK_SEG;
// K is resident as *raw cache bytes*: the nope dims are e4m3 (1 B) and the rope
// dims bf16 (2 B), exactly as they sit in global memory, so the copy never
// converts and the plane halves. A row is [nope fp8][rope bf16][pad]; the 16 B
// pad keeps the token stride a 16 B multiple (16 B cp.async destinations, 16 B
// ldmatrix rows) and rotates the banks.
static constexpr int K_ROW_BYTES = HEAD_DIM_NOPE + HEAD_DIM_ROPE*2;
static constexpr int K_ROW_PAD = K_ROW_BYTES + 16;
// Q is resident the same way, and for the same reason: the fp8 QK mma wants its
// A operand in e4m3, so Q is quantized per (row, 128-dim tile) once per block
// instead of per segment. Only the nope dims live here - the rope segment keeps
// the bf16 mma and reads its Q from gmem, so the plane is 512 B/row.
static constexpr int Q_ROW_BYTES = HEAD_DIM_NOPE;
static constexpr int Q_ROW_PAD = Q_ROW_BYTES + 16;
static constexpr int Q_BYTES = BLOCK_M * Q_ROW_PAD;
// byte offset of dim d inside its own row
static constexpr int k_dim_byte(int d) {
    return d < HEAD_DIM_NOPE ? d : HEAD_DIM_NOPE + (d - HEAD_DIM_NOPE)*2;
}
// The row is 640 B (V32) / 576 B (V4) of data + 16 B pad. A row is addressed
// whole; the 64-dim segments the gemms work in are column ranges inside it.
static constexpr int K_BYTES = TOPK_BLOCK_SIZE * K_ROW_PAD;
static constexpr int K_ROW_PAD_BF16 = K_ROW_PAD/2;   // bf16-element row stride

// Segments stay the gemm unit, but they are read as *bytes* out of the flat
// plane through k_dim_byte(); only a wholly-nope segment is a contiguous
// (TOPK, 64) byte tile, and those are the ones that build a SmemLayoutV segment.
//
// The V of the PV gemm is the nope part of this same buffer - no copy, no
// transpose, and no dequant until the fragment load.
using SmemLayoutV = Layout<
    Shape<Int<HEAD_DIM_V/D_QK_SEG>, Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>,
    Stride<Int<D_QK_SEG>, Int<K_ROW_PAD>, _1>
>;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}
), Shape<_1, _1>{}));

// O staging buffer for the no-split TMA store: keep the 128B-swizzled GMMA layout
// so the 5D tensor_map_o descriptor built by the host works unchanged.
// Half of O at a time (256 of the 512 V dims). A full-width staging buffer is
// 64 KB and the Q plane (33 KB, row 12) no longer fits beside it; the epilogue
// stores the two halves back to back out of the same buffer instead, which is
// free because it runs once per batch, not per page.
using SmemLayoutOBuf = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V/2>>{},
    Step<_1, _2>{}
));

struct SharedMemoryPlan {
    CUTE_ALIGNAS(1024) union {
        array_aligned<uint8_t, K_BYTES> k;                   // K (then used as V, first 512 dims)
        array_aligned<bf16, cosize_v<SmemLayoutOBuf>> oBuf;  // O staging for no-split TMA store
    } u;
    // Per-token dequant scales, so the fragment load can convert without the
    // producer having done it. NUM_SCALES per token, indexed dim/QUANT_TILE_SIZE.
    bf16 sScales[TOPK_BLOCK_SIZE * NUM_SCALES];
    // Q as e4m3, one row per mma row, quantized once per block. The rope dims
    // stay in gmem and keep the bf16 mma.
    CUTE_ALIGNAS(1024) array_aligned<uint8_t, Q_BYTES> q;
    // Per-(row, 128-dim tile) Q scales, mirrored to sScales' indexing.
    bf16 sQscales[BLOCK_M * NUM_SCALES];
    // WG1's QK partial, published for WG0 to add before the softmax. fp16: the
    // Q plane costs 33 KB, so the fp32 form (16 KB) no longer fits the 99 KB cap;
    // fp16 lands at 91 KB and keeps the 1.2e-4 rel gate, where bf16's 2e-4 does not.
    __half sP[BLOCK_M * TOPK_BLOCK_SIZE];
    CUTE_ALIGNAS(1024) array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    bool is_kv_valid[TOPK_BLOCK_SIZE];

    float sM[BLOCK_M], sL[BLOCK_M], sScale[BLOCK_M], sOScale[BLOCK_M];
};

// GMMA(wgmma) is unsupported on sm_120a; use register-file mma.sync.
// TMMA: SM80_16x8x16_F32BF16BF16F32_TN with AtomLayoutMNK (_4,_1,_1) -> 128 threads.
// Verified C fragment on sm_120a: 2 rows/thread {base, base+8}, base = 16*(t/32) +
// (t%32)/4, cols 8*(g/4) + 2*(t%4) + (g%4)%2. The A/B partitions are layout-matched
// to this fragment by construction.
using TMMA_SM120 = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{},
    Layout<Shape<_4, _1, _1>>{},
    Layout<Shape<_1, _1, _1>>{}
));

using TiledMMA_QK = TMMA_SM120;

// The nope dims run on the fp8 atom: e4m3 A and B, k = 32. Its B fragment is 2
// registers of 4 consecutive bytes each (k = 4*(t%4) + {0..3} and +16, see
// MMA_Traits<SM89_16x8x32_F32E4M3E4M3F32_TN>), i.e. exactly two 4-byte aligned
// reads out of the raw plane - no conversion and no scale in the inner loop. The
// C fragment is the same SM80_16x8_Row layout as TMMA_SM120, so rP, rS, the
// save_rS_to_sS map and scale_softmax are shared.
using TMMA_SM120_FP8 = decltype(make_tiled_mma(
    SM89_16x8x32_F32E4M3E4M3F32_TN{},
    Layout<Shape<_4, _1, _1>>{},
    Layout<Shape<_1, _1, _1>>{}
));
using TiledMMA_QK8 = TMMA_SM120_FP8;
using TiledMMA_PV_LocalP = TMMA_SM120;   // WG0: rS stays in registers (gemm_rs)
using TiledMMA_PV_RemoteP = TMMA_SM120;  // WG1: rS re-read from sS (gemm_ss)

// Same reg->(row,col) map as the old GMMA layout (homomorphic, verified on sm_120a).
static __forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
    return (idx_in_warpgroup/32)*16 + local_row_idx*8 + (idx_in_warpgroup%32/4);
}

// Save rS to sS. cute::copy(rS, partition_fragment_A(sS)) is broken here
// (the target fragment lands in LOCAL memory, like partition_fragment_C of the O
// buffer), so the shared buffer is never actually written (verified numerically:
// gemm_ss then reads all zeros). Scatter each C-layout register explicitly:
// rS(g) with g=4*nb+2*lut+cp holds P(row=base+8*lut, token=8*nb+2*(t%4)+cp);
// write it to sS(row, token), matching the SW128 A-layout reads of gemm_ss.
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
// to sScale. WG0 only. Layout-agnostic: iterates even-indexed regs (see phase1).
// Called once per PV C-segment (4x). do_softmax only on segment 0: computes the
// running max/softmax and fills scale_for_olds. Every call then rescales its rO
// segment by scale_for_olds, so all 256 output cols share the same running-max
// correction. (PV output is 4 independent 64-col C fragments, see splitkv_mla.cuh.)
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

template<typename TMAParams>
static __device__ __forceinline__ void
devfunc(const SparseAttnDecodeParams &params, const TMAParams &tma_params);

static void run(const SparseAttnDecodeParams &params);

};

};
