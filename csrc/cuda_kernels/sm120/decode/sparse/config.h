#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
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
// 8-column pad per token (144 B token stride, bank rotation 4): the PV
// ldmatrix reads and the QK B copies are conflict free, and 16 B stores stay
// aligned because 144 is a multiple of 16.
static constexpr int K_ROW_PAD = 72;

// K/V shared layout: (SEG, TOKEN, COL64) -- each 64-col segment is a
// standalone (64,64) NKContig stride (64,1), the mma.sync B-operand layout
// verified on sm_120a. A single 576-stride K row does NOT
// work for B reads, so the producer
// scatters per-segment int128 stores (segment stride = 64*64, token stride 64,
// col stride 1). V is NOT stored separately: in MLA decode the shared-buffer
// value is the dequantized nope part of K, i.e. segments 0..7 of the K buffer.
using SmemLayoutK = Layout<
    Shape<Int<NUM_QK_SEG>, Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>,
    Stride<Int<TOPK_BLOCK_SIZE*K_ROW_PAD>, Int<K_ROW_PAD>, _1>
>;
using SmemLayoutV = Layout<
    Shape<Int<HEAD_DIM_V/D_QK_SEG>, Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>,
    Stride<Int<TOPK_BLOCK_SIZE*K_ROW_PAD>, Int<K_ROW_PAD>, _1>
>;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}
), Shape<_1, _1>{}));

// O staging buffer for the no-split TMA store: keep the 128B-swizzled GMMA layout
// so the 5D tensor_map_o descriptor built by the host works unchanged.
using SmemLayoutOBuf = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
    Step<_1, _2>{}
));

struct SharedMemoryPlan {
    CUTE_ALIGNAS(1024) union {
        array_aligned<bf16, cosize_v<SmemLayoutK>> k;        // K (then used as V, first 512 dims)
        array_aligned<bf16, cosize_v<SmemLayoutOBuf>> oBuf;  // O staging for no-split TMA store
    } u;
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
