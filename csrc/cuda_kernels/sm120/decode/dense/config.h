#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/legacy/defines.h"
#include "params.h"

namespace sm120::decode::dense {

using namespace cute;

// Padded per-token row stride (elements) for the K/V smem layout: 64 -> 72
// (144B) makes the m16n8k16 TN B-fragment reads fully conflict-free (bank =
// (4q+r)%32 covers all 32 banks; at stride 64 it was 8-way). K/V smem grows
// 73.7KB -> 82.9KB, still under the optin max (~99KB).
static constexpr int K_ROW_PAD = 72;

// mma.sync atom per InputT, shared by QK and PV. Both consume the element type
// InputT (raw KV for QK, softmax output P for PV -- upstream sm90 dense also
// keeps P in InputT), so fp16 requests must run through the fp16 atom; feeding
// them to the bf16 atom truncates the mantissa and exceeds the official
// tolerances. Built as a device-only function (not a class-member typedef)
// because the cute atom types are gated on the CUTE_ARCH_MMA_SM80 device-pass
// macro; a namespace/class-scope decltype would instantiate in the host pass
// where the fp16 atom is undefined.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
template<typename InputT>
static __forceinline__ __device__ auto make_input_tiled_mma() {
    if constexpr (std::is_same_v<InputT, cutlass::half_t>)
        return make_tiled_mma(
            SM80_16x8x16_F32F16F16F32_TN{},
            Layout<Shape<_4, _1, _1>>{},
            Layout<Shape<_1, _1, _1>>{});
    else
        return make_tiled_mma(
            SM80_16x8x16_F32BF16BF16F32_TN{},
            Layout<Shape<_4, _1, _1>>{},
            Layout<Shape<_1, _1, _1>>{});
}
#endif  // __CUDA_ARCH__ >= 1200

template<typename InputT, int HEAD_DIM_K>
class KernelTemplate {
public:

static_assert(HEAD_DIM_K == 512 || HEAD_DIM_K == 576);
// nvcc (CUDA 12.9) rejects odr-uses of class-template static constexpr int
// members in device code; runtime uses go through local constexpr aliases at
// the top of devfunc. Compile-time uses (Int<...>, layouts) are fine.
static constexpr int BLOCK_M = 64;
static constexpr int PAGE_BLOCK_SIZE = 64;
// 256-thread CTA = 2 warpgroups (no tcgen05, no wgmma on sm_120a):
//   WG0: QK (register mma.sync, Q re-read from gmem per segment) + mask/softmax
//        -> rS/sS, then PV over V dims [0,256).
//   WG1: PV over V dims [256,512) using sS from WG0.
// A 512-wide rO would need 256 fp32 regs per thread, so the PV output is split
// in half across warpgroups.
static constexpr int NUM_THREADS = 128*2;

static constexpr int HEAD_DIM_V = 512;
static constexpr int HEAD_DIM_ROPE = 64;
static constexpr int HEAD_DIM_NOPE = HEAD_DIM_K - HEAD_DIM_ROPE;

// QK runs in HEAD_DIM_K/64 segments of 64 dims (9 for V32 / 8 for V4):
// 64x64 gemm = 32 A + 32 B + 32 C regs/thread, no register spill.
static constexpr int D_QK_SEG = 64;
static constexpr int NUM_QK_SEG = HEAD_DIM_K / D_QK_SEG;

// K/V shared layout: (SEG, TOKEN, COL64) -- each 64-col segment is a standalone
// (64,64) contiguous tile, the mma.sync B-operand layout verified on sm_120a
// (a single HEAD_DIM_K-stride K row is wrong for B reads). The producer
// scatters per-segment int128 stores (segment stride 64*72, token stride
// K_ROW_PAD=72, col stride 1). V is the nope part of K, segments 0..7 of the K
// buffer: dense raw KV carries no RoPE section, and even for HEAD_DIM_K==576 V
// only reads the first 512 dims.
using SmemLayoutK = Layout<
    Shape<Int<NUM_QK_SEG>, Int<PAGE_BLOCK_SIZE>, Int<D_QK_SEG>>,
    Stride<Int<PAGE_BLOCK_SIZE*K_ROW_PAD>, Int<K_ROW_PAD>, _1>
>;
using SmemLayoutV = Layout<
    Shape<Int<HEAD_DIM_V/D_QK_SEG>, Int<PAGE_BLOCK_SIZE>, Int<D_QK_SEG>>,
    Stride<Int<PAGE_BLOCK_SIZE*K_ROW_PAD>, Int<K_ROW_PAD>, _1>
>;

using SmemLayoutS = decltype(coalesce(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<InputT>{},
    Shape<Int<BLOCK_M>, Int<PAGE_BLOCK_SIZE>>{}
), Shape<_1, _1>{}));

struct SharedMemoryPlan {
    CUTE_ALIGNAS(1024) array_aligned<InputT, cosize_v<SmemLayoutK>> k;   // K (first 512 dims reused as V)
    CUTE_ALIGNAS(1024) array_aligned<InputT, cosize_v<SmemLayoutS>> s;
    // WG1's QK partial tile (bf16): the K segments are split across the two
    // warpgroups, so WG1 publishes here and WG0 adds before the softmax.
    CUTE_ALIGNAS(1024) array_aligned<InputT, BLOCK_M*PAGE_BLOCK_SIZE> sP;   // P in InputT (softmax output)

    float sM[BLOCK_M], sL[BLOCK_M], sScale[BLOCK_M], sOScale[BLOCK_M];
};

// GMMA (wgmma) is unsupported on sm_120a; use register-file mma.sync.
// TMMA: SM80_16x8x16_F32*16F32_TN (atom per InputT, see make_input_tiled_mma)
// with AtomLayoutMNK (_4,_1,_1) -> 128 threads. C fragment on sm_120a is
// 2 rows/thread {base, base+8}, base = 16*(t/32) + (t%32)/4, cols 8*(g/4) +
// 2*(t%4) + (g%4)%2; A/B partitions are layout-matched to this fragment. QK and
// PV use the same atom; QK consumes raw InputT KV, PV consumes P in InputT.

// Same reg->(row,col) map as the old GMMA layout (homomorphic, verified on sm_120a).
static __forceinline__ __device__ int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
    return (idx_in_warpgroup/32)*16 + local_row_idx*8 + (idx_in_warpgroup%32/4);
}

// Save rS to sS. cute::copy(rS, partition_fragment_A(sS)) does not work here:
// the target fragment lands in LOCAL memory (like partition_fragment_C of the
// O buffer), so the shared buffer is never written (gemm_ss then reads zeros).
// Scatter each C-layout register explicitly: rS(g), g=4*nb+2*lut+cp, holds
// P(row=base+8*lut, token=8*nb+2*(t%4)+cp); write it to sS(row, token), matching
// the SW128 A-layout reads of gemm_ss.
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
// Causal mask: a token position (global within the q row, block_idx*64 + local)
// is masked (-inf) iff it is >= rRightBorder[lut]. rRightBorder is per row-lut:
//   causal:  min(seqlen_k - mask_len, boundary_of_iterated_blocks)
//   noncausal: seqlen_k.
// Tokens at positions >= seqlen_k that fall OUTSIDE rRightBorder are handled by
// zero-filling their K row at load time (sm90 semantics: OOB V contributes 0).
template<typename Tensor0, typename Tensor1, typename Tensor2>
static __forceinline__ __device__ void scale_softmax(
    Tensor0 &rP,
    Tensor1 &rS,
    Tensor2 &rO,
    float scale_softmax_log2,
    float sScale[],
    float rM[2],
    float rL[2],
    int rRightBorder[2],
    int block_idx,
    int idx_in_warpgroup,
    float scale_for_olds[2],
    bool do_softmax
) {
    if (do_softmax) {
        const int qcol = 2*(idx_in_warpgroup%4);
        constexpr int P_COL_SLOTS = decltype(size(rP))::value/4;   // PAGE_BLOCK_SIZE/8 == 8
        const int toff = block_idx*PAGE_BLOCK_SIZE;
        CUTE_UNROLL
        for (int lut = 0; lut < 2; ++lut) {
            float cur_max = -INFINITY;
            CUTE_UNROLL
            for (int nb = 0; nb < P_COL_SLOTS; ++nb) {
                CUTE_UNROLL
                for (int cp = 0; cp < 2; ++cp) {
                    const int g = 4*nb + 2*lut + cp;
                    if (toff + 8*nb + qcol + cp >= rRightBorder[lut])
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
                    rS(g) = (InputT)p;
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
devfunc(const DenseAttnDecodeParams &params);

static void run(const DenseAttnDecodeParams &params);

};

};
