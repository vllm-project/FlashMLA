#pragma once

#include <math_constants.h>
#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/arch.h>
#include <kerutils/kerutils.cuh>

#include "cuda_kernels/legacy/defines.h"
#include "params.h"
#include "cuda_kernels/kv_cache_format.h"

#include "kernel.h"

namespace sm120::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

using namespace cute;

// Fused "Q Norm + Q RoPE + Sparse Attention + O RoPE + O FP8 Cast" kernel
// (DeepSeek V4 / V4.1, d_qk = d_v = 512, h_kv = 1, exact top-k sparse attention),
// register mma.sync (no tcgen05/wgmma on sm_120a). Same segmented dataflow and
// layout as the sm120 sparse decode/prefill kernels (compare decode/sparse):
//   CTA = 256 threads / 2 warpgroups.
//   WG0: QK -> mask/softmax -> rS/sS, then PV over V dims [0,256).
//   WG1: PV over V dims [256,512) using sS from WG0.
//   Single KV smem buffer, K dequantized / copied dim-contiguous, transposed in
//   place to dim-major before PV.
// Differences vs the sparse core:
//   - Q comes from `permute_q_b_proj` (permuted layout, see api), which is not
//     the h-contiguous [s_q, h_q, d_qk] layout: element (h, d) of a token lives
//     at physical offset (d%16) + h*16 + (d/16)*h_q*16.
//   - Q rowwise RMSNorm (V4 only, enable_q_norm) + non-neox RoPE (last 64 dims)
//     are folded into the QK pipeline.
//   - O gets conjugate RoPE (last 64 dims) then an FP8 e4m3 cast with per-32
//     ue8m0 scales (rounded, packed 4-per-uint32) into the permuted out_fp8
//     layout expected by `permute_wv_proj`.
//   - Prefill: raw bf16 KV [s_kv, 1, d_qk] straight-copied (no dequant);
//     Decode: paged fp8/fp4 KV dequantized (V4 / V4.1, plus V4.1-fp4 extra).
template<Config CONFIG>
class Kernel {
public:

static constexpr SparseAttnFwdMode FWD_MODE = CONFIG.FWD_MODE;
static constexpr ModelType MODEL_TYPE = CONFIG.MODEL_TYPE;
static constexpr ModelType EXTRA_MODEL_TYPE = CONFIG.EXTRA_MODEL_TYPE;
static constexpr uint32_t H_Q = CONFIG.H_Q;
static constexpr bool ENABLE_Q_NORM = CONFIG.ENABLE_Q_NORM;

using Params = ParamT<FWD_MODE>;

static_assert(FWD_MODE == SparseAttnFwdMode::Prefill || FWD_MODE == SparseAttnFwdMode::Decode);
static_assert(H_Q == 64 || H_Q == 128);
static_assert(is_valid_kv_format_pair(MODEL_TYPE, EXTRA_MODEL_TYPE));

static constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;

// Model parameters
static constexpr uint32_t D_QK = 512;
static constexpr uint32_t D_VO = 512;
static constexpr uint32_t O_QUANT_TILE_SIZE = 32;   // fixed per-32 output scales
static constexpr uint32_t D_ROPE = 64;
static constexpr uint32_t D_NOPE = D_QK - D_ROPE;   // 448
static constexpr uint32_t WV_GROUP_SIZE = 8;

// 2-CTA clusters / tcgen05 multicast do not exist here; every CTA covers
// BLOCK_M = 64 Q rows (heads) and ONE wv_group (8 heads) is processed by a single
// CTA (grid.y covers wv groups when H_Q > 64's worth). All 512 V dims of a wv
// group live in one CTA, with WG0/WG1 splitting the 512 cols as in the sparse core.
static constexpr int BLOCK_M = 64;
static constexpr int TOPK_BLOCK_SIZE = 64;
static constexpr int NUM_THREADS = 128*2;

// QK runs in 8 x 64-dim segments; K/V shared layout (SEG, TOKEN, COL64), each
// 64-col segment a standalone (64,64) NKContig stride (64,1).
static constexpr int D_QK_SEG = 64;
static constexpr int NUM_QK_SEG = D_QK / D_QK_SEG;

// Paged quantized KV cache format for decoding + per-token byte strides.
template<ModelType MT>
struct KVFormat {
    using Base = KVCacheFormat<MT>;
    static constexpr bool IS_FP4 = Base::IS_FP4;
    static constexpr bool IS_INTERLEAVED = Base::IS_INTERLEAVED;
    static constexpr uint32_t QUANT_BYTES = Base::QUANT_BYTES;
    static constexpr uint32_t D_FP4 = Base::D_FP4;
    static constexpr uint32_t D_FP8 = Base::D_FP8;
    static constexpr uint32_t D_BF16 = Base::D_BF16;
    static constexpr uint32_t QUANT_TILE_SIZE = Base::QUANT_TILE_SIZE;
    static constexpr uint32_t NUM_SCALES_EACH_TOKEN = Base::NUM_SCALES_EACH_TOKEN;
    static constexpr uint32_t BYTES_PER_TOKEN = Base::BYTES_PER_TOKEN;
    static constexpr uint32_t D_QK = Base::D_QK;
    static constexpr uint32_t TMA_K_STRIDE = Base::TMA_K_STRIDE;   // bytes between two tokens' data rows
    static constexpr uint32_t STRIDE_K_ROW = TMA_K_STRIDE;         // alias kept for the sparse-decode naming
};
static constexpr bool HAS_FP4_KV = EXTRA_MODEL_TYPE == ModelType::V41_FP4;

// Shared layouts: K=dim-major for QK-B; V = K's nope part (first 8 segments),
// transposed to dim-major (dim*64+token) in place before PV.
using SmemLayoutK = Layout<
    Shape<Int<NUM_QK_SEG>, Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>,
    Stride<Int<TOPK_BLOCK_SIZE*D_QK_SEG>, Int<D_QK_SEG>, _1>
>;
using SmemLayoutV = Layout<
    Shape<Int<D_VO/D_QK_SEG>, Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>,
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
    float sQNorm[BLOCK_M];   // per-row RMSNorm fold factor (rsqrt(q_sqr/D_QK+eps)), QK softmax does O of the fold
};
static_assert(sizeof(SharedMemoryPlan) <= 101376);

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
// to sScale. WG0 only. See the sparse decode/prefill config.h (identical logic):
// every PV C-segment (4x) rescales by scale_for_olds, do_softmax only on segment 0.
//
// [Fused] When ENABLE_Q_NORM, the per-row RMSNorm denominator is folded into the
// softmax: row r = base+8*lut is pre-scaled by sQNorm[r] = rsqrt(q_sqr/D_QK+eps)
// (see SM100G: score_multiplier = sm_scale_div_log2 * sQNorm, applied on the
// log2-domain scores). Pass sQNorm == nullptr to disable the fold.
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
    bool do_softmax,
    const float* sQNorm = nullptr
) {
    if (do_softmax) {
        const int base = 16*(idx_in_warpgroup/32) + (idx_in_warpgroup%32)/4;
        const int qcol = 2*(idx_in_warpgroup%4);
        constexpr int P_COL_SLOTS = decltype(size(rP))::value/4;
        CUTE_UNROLL
        for (int lut = 0; lut < 2; ++lut) {
            const float norm = sQNorm == nullptr ? 1.0f : sQNorm[base + 8*lut];
            const float eff_scale_log2 = scale_softmax_log2 * norm;
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

            cur_max *= eff_scale_log2;
            float old_max = rM[lut];
            rM[lut] = max(cur_max, old_max);
            scale_for_olds[lut] = exp2f(old_max - rM[lut]);

            float cur_sum = 0;
            CUTE_UNROLL
            for (int nb = 0; nb < P_COL_SLOTS; ++nb) {
                CUTE_UNROLL
                for (int cp = 0; cp < 2; ++cp) {
                    const int g = 4*nb + 2*lut + cp;
                    const float p = exp2f(rP(g)*eff_scale_log2 - rM[lut]);
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
    constexpr int O_COL_SLOTS = decltype(size(rO))::value/4;
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
devfunc(const Params &params);

static void run(const Params &params);

};

}
