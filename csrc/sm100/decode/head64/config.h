#pragma once

#include "kernel.h"

#include <cuda_fp8.h>
#include <cutlass/barrier.h>
#include <cute/tensor.hpp>

#include <kerutils/kerutils.cuh>

#include "defines.h"
#include "params.h"

namespace sm100::decode::head64 {

using cutlass::arch::fence_view_async_shared;
using cutlass::arch::NamedBarrier;
using e8m0 = __nv_fp8_e8m0;
using e4m3 = cutlass::float_e4m3_t;
using namespace cute;

enum NamedBarriers : uint32_t {
    main_loop_sync = 0,
    wg0_sync = 1,
    wg0_warp02_sync = 2,
    wg0_warp13_sync = 3,
    everyone_sync = 4
};

template<ModelType MODEL_TYPE>
struct KernelTemplate {

// NVFP4 format: V3.2 geometry, NoPE stored as e2m1 (2 elems/byte) with per-16-element
// e4m3 scale factors; RoPE stored as plain e4m3 with no scale factors at all — e4m3 has
// 4 exponent bits, so it spans the RoPE magnitude range unaided and a block scale would
// only cost bytes.
static constexpr bool IS_NVFP4 = MODEL_TYPE == ModelType::V32_NVFP4_FP8ROPE;
// "V32 geometry": d_qk = 576 = 512 NoPE + 64 RoPE, V = NoPE only
static constexpr bool IS_V32_GEOM = MODEL_TYPE == ModelType::V32 || IS_NVFP4;

static constexpr int D_Q = IS_V32_GEOM ? 576 : 512;
static constexpr int D_K = D_Q;
static constexpr int D_V = 512;
static constexpr int D_NOPE = IS_V32_GEOM ? 512 : 448;
static constexpr int D_ROPE = 64;
static constexpr int QUANT_TILE_SIZE = IS_NVFP4 ? 16 : (MODEL_TYPE == ModelType::V32 ? 128 : 64);
static constexpr bool V_HAVE_ROPE = IS_V32_GEOM ? false : true;
static constexpr int NUM_SCALES_EACH_TOKEN = MODEL_TYPE == ModelType::V32 ? 4 : 8;    // Padding is included. Unused for NVFP4 (scales live in the raw tail buffer).

// Raw (quantized) byte counts per token
static constexpr int NOPE_RAW_BYTES = IS_NVFP4 ? D_NOPE/2 : D_NOPE;  // e2m1 packs 2/byte
static constexpr int ROPE_RAW_BYTES = D_ROPE;                        // e4m3, 1 byte per element
// NVFP4 tail region = [64B e4m3 rope | 32B e4m3 nope SF], already a multiple of 16B.
// The tail is TMA-gathered as one box, so scales arrive together with the data they scale.
static constexpr int NVFP4_NUM_NOPE_SCALES = D_NOPE/16;      // 32
static constexpr int NVFP4_SF_NOPE_OFFSET = ROPE_RAW_BYTES;  // within tail
// The 32 SF bytes are NOT stored in element-block order. A dequant thread owns the scale
// groups {4c + q : c = 0..7} for its own q = (idx_in_group/2) in [0, 4), which in element
// order are 8 bytes with stride 4. They are permuted at quantization time so that the scale
// for element block s (covering NoPE dims [16s, 16s+16)) lives at
//     NVFP4_SF_NOPE_OFFSET + NVFP4_SF_BYTE(s),  NVFP4_SF_BYTE(s) = 8*(s&3) + (s>>2)
// which maps thread q's eight scales onto the contiguous bytes [8q, 8q+8) -> one LDS.64.
// Keep this in lockstep with the quantizer (tests/quant.py and the vLLM cache writer).
static constexpr int nvfp4_sf_byte(int s) { return 8*(s & 3) + (s >> 2); }
// Thread q's scale groups {4c+q} must land on the contiguous bytes 8q+c. That also
// makes it a permutation of [0, 32), since both sides cover that range exactly once.
static constexpr bool nvfp4_sf_perm_ok() {
    for (int q = 0; q < 4; ++q)
        for (int c = 0; c < 8; ++c)
            if (nvfp4_sf_byte(4*c + q) != 8*q + c) return false;
    return true;
}
static_assert(nvfp4_sf_perm_ok());
static constexpr int TAIL_BYTES = IS_NVFP4 ? ROPE_RAW_BYTES + NVFP4_NUM_NOPE_SCALES : 16;  // 96; dummy 16 for non-NVFP4
// SMEM staging stride for one tma_gather4 call (4 tokens' tails, packed): TMA requires the
// unswizzled SMEM destination to be 128B-aligned, so pad between 4-token groups.
static constexpr int TAIL_GROUP_STRIDE = ku::ceil(4*TAIL_BYTES, 128);  // 384
static constexpr int BYTES_PER_TOKEN =
    MODEL_TYPE == ModelType::V32 ? D_NOPE + 2*D_ROPE + 4*(D_NOPE/128) :  // 656
    MODEL_TYPE == ModelType::MODEL1 ? D_NOPE + 2*D_ROPE + 8 :            // 584 (per-block scale suffix layout)
    NOPE_RAW_BYTES + TAIL_BYTES;                                         // NVFP4: 352
static constexpr int TMA_K_STRIDE = MODEL_TYPE == ModelType::V32 ? D_NOPE+2*D_ROPE+4*(D_NOPE/QUANT_TILE_SIZE) :
    MODEL_TYPE == ModelType::MODEL1 ? D_NOPE+2*D_ROPE :
    BYTES_PER_TOKEN;   // Stride of K's tensormap. This stride must 1) be a factor of the actual stride between tokens 2) large enough to cover the entire KV cache. Since TMA copy's coordinate can only be 32bit signed integers, this number must >= 128, perferrably >= 256. So we set this to 656 for V32, 576 for MODEL1, and BYTES_PER_TOKEN for NVFP4. Extra padding may be necessary for KV blocks.
static_assert(D_NOPE + D_ROPE == D_Q);
static_assert(V_HAVE_ROPE ? (D_NOPE + D_ROPE == D_V) : (D_NOPE == D_V));
static_assert(!IS_NVFP4 || (TMA_K_STRIDE % 16 == 0 && TMA_K_STRIDE >= 256));
static_assert(!IS_NVFP4 || BYTES_PER_TOKEN == 352);  // Keep in sync with the bytes_per_token literal in csrc/api/sparse_decode.h
// The permuted SF layout is read 8 bytes at a time from the 128B-aligned raw_tail staging
// buffer, at (t/4)*TAIL_GROUP_STRIDE + (t%4)*TAIL_BYTES + NVFP4_SF_NOPE_OFFSET + 8q, so
// every term of that offset must be a multiple of 8.
static_assert(!IS_NVFP4 || (TAIL_GROUP_STRIDE % 8 == 0 && TAIL_BYTES % 8 == 0 &&
                            NVFP4_SF_NOPE_OFFSET % 8 == 0 && NVFP4_NUM_NOPE_SCALES == 32));

static constexpr int B_H = 64;
static constexpr int B_TOPK = 64;
static constexpr int NUM_BUFS = 2;
static constexpr int NUM_INDEX_BUFS = 4;    // Number of buffers for indices (tma_coords) & is_token_valid & scales
static constexpr int NUM_THREADS = 128*3;  // 128 exp + 1/32 utcmma + 1/32 raw KV producer + 1/32 rope producer + 32 index+scale+valid_mask producer + 128 dequant
static constexpr float MAX_INIT_VAL = -1e30f;  // To avoid (-inf) - (-inf) = NaN

static constexpr int D_Q_SW128 = 512;
static constexpr int D_Q_SW64 = IS_V32_GEOM ? 64 : 0;
static_assert(D_Q_SW128 + D_Q_SW64 == D_Q);
static constexpr int K_ROPE_SW = IS_V32_GEOM ? 64 : 128; // RoPE part stored in SW64 (for V32 geometry) or SW128 (for MODEL1), in bytes

template<
    typename Shape_Q_SW128, typename TMA_Q_SW128,
    typename Shape_O, typename TMA_O
>
struct TmaParams {
    Shape_Q_SW128 shape_Q_SW128; TMA_Q_SW128 tma_Q_SW128;
    Shape_O shape_O; TMA_O tma_O;
    CUtensorMap tensor_map_q_sw64;  // Invalid if D_Q_SW64 == 0
    CUtensorMap tensor_map_kv_nope;
    CUtensorMap tensor_map_kv_rope;
    CUtensorMap tensor_map_extra_kv_nope;
    CUtensorMap tensor_map_extra_kv_rope;
};

// Tensor memory columns
struct tmem_cols {
    //   0 ~ 256: output
    // 256 ~ 256 + 64*D_Q/256: Q
    // 400 ~ 464: P
    static constexpr int O = 0;
    static constexpr int Q = 256;
    static constexpr int Q_Tail = 256 + B_H*D_NOPE/2/128;
    static constexpr int P = 400;
};

template<int NUM_TILES>
using SmemLayoutQTiles = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<NUM_TILES*64>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

using SmemLayoutQ_SW128 = SmemLayoutQTiles<D_Q_SW128/64>;

using SmemLayoutOBuf = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<D_V>>{}
));

using SmemLayoutOBuf_TMA = decltype(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64>>{}
)); // A TMA tile

static_assert(D_V == 512);
using SmemLayoutOAccumBuf = Layout<
    Shape<Int<B_H>, Int<D_V>>,
    Stride<Int<520>, _1>	// We use stride = 520 here to avoid bank conflict
>;

using SmemLayoutS = decltype(tile_to_shape(
    UMMA::Layout_K_INTER_Atom<bf16>{},
    Shape<Int<B_H>, Int<B_TOPK>>{},
    Step<_1, _2>{}
));

template<int NUM_TILES>
using SmemLayoutKTiles_SW128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles_DualGemm_SW128 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW128_Atom<bf16>{},
    Shape<Int<B_H*2>, Int<64*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed_SW128 = decltype(composition(
    SmemLayoutKTiles_SW128<NUM_TILES>{},
    Layout<
        Shape<Int<64*NUM_TILES>, Int<B_TOPK>>,
        Stride<Int<B_TOPK>, _1>
    >{}
));

template<int NUM_TILES>
using SmemLayoutKTiles_SW64 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H>, Int<32*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTiles_DualGemm_SW64 = decltype(coalesce(tile_to_shape(
    UMMA::Layout_K_SW64_Atom<bf16>{},
    Shape<Int<B_H*2>, Int<32*NUM_TILES>>{},
    Step<_1, _2>{}
), Shape<_1, _1>{}));

template<int NUM_TILES>
using SmemLayoutKTilesTransposed_SW64 = decltype(composition(
    SmemLayoutKTiles_SW64<NUM_TILES>{},
    Layout<
        Shape<Int<32*NUM_TILES>, Int<B_TOPK>>,
        Stride<Int<B_TOPK>, _1>
    >{}
));

struct SharedMemoryPlan {
    union {
        struct {
            array_aligned<bf16, cosize_v<SmemLayoutQ_SW128>> q;
            bf16 q_sw64[B_H*D_Q_SW64];  // NOTE D_Q_SW64 may be 0 but array_aligned<bf16, 0> will have a size of 16, so we use array here. The former tensor (`q`) promises its alignment.
            union {
                array_aligned<bf16, cosize_v<SmemLayoutOBuf>> o_buf;
                array_aligned<float, cosize_v<SmemLayoutOAccumBuf>> o_accum_buf;
            } o;
        } qo;
        struct {
            struct {
                array_aligned<bf16, B_H*D_NOPE> nope; // NoPE part, dequantized
                array_aligned<bf16, B_H*D_ROPE> rope; // RoPE part, dequantized. SW64 in v32 mode, SW128 in MODEL1 mode
            } dequant[NUM_BUFS];
            static_assert(sizeof(dequant) >= sizeof(bf16) * (B_H*D_Q)); // So that Q does not covers raw_nope
            array_aligned<uint8_t, B_TOPK*NOPE_RAW_BYTES> raw_nope[NUM_BUFS];  // Raw (quantized) NoPE part
            // NVFP4 only: raw tail region per token = [rope raw | nope SFs],
            // TMA-gathered as one box per 4 tokens; groups of 4 packed tails are padded to
            // TAIL_GROUP_STRIDE for TMA alignment. Token t lives at
            // (t/4)*TAIL_GROUP_STRIDE + (t%4)*TAIL_BYTES.
            // Dummy-sized (16B total, unused) for other formats.
            array_aligned<uint8_t, IS_NVFP4 ? (B_TOPK/4)*TAIL_GROUP_STRIDE : 16, IS_NVFP4 ? 128 : 16> raw_tail[IS_NVFP4 ? NUM_BUFS : 1];
        } kv;
    } u;
    union {
        float4 p_exchange_buf[4][16 * B_TOPK / 4];
        array_aligned<bf16, cosize_v<SmemLayoutS>> s;
    } s_p;
    CUTE_ALIGNAS(16) float rowwise_max_buf[128];
    char is_token_valid[NUM_INDEX_BUFS][B_TOPK/8];
    int tma_coord[NUM_INDEX_BUFS][B_TOPK];
    e8m0 scales[NUM_INDEX_BUFS][B_TOPK][IS_NVFP4 ? 1 : NUM_SCALES_EACH_TOKEN];  // Unused for NVFP4 (scales live in raw_tail)
    array_aligned<uint32_t, 1> tmem_start_addr;
    transac_bar_t bar_last_store_done;
    transac_bar_t bar_q_tma, bar_q_utccp;
    transac_bar_t bar_rope_ready[NUM_BUFS];   // Non-NVFP4: rope TMA done (init 1, expect_tx). NVFP4: rope dequant done (init 128, arrived by the dequant warpgroup)
    transac_bar_t bar_nope_ready[NUM_BUFS];
    transac_bar_t bar_raw_ready[NUM_BUFS], bar_raw_free[NUM_BUFS];
    // NVFP4 only: raw tail TMA done (init 1, expect_tx) / raw tail consumed by dequant WG (init 128)
    transac_bar_t bar_rawtail_ready[NUM_BUFS], bar_rawtail_free[NUM_BUFS];
    transac_bar_t bar_valid_coord_scale_ready[NUM_INDEX_BUFS], bar_valid_coord_scale_free[NUM_INDEX_BUFS];
    transac_bar_t bar_qk_done[NUM_BUFS], bar_so_ready[NUM_BUFS], bar_sv_done[NUM_BUFS];
};

using TiledMMA_P = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<bf16, bf16, float, B_H, B_TOPK*2, UMMA::Major::K, UMMA::Major::K>{}
)); // *2 for dual gemm

using TiledMMA_O = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<bf16, bf16, float, B_H, 256, UMMA::Major::K, UMMA::Major::MN>{}
));

template<typename TmaParam>
static __device__ void
flash_fwd_splitkv_mla_fp8_sparse_kernel_devfunc(const SparseAttnDecodeParams &params, const TmaParam &tma_params);

static void run(const SparseAttnDecodeParams &params);

};

}