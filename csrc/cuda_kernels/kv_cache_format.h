#pragma once

#include "params.h"

// The byte layout of one token of a paged quantized KV cache.
//
// Each V4.1 token stores [raw data, 1-byte scales], including the RoPE dims, so consecutive
// tokens sit BYTES_PER_TOKEN apart ("interleaved"):
//      fp8: 512 B e4m3 + 16 B ue8m0 = 528 B;  fp4: 256 B e2m1 + 32 B e4m3 = 288 B.
// V3.2 and V4 instead keep `page_block_size` raw rows of TMA_K_STRIDE bytes at the start of a
// page block and all the scale rows after them, so their raw data strides by TMA_K_STRIDE, not
// by BYTES_PER_TOKEN. Upstream dropped those two formats in the 2026.09.30 release; the kernels
// that read them are kept under cuda_kernels/legacy/, and no kernel reads both layouts.
// See tests/quant.py for the quantization and dequantization of each.
template<ModelType MT>
struct KVCacheFormat {
    static constexpr ModelType MODEL_TYPE = MT;
    static constexpr bool IS_V32 = MT == ModelType::V32;
    static constexpr bool IS_NVFP4 = MT == ModelType::V32_NVFP4_FP8ROPE;
    static constexpr bool IS_FP4 = MT == ModelType::V41_FP4;
    // Whether every token carries its own scales. Only the V4.1 formats are addressed this way
    // by the current kernels; the V3.2 NVFP4 format below happens to be token-contiguous too,
    // but it is read by a legacy kernel that addresses it through TMA_K_STRIDE.
    static constexpr bool IS_INTERLEAVED = MT == ModelType::V41 || MT == ModelType::V41_FP4;
    static constexpr int D_QK = IS_V32 || IS_NVFP4 ? 576 : 512;
    static constexpr int D_ROPE = 64;
    static constexpr int D_NOPE = D_QK - D_ROPE;
    static constexpr int D_FP4 = IS_NVFP4 ? D_NOPE : (IS_FP4 ? D_QK : 0);               // Dimensions stored as fp4 e2m1
    static constexpr int D_FP8 = IS_NVFP4 ? D_ROPE : (IS_FP4 ? 0 : (MT == ModelType::V41 ? D_QK : D_NOPE)); // Dimensions stored as fp8 e4m3
    static constexpr int D_BF16 = D_QK - D_FP4 - D_FP8;                                 // Dimensions stored as bf16 (the RoPE part of V3.2 / V4), not needing dequant
    static constexpr int QUANT_TILE_SIZE = IS_NVFP4 || IS_FP4 ? 16 : (MT == ModelType::V41 ? 32 : (MT == ModelType::V4 ? 64 : 128)); // Dimensions sharing one scale
    static constexpr int NUM_SCALES_EACH_TOKEN = (IS_V32 || IS_NVFP4 ? D_NOPE : D_QK) / QUANT_TILE_SIZE; // 4 / 8 (7 + 1 byte padding) / 16 / 32
    static constexpr int SCALE_BYTES = IS_V32 ? 4 : 1;                                  // fp32 (V3.2), ue8m0 (V4 / V4.1) or e4m3 (fp4)
    static constexpr int QUANT_BYTES = D_FP4 / 2 + D_FP8;                               // The quantized (fp4 / fp8) data of a token
    // Bytes between two tokens in the data region of a page block: 656 / 576 / 512 / 256. The stride of the tensor maps of the
    // quantized part, so it must be >= 256 for the int32 TMA coordinates to cover a whole KV cache
    static constexpr int TMA_K_STRIDE = IS_NVFP4 ? QUANT_BYTES + NUM_SCALES_EACH_TOKEN :
        QUANT_BYTES + (IS_V32 ? NUM_SCALES_EACH_TOKEN * SCALE_BYTES : 0) + 2 * D_BF16;
    static constexpr int BYTES_PER_TOKEN = IS_NVFP4 ? TMA_K_STRIDE :
        TMA_K_STRIDE + (IS_V32 ? 0 : NUM_SCALES_EACH_TOKEN * SCALE_BYTES);   // 656 / 584 / 528 / 288 / 352
    static_assert(!IS_NVFP4 || BYTES_PER_TOKEN == 352);
    // An interleaved token is exactly its raw data plus its own scales, with nothing in between
    static_assert(!IS_INTERLEAVED || BYTES_PER_TOKEN == QUANT_BYTES + NUM_SCALES_EACH_TOKEN * SCALE_BYTES);
};

// A CTA's contiguous share of a token. In a pair, CTA0 loads its scales separately;
// CTA1's TMA view includes the trailing scales. Shared-memory padding is kernel-specific.
template<typename F, int NUM_CTAS, int CTA>
struct KVCachePart : F {
    static_assert((NUM_CTAS == 1 || NUM_CTAS == 2) && 0 <= CTA && CTA < NUM_CTAS);
    // The offsets below place the scales right after the raw data of the same token
    static_assert(F::IS_INTERLEAVED, "KVCachePart requires a format whose tokens carry their own scales");
    static constexpr int QUANT_DIMS = F::D_QK / NUM_CTAS;
    static constexpr int QUANT_BYTES = F::QUANT_BYTES / NUM_CTAS;
    static constexpr int NUM_SCALES = F::NUM_SCALES_EACH_TOKEN / NUM_CTAS;
    static constexpr int RAW_TOKEN_OFFSET = CTA * QUANT_BYTES;
    static constexpr int SCALE_TOKEN_OFFSET = F::QUANT_BYTES + CTA * NUM_SCALES;
    static constexpr bool SEPARATE_SCALES = NUM_CTAS == 2 && CTA == 0;
    static constexpr int RAW_TOKEN_DATA_BYTES = QUANT_BYTES + (SEPARATE_SCALES ? 0 : F::NUM_SCALES_EACH_TOKEN);
    static constexpr int SCALE_OFFSET = SCALE_TOKEN_OFFSET - RAW_TOKEN_OFFSET;
};

// Runtime counterpart of KVCacheFormat<MT>::BYTES_PER_TOKEN
constexpr int kv_cache_bytes_per_token(ModelType mt) {
    switch (mt) {
        case ModelType::V32: return KVCacheFormat<ModelType::V32>::BYTES_PER_TOKEN;
        case ModelType::V4: return KVCacheFormat<ModelType::V4>::BYTES_PER_TOKEN;
        case ModelType::V41: return KVCacheFormat<ModelType::V41>::BYTES_PER_TOKEN;
        case ModelType::V41_FP4: return KVCacheFormat<ModelType::V41_FP4>::BYTES_PER_TOKEN;
        case ModelType::V32_NVFP4_FP8ROPE: return KVCacheFormat<ModelType::V32_NVFP4_FP8ROPE>::BYTES_PER_TOKEN;
    }
    return 0;
}

// The (kv, extra_kv) format pairs that exist: extra_kv has the format of kv, or is the fp4 cache next to a (fp8) kv.
constexpr bool is_valid_kv_format_pair(ModelType kv, ModelType extra_kv) {
    return extra_kv == kv || (kv == ModelType::V41 && extra_kv == ModelType::V41_FP4);
}

template<ModelType KV, ModelType EXTRA_KV = KV>
struct KVFormatPair {
    static_assert(is_valid_kv_format_pair(KV, EXTRA_KV));
    static constexpr ModelType kv = KV, extra_kv = EXTRA_KV;
};

// A list of KVFormatPair, see dispatch_kv_formats in csrc/api/common.h
template<typename... Pairs>
struct KVFormatPairs {};
