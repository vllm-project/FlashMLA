#pragma once

#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <math_constants.h>

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>

#include <kerutils/kerutils.cuh>

#include "cuda_kernels/legacy/defines.h"
#include "params.h"
#include "cuda_kernels/kv_cache_format.h"
#include "cuda_kernels/sm100/helpers.h"                       // fp4 / fp8 -> bf16 conversion primitives (sm100::)
#include "../../../../decode/sparse/components/dequant.h" // sm120::decode::sparse: 128-bit global loads + fp8x8 -> bf16x8

#include "config.h"
#include "kernel.h"

using namespace cute;

namespace sm120::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn {

static constexpr float MAX_INIT_VAL = -1e30f;    // Prevent (-inf) - (-inf) = nan
static constexpr float O_QUANT_CLAMP_MIN_VALUE = 1e-4f;   // matches the sm100 O-epilogue
using cutlass::arch::fence_view_async_shared;
using fp8_e4m3 = cutlass::float_e4m3_t;
using fp8x16 = sm120::decode::sparse::fp8x16;   // bf16x8 comes from the global scope (cuda_kernels/legacy/defines.h)
using sm120::decode::sparse::L1CacheHint;
using sm120::decode::sparse::L2PrefetchHint;
using sm120::decode::sparse::load_128b_from_gmem;

// ---------------------------------------------------------------------------
// Rotational (non-neox) RoPE of one consecutive-dim pair, last 64 dims only.
//    y = (x0*cos - x1*sin, x0*sin + x1*cos)
// Q-side uses apply_rope; the O-side RoPE is the conjugate rotation (see
// apply_conjugate_rope below), both taking cos/sin of freq = col/2.
// ---------------------------------------------------------------------------
__device__ __forceinline__
float2 apply_rope(const float2 &x, float cos_v, float sin_v) {
    return {(x.x * cos_v - x.y * sin_v), (x.x * sin_v + x.y * cos_v)};
}

__device__ __forceinline__
float2 apply_conjugate_rope(const float2 &x, float cos_v, float sin_v) {
    return {(x.x * cos_v + x.y * sin_v), (-x.x * sin_v + x.y * cos_v)};
}

// ---------------------------------------------------------------------------
// Q is stored in the permuted layout produced by `permute_q_b_proj`: element
// (head h, global dim d) of a query lives at physical bf16 offset
//     (d % 16) + h*16 + (d/16) * H_Q * 16        (H_Q = CONFIG.H_Q, not BLOCK_M)
// i.e. the 16 dims of one (h, d/16) block are contiguous. All reads go through
// this layout; `stride_q_h_q` is unused (the h stride is hard-coded to 16).
// ---------------------------------------------------------------------------

// Cache this thread's 8 RoPE (cos, sin) values. A thread's QK/O fragment lanes
// cover cols 8*b + 2*(t%4) + {0,1}; the pair's freq index is col/2 = 4b + (t%4),
// so per thread the 8 freqs are {4b + (t%4)} for b in 0..7. `cached[b]` holds
// the cos/sin of freq 4b + (t%4). Both warpgroups need this (WG0 for Q, WG1 for
// the conjugate O RoPE), so it is computed by every thread.
template<Config CONFIG>
__device__ __forceinline__ void cache_rope_values(
    const typename Kernel<CONFIG>::Params& params,
    int s_q_idx,
    int idx_in_warpgroup,
    float cached_cos[8],
    float cached_sin[8]
) {
    static constexpr uint32_t D_ROPE = Kernel<CONFIG>::D_ROPE;
    const int t = idx_in_warpgroup;
    const uint32_t cur_position = __ldg(params.token_positions + s_q_idx);
    // 32-bit cache offset, matching sm100's form (params.cos_sin_cache
    // + pos*D_ROPE): off_c <= 131071*64 = 8,388,480 fits uint32, so the offset
    // index stays 32-bit and the load is LDG [base64][off32]. pos is clamped to
    // 131071 (the cache's row count) so a malformed/out-of-range cur_position
    // cannot walk past the allocated cos/sin cache (defensive only; upstream
    // callers always pass valid positions).
    const uint32_t pos = min(cur_position, (uint32_t)(131072 - 1));
    const uint32_t off_c = pos * D_ROPE;
    CUTE_UNROLL
    for (int b = 0; b < 8; ++b) {
        const int freq = 4*b + (t%4);
        cached_cos[b] = __ldg(params.cos_sin_cache + off_c + freq);
        cached_sin[b] = __ldg(params.cos_sin_cache + off_c + D_ROPE/2 + freq);
    }
}

// Compute the per-row Q RMSNorm denominators (WG0 only). |q|^2 is summed over
// all 512 dims by the 4 threads of each base group (rows base & base+8), then
// reduced with the same shfl_xor(1,2) pattern as rL/rM. The denominator is
// stored to smem so that scale_softmax can fold it into the score multiplier:
//     eff_scale_log2 = sm_scale_div_log2 * rsqrt(q_sqr / D_QK + eps)
// (|q|^2 is rotation-invariant, so the raw Q is perfectly fine here.)
template<Config CONFIG>
__device__ __forceinline__ void compute_q_norm(
    const typename Kernel<CONFIG>::Params& params,
    typename Kernel<CONFIG>::SharedMemoryPlan& plan,
    int s_q_idx,
    int head_block_idx,
    int idx_in_warpgroup
) {
    static constexpr uint32_t H_Q = Kernel<CONFIG>::H_Q;
    static constexpr uint32_t D_QK = Kernel<CONFIG>::D_QK;
    static constexpr uint32_t D_QK_SEG = Kernel<CONFIG>::D_QK_SEG;
    static constexpr uint32_t BLOCK_M = Kernel<CONFIG>::BLOCK_M;
    const int t = idx_in_warpgroup;
    const int base = 16*(t/32) + (t%32)/4;

    float sq[2] = {0.0f, 0.0f};
    const bf16* q_token_base = (const bf16*)params.q + (int64_t)s_q_idx*(int64_t)params.stride_q_s_q;
    CUTE_NO_UNROLL
    for (int seg = 0; seg < 8; ++seg) {
        CUTE_UNROLL
        for (int g = 0; g < 32; ++g) {
            const int lut = (g%4)/2, cp = (g%4)%2;
            const int row = base + 8*lut;
            const int col = 8*(g/4) + 2*(t%4) + cp;
            const int d = seg*D_QK_SEG + col;
            const uint32_t h = head_block_idx*BLOCK_M + row;
            const int64_t off = (int64_t)h*16 + (int64_t)(d/16)*(int64_t)H_Q*16 + (d%16);
            const float val = __ldg((const __nv_bfloat16*)(q_token_base + off));
            sq[lut] = fmaf(val, val, sq[lut]);
        }
    }
    sq[0] += __shfl_xor_sync(0xffffffff, sq[0], 1);
    sq[0] += __shfl_xor_sync(0xffffffff, sq[0], 2);
    sq[1] += __shfl_xor_sync(0xffffffff, sq[1], 1);
    sq[1] += __shfl_xor_sync(0xffffffff, sq[1], 2);
    if (idx_in_warpgroup % 4 == 0) {
        plan.sQNorm[base]     = rsqrtf(sq[0] / (float)D_QK + params.rms_norm_eps);
        plan.sQNorm[base + 8] = rsqrtf(sq[1] / (float)D_QK + params.rms_norm_eps);
    }
}

// ---------------------------------------------------------------------------
// Load this thread's QK A-fragment (row/col map: base = 16*(t/32)+(t%32)/4,
// row = base + 8*((g%4)/2), col = 8*(g/4)+2*(t%4)+((g%4)%2)) of Q segment `seg`
// (64 dims) from the permuted global Q into rQa, with the non-neox RoPE applied
// on the last 64 dims (seg == 7). When `sq` is non-null the per-row |q|^2
// accumulates for the RMSNorm pre-pass.
// ---------------------------------------------------------------------------
template<Config CONFIG, typename Tensor>
__device__ __forceinline__ void load_q_fragment(
    const typename Kernel<CONFIG>::Params& params,
    Tensor& rQa,
    int seg,
    int s_q_idx,
    int head_block_idx,
    int idx_in_warpgroup,
    const float cached_cos[8],
    const float cached_sin[8],
    float* sq = nullptr
) {
    static constexpr uint32_t H_Q = Kernel<CONFIG>::H_Q;
    static constexpr uint32_t D_QK_SEG = Kernel<CONFIG>::D_QK_SEG;
    static constexpr uint32_t BLOCK_M = Kernel<CONFIG>::BLOCK_M;
    const int t = idx_in_warpgroup;
    const int base = 16*(t/32) + (t%32)/4;

    const bf16* q_token_base = (const bf16*)params.q + (int64_t)s_q_idx*(int64_t)params.stride_q_s_q;

    CUTE_UNROLL
    for (int g = 0; g < 32; ++g) {
        const int lut = (g%4)/2, cp = (g%4)%2;
        const int row = base + 8*lut;
        const int col = 8*(g/4) + 2*(t%4) + cp;
        const int d = seg*D_QK_SEG + col;
        const uint32_t h = head_block_idx*BLOCK_M + row;
        const int64_t off = (int64_t)h*16 + (int64_t)(d/16)*(int64_t)H_Q*16 + (d%16);
        float val = __ldg((const __nv_bfloat16*)(q_token_base + off));
        if (sq != nullptr)
            sq[lut] = fmaf(val, val, sq[lut]);
        rQa(g) = (bf16)val;   // explicit cast: cutlass bfloat16_t has no operator=(float)
    }
    // RoPE pairs (consecutive cols) of the last 64 dims; freq = 4b + (t%4) -> cached[b]
    if (seg == 7) {
        CUTE_UNROLL
        for (int b = 0; b < 8; ++b) {
            CUTE_UNROLL
            for (int lut = 0; lut < 2; ++lut) {
                const int g = 4*b + 2*lut;
                const float2 x = {(float)rQa(g), (float)rQa(g+1)};
                const float2 y = apply_rope(x, cached_cos[b], cached_sin[b]);
                rQa(g) = (bf16)y.x;
                rQa(g+1) = (bf16)y.y;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dequantize ONE 64-token KV block into smem (decode only). Unified slot space
// (mirrors the sm100 fused decode): slots [0, num_orig_slots) come from `kv`
// (indices), the rest from `extra_kv` (extra_indices), resolved PER TOKEN.
//   have_extra:  num_orig_slots = params.topk (padded); valid = topk_length
//   no extra:    num_orig_slots = 0xFFFFFFFF (orig only), valid = topk_length
// `dequant_token_format<Fmt>` handles one token (its 4-thread dim strip) for a
// compile-time format: fp8 e4m3 (V4/V41 nope), fp4 e2m1 (V41_FP4) + bf16 rope
// (V4). V4 clusters a page block's scale rows after its data rows; V41 / V41_FP4
// interleave each token's scales with that token's own data.
// ---------------------------------------------------------------------------
template<Config CONFIG, typename Fmt>
__device__ __forceinline__ void dequant_token_format(
    const typename Kernel<CONFIG>::Params& params,
    typename Kernel<CONFIG>::SharedMemoryPlan& plan,
    int block_index, int rel_idx,
    const uint8_t* k_ptr, int64_t k_block_stride, int page_block_size,
    bool is_valid
) {
    static constexpr uint32_t TOPK_BLOCK_SIZE = Kernel<CONFIG>::TOPK_BLOCK_SIZE;
    static constexpr uint32_t D_QK_SEG = Kernel<CONFIG>::D_QK_SEG;
    constexpr int SEG_STRIDE = TOPK_BLOCK_SIZE * D_QK_SEG;
    constexpr int NUM_SCALES = Fmt::NUM_SCALES_EACH_TOKEN;

    const int token = threadIdx.x & 63;
    const int tg = threadIdx.x >> 6;    // 0..3 -> one of the 4 dim strips per token

    // V4 keeps a page block's raw data rows and its scale rows in two separate regions, so the
    // raw rows stride by TMA_K_STRIDE. V41 / V41_FP4 (interleaved) store a token's scales right
    // after that token's own data, so the stride is the whole token and the scales follow it.
    constexpr uint32_t TOKEN_STRIDE = Fmt::IS_INTERLEAVED ? Fmt::BYTES_PER_TOKEN : Fmt::TMA_K_STRIDE;
    const uint8_t* gK_base = k_ptr + (int64_t)block_index*k_block_stride + (int64_t)rel_idx*TOKEN_STRIDE;
    const uint8_t* gK_scales = Fmt::IS_INTERLEAVED
        ? gK_base + Fmt::QUANT_BYTES
        : k_ptr + (int64_t)block_index*k_block_stride
            + (int64_t)page_block_size*Fmt::TMA_K_STRIDE + (int64_t)rel_idx*NUM_SCALES;

    // (SEG, TOKEN, COL64) smem layout: dim d -> seg = d/64, col = d%64,
    // element address = seg*SEG_STRIDE + token*64 + col.
    auto sk_addr = [&](int dim) -> __int128_t* {
        return (__int128_t*)(plan.k.data() + (dim / D_QK_SEG)*SEG_STRIDE + token*D_QK_SEG + dim % D_QK_SEG);
    };

    // ------------------- scales as bf16 -------------------------------
    if constexpr (Fmt::D_FP4 > 0) {
        static_assert(Fmt::IS_FP4 && NUM_SCALES == 32);   // V41_FP4: 32x e4m3 (tile 16)
        bf16 scales[32];
        CUTE_UNROLL
        for (int i = 0; i < NUM_SCALES; i += 2) {
            const ku::nvbf16x2 sc_tmp = ::sm100::e4m3x2_to_bf16x2(*(uint16_t*)(gK_scales + i));
            *(uint32_t*)(scales + i) = *(const uint32_t*)&sc_tmp;
        }
        if (!is_valid) {
            CUTE_UNROLL
            for (int i = 0; i < NUM_SCALES; ++i) scales[i] = (bf16)0.0f;
        }
        // fp4 e2m1: 2 dims per byte; 16 dims = 8 bytes = 2 x uint32 (fp4x8)
        CUTE_UNROLL
        for (int i = 0; i < 8; ++i) {
            const int dim16 = (tg*8 + i)*16;
            if (dim16 >= Fmt::D_FP4) break;
            const uint8_t* src = gK_base + dim16/2;
            uint32_t packed0 = *(const uint32_t*)(src);
            uint32_t packed1 = *(const uint32_t*)(src + 4);
            if (!is_valid) { packed0 = packed1 = 0; }
            // cutlass bf16 and __nv_bfloat16 share the same 2-byte representation:
            // broadcast via bit-reinterpret (same trick as the fp8 path below).
            const ku::nvbf16x2 sc2 = __bfloat162bfloat162(*(__nv_bfloat16*)&scales[dim16 / Fmt::QUANT_TILE_SIZE]);
            ku::nvbf16x2 out0[4], out1[4];
            ::sm100::fp4x8_to_bf16x2x4(packed0, out0);
            ::sm100::fp4x8_to_bf16x2x4(packed1, out1);
            CUTE_UNROLL
            for (int j = 0; j < 4; ++j) {
                out0[j] = __hmul2(out0[j], sc2);
                out1[j] = __hmul2(out1[j], sc2);
            }
            *sk_addr(dim16)     = *(__int128_t*)(out0);
            *sk_addr(dim16 + 8) = *(__int128_t*)(out1);
        }
    } else if constexpr (Fmt::D_FP8 > 0) {
        bf16 scales[NUM_SCALES];
        CUTE_UNROLL
        for (int i = 0; i < NUM_SCALES; i += 2)
            *(__nv_bfloat162_raw*)(scales + i) = __nv_cvt_e8m0x2_to_bf162raw(*(__nv_fp8x2_storage_t*)(gK_scales + i));
        if (!is_valid) {
            CUTE_UNROLL
            for (int i = 0; i < NUM_SCALES; ++i) scales[i] = (bf16)0.0f;
        }
        // fp8 e4m3: 16 dims per 16B step
        CUTE_UNROLL
        for (int i = 0; i < 8; ++i) {
            const int dim16 = (tg*8 + i)*16;
            if (dim16 >= Fmt::D_FP8) break;
            fp8x16 cur = load_128b_from_gmem<fp8x16, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(gK_base + dim16);
            if (!is_valid) *(uint128_t*)(&cur) = uint128_t();
            const int si = dim16 / Fmt::QUANT_TILE_SIZE;
            __nv_bfloat162 sc2 = __bfloat162bfloat162(*(__nv_bfloat16*)(&scales[si]));
            bf16x8 lo = cvt_fp8x8_bf16x8(cur.lo, sc2);
            bf16x8 hi = cvt_fp8x8_bf16x8(cur.hi, sc2);
            *sk_addr(dim16)     = *(__int128_t*)&lo;
            *sk_addr(dim16 + 8) = *(__int128_t*)&hi;
        }
    }

    // ------------------- bf16 RoPE dims (V4 only) ----------------------
    if constexpr (Fmt::D_BF16 > 0) {
        static_assert(Fmt::D_FP8 == 448 && Fmt::D_BF16 == 64);
        const int base_dim = Fmt::D_QK - Fmt::D_BF16;     // 448
        CUTE_UNROLL
        for (int j = 0; j < 2; ++j) {
            // gK_base is a byte pointer: the bf16 RoPE segment of one token is
            // 128B starting at byte `base_dim`, split into 4 tg x 32B, each loaded
            // as 2x16B (tg*32 + j*16). Do not reuse the dim-space offsets here;
            // that would land 8B-aligned and fault a 16B vector load.
            bf16x8 cur = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_base + base_dim + tg*32 + j*16);
            if (!is_valid) *(uint128_t*)(&cur) = uint128_t();
            *sk_addr(base_dim + tg*16 + j*8) = *(__int128_t*)&cur;
        }
    }
}

template<Config CONFIG>
__device__ __forceinline__ void dequant_k_block_fused(
    const typename Kernel<CONFIG>::Params& params,
    typename Kernel<CONFIG>::SharedMemoryPlan& plan,
    int s_q_idx, int block_idx
) {
    static constexpr ModelType MT = CONFIG.MODEL_TYPE;
    static constexpr ModelType EMT = CONFIG.EXTRA_MODEL_TYPE;
    using OrigFmt = typename Kernel<CONFIG>::template KVFormat<MT>;
    using ExtraFmt = typename Kernel<CONFIG>::template KVFormat<EMT>;

    static constexpr uint32_t TOPK_BLOCK_SIZE = Kernel<CONFIG>::TOPK_BLOCK_SIZE;

    const int token = threadIdx.x & 63;
    const int pos = block_idx*TOPK_BLOCK_SIZE + token;

    const bool have_extra = params.extra_kv != nullptr && params.extra_topk > 0;
    const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
    const int extra_topk_length = params.extra_topk_length != nullptr ? __ldg(params.extra_topk_length + s_q_idx) : params.extra_topk;
    const uint32_t num_orig_slots = have_extra ? (uint32_t)params.topk : 0xFFFFFFFFu;

    const bool in_extra = have_extra && (uint32_t)pos >= num_orig_slots;
    const uint32_t local_pos = in_extra ? (uint32_t)pos - num_orig_slots : (uint32_t)pos;
    const uint32_t valid_len = in_extra ? (uint32_t)extra_topk_length : (uint32_t)topk_length;

    // Predicate the load on `local_pos < valid_len`: the last KV block may run
    // beyond the end of an indices row, so an unconditional load could read OOB.
    int token_index = -1;
    if (local_pos < valid_len) {
        const int* idx_ptr = in_extra ? params.extra_indices : params.indices;
        const int stride_idx = in_extra ? params.stride_extra_indices_s_q : params.stride_indices_s_q;
        token_index = __ldg(idx_ptr + (int64_t)s_q_idx*stride_idx + (int64_t)local_pos);
    }
    const bool is_valid = token_index >= 0;
    plan.is_kv_valid[token] = is_valid;

    const int page_block_size = in_extra ? params.extra_page_block_size : params.page_block_size;
    const int64_t k_block_stride = in_extra ? params.stride_extra_kv_block : params.stride_kv_block;
    const uint8_t* k_ptr = (const uint8_t*)(in_extra ? params.extra_kv : params.kv);

    // uint32 div/mod: token_index == -1 (UINT_MAX) keeps block_index/rel in range,
    // and the loads are masked to zero when !is_valid.
    const int block_index = token_index == -1 ? 0 : (int)((uint32_t)token_index / (uint32_t)page_block_size);
    const int rel_idx = (uint32_t)token_index % (uint32_t)page_block_size;

    if (in_extra) {
        dequant_token_format<CONFIG, ExtraFmt>(params, plan, block_index, rel_idx, k_ptr, k_block_stride, page_block_size, is_valid);
    } else {
        dequant_token_format<CONFIG, OrigFmt>(params, plan, block_index, rel_idx, k_ptr, k_block_stride, page_block_size, is_valid);
    }
}

// ---------------------------------------------------------------------------
// Prefill: straight bf16 copy of one 64-token KV block. `kv` is raw bf16
// [s_kv, 1, d_qk]; `indices` (stride stride_indices_s_q) selects the rows of
// site s_q_idx. Only reads KV (no dequant); validity is bounded by s_kv.
// ---------------------------------------------------------------------------
template<Config CONFIG>
__device__ __forceinline__ void copy_k_block_bf16(
    const typename Kernel<CONFIG>::Params& params,
    typename Kernel<CONFIG>::SharedMemoryPlan& plan,
    int s_q_idx, int block_idx
) {
    static constexpr uint32_t TOPK_BLOCK_SIZE = Kernel<CONFIG>::TOPK_BLOCK_SIZE;
    static constexpr uint32_t D_QK_SEG = Kernel<CONFIG>::D_QK_SEG;
    constexpr int SEG_STRIDE = TOPK_BLOCK_SIZE * D_QK_SEG;

    const int token = threadIdx.x & 63;
    const int tg = threadIdx.x >> 6;    // 0..3 -> one of the 4 dim strips per token
    const int pos = block_idx*TOPK_BLOCK_SIZE + token;
    const int topk_cur = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;

    int token_index = pos < topk_cur ? __ldg(params.indices + (int64_t)s_q_idx*params.stride_indices_s_q + pos) : -1;
    const bool is_valid = token_index >= 0 && token_index < params.s_kv;
    plan.is_kv_valid[token] = is_valid;

    const int row = is_valid ? token_index : 0;   // masked read
    const bf16* gK_row = (const bf16*)params.kv + (int64_t)row*params.stride_kv_s_kv;

    auto sk_addr = [&](int dim) -> __int128_t* {
        return (__int128_t*)(plan.k.data() + (dim / D_QK_SEG)*SEG_STRIDE + token*D_QK_SEG + dim % D_QK_SEG);
    };

    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
        const int dim16 = (tg*8 + i)*16;
        CUTE_UNROLL
        for (int j = 0; j < 2; ++j) {
            bf16x8 cur = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>(gK_row + dim16 + j*8);
            if (!is_valid) *(uint128_t*)(&cur) = uint128_t();
            *sk_addr(dim16 + j*8) = *(__int128_t*)&cur;
        }
    }
}

// ---------------------------------------------------------------------------
// Device entry: the fused "Q RMSNorm + Q RoPE + Sparse Attention + conjugate
// O RoPE + FP8 cast" kernel for SM120 (register mma.sync only). Same segmented
// 2-warpgroup dataflow as the verified sm120 sparse decode / prefill kernels:
//   WG0: QK (Q from the permuted global layout) -> mask/softmax -> rS/sS,
//        then PV over V dims [0, 256)
//   WG1: PV over V dims [256, 512) using sS
//   K fills all 512 dims (QK + V), transposed in place by thread 0.
// Epilogue (both WGs): per-row o_scales from sOScale (sm100 semantics: the
// per-32 scale interacts with the running softmax sum), conjugate RoPE on the
// last 64 dims (WG1's last segment), per-32-tile absmax and e4m3 cast into the
// permuted out_fp8 layout, sf bytes into the MN-major out_sf layout.
// ---------------------------------------------------------------------------
template<Config CONFIG>
__device__ __forceinline__ void Kernel<CONFIG>::devfunc(const Params &params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    constexpr bool IS_DECODE = is_decode_v<FWD_MODE>;

    const int num_head_blocks = H_Q / BLOCK_M;
    const int s_q_idx = blockIdx.x / num_head_blocks;
    const int head_block_idx = blockIdx.x % num_head_blocks;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    Tensor sK = make_tensor(make_smem_ptr(plan.k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(plan.k.data()), SmemLayoutV{});   // K's 8 segments, (SEG, TOPK, 64)
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});

    TiledMMA_QK tiled_mma_QK{};
    auto thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
    TiledMMA_PV_LocalP tiled_mma_PV{};
    auto thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

    // Accumulators. WG0 owns rL/rM (the block loop also scales WG1's rO), so
    // only WG0's rO needs zeroing here.
    float rL[2], rM[2];
    using ROType = decltype(partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}));
    ROType rO[4] = {
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
    };
    Tensor rP = partition_fragment_C(tiled_mma_QK, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
    Tensor rS = make_tensor<bf16>(partition_shape_A(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

    // attn_sink in log2 domain, per global head index of this CTA's rows.
    float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
    if (params.attn_sink != nullptr) {
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
            const int head_idx = head_block_idx*BLOCK_M + get_AorC_row_idx(i, idx_in_warpgroup);
            rAttn_sink[i] = __ldg((float*)params.attn_sink + head_idx) * CUDART_L2E_F;
        }
    }

    if (warpgroup_idx == 0) {
        rL[0] = rL[1] = 0.0f;
        rM[0] = rM[1] = MAX_INIT_VAL;
        CUTE_UNROLL
        for (int s = 0; s < 4; ++s)
            cute::fill(rO[s], 0.f);
    }

    // RoPE cos/sin cache (both warpgroups) + Q RMSNorm denominators (WG0).
    float cached_cos[8], cached_sin[8];
    cache_rope_values<CONFIG>(params, s_q_idx, idx_in_warpgroup, cached_cos, cached_sin);
    if (warpgroup_idx == 0 && CONFIG.ENABLE_Q_NORM)
        compute_q_norm<CONFIG>(params, plan, s_q_idx, head_block_idx, idx_in_warpgroup);
    __syncthreads();

    // Number of 64-token KV blocks. Prefill / no-extra decode: the fused
    // semantics (sm100) run max(ceil(topk_length/64), 1) blocks and mask the
    // tail. With an extra KV cache the orig + extra slot spaces are unified:
    // ceil((topk + extra_topk_length)/64), blocks may straddle the boundary.
    int num_kv_blocks;
    if constexpr (IS_DECODE) {
        const int topk_length = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
        const int extra_topk_length = params.extra_topk_length != nullptr ? __ldg(params.extra_topk_length + s_q_idx) : params.extra_topk;
        const bool have_extra = params.extra_kv != nullptr && params.extra_topk > 0;
        if (have_extra)
            num_kv_blocks = (params.topk + extra_topk_length + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE;
        else
            num_kv_blocks = max((topk_length + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE, 1);
    } else {
        const int topk_cur = params.topk_length != nullptr ? __ldg(params.topk_length + s_q_idx) : params.topk;
        num_kv_blocks = max((topk_cur + TOPK_BLOCK_SIZE - 1) / TOPK_BLOCK_SIZE, 1);
    }

    const float* sQNormPtr = CONFIG.ENABLE_Q_NORM ? plan.sQNorm : nullptr;

    CUTE_NO_UNROLL
    for (int block_idx = 0; block_idx < num_kv_blocks; ++block_idx) {
        // -------- phase 1: load / dequant K block (all 256 threads) --------
        if constexpr (IS_DECODE) {
            dequant_k_block_fused<CONFIG>(params, plan, s_q_idx, block_idx);
        } else {
            copy_k_block_bf16<CONFIG>(params, plan, s_q_idx, block_idx);
        }
        __syncthreads();

        // -------- phase 2 (WG0): QK + mask/softmax -> rS/sS --------
        if (warpgroup_idx == 0) {
            cute::clear(rP);
            CUTE_NO_UNROLL
            for (int seg = 0; seg < 8; ++seg) {
                Tensor sKseg = make_tensor(
                    sK.data() + seg*D_QK_SEG*TOPK_BLOCK_SIZE,
                    Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>, Stride<Int<D_QK_SEG>, _1>>{});
                Tensor rQa = thr_mma_QK.partition_fragment_A(
                    make_tensor(make_gmem_ptr((bf16*)params.q), make_shape(Int<BLOCK_M>{}, Int<D_QK_SEG>{})));
                Tensor rKb = thr_mma_QK.partition_fragment_B(sKseg);
                load_q_fragment<CONFIG>(params, rQa, seg, s_q_idx, head_block_idx, idx_in_warpgroup, cached_cos, cached_sin);
                cute::copy(thr_mma_QK.partition_B(sKseg), rKb);
                cute::gemm(tiled_mma_QK, rQa, rKb, rP);
            }

            // softmax on segment 0; the QK scale folds in the per-row RMSNorm
            // denominator when ENABLE_Q_NORM (sQNorm precomputed above).
            float scale_for_olds[2];
            CUTE_UNROLL
            for (int s = 0; s < 4; ++s)
                scale_softmax(rP, rS, rO[s], params.sm_scale_div_log2, plan.sScale, rM, rL, plan.is_kv_valid,
                              block_idx, idx_in_warpgroup, scale_for_olds, s == 0, sQNormPtr);

            save_rS_to_sS(rS, sS, idx_in_warpgroup);
            fence_view_async_shared();
            // In-place K -> V transpose for the PV B operand (thread 0 only;
            // __syncthreads() below establishes visibility for both WGs).
            {
                // Parallel K->V transpose across the 128 WG0 threads. Visibility
                // to both warpgroups via the __syncthreads() right after this
                // branch.
                bf16* svbase = plan.k.data();
                const int seg_elems = D_QK_SEG * TOPK_BLOCK_SIZE;
                const int total = 8 * seg_elems;
                CUTE_UNROLL
                for (int X = threadIdx.x; X < total; X += NUM_THREADS / 2) {
                    const int sg = X / seg_elems;
                    const int x = X - sg * seg_elems;
                    const int t_ = x / D_QK_SEG, d = x % D_QK_SEG;
                    const int y = d*TOPK_BLOCK_SIZE + t_;
                    if (x < y) { bf16 tmp = svbase[sg*seg_elems + x]; svbase[sg*seg_elems + x] = svbase[sg*seg_elems + y]; svbase[sg*seg_elems + y] = tmp; }
                }
            }
            fence_view_async_shared();
        }
        __syncthreads();

        // -------- phase 3: PV (each WG its half of V = [0,256) / [256,512)) --------
        if (warpgroup_idx == 0) {
            Tensor rSv = thr_mma_PV.partition_fragment_A(sS);
            cute::copy(thr_mma_PV.partition_A(sS), rSv);
            CUTE_NO_UNROLL
            for (int seg = 0; seg < 4; ++seg) {
                Tensor sVseg = make_tensor(
                    sV.data() + seg*D_QK_SEG*TOPK_BLOCK_SIZE,
                    Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>, Stride<Int<D_QK_SEG>, _1>>{});
                Tensor rVb = thr_mma_PV.partition_fragment_B(sVseg);
                cute::copy(thr_mma_PV.partition_B(sVseg), rVb);
                cute::gemm(tiled_mma_PV, rSv, rVb, rO[seg]);
            }
        } else {
            // Re-scale WG1's accumulator by this block's scale-for-olds, exactly
            // as if it had run the softmax (mirrors the sm120 sparse kernels).
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
                    sV.data() + (4 + seg)*D_QK_SEG*TOPK_BLOCK_SIZE,
                    Layout<Shape<Int<TOPK_BLOCK_SIZE>, Int<D_QK_SEG>>, Stride<Int<D_QK_SEG>, _1>>{});
                Tensor rVb = thr_mma_PV.partition_fragment_B(sVseg);
                cute::copy(thr_mma_PV.partition_B(sVseg), rVb);
                cute::gemm(tiled_mma_PV, rSv, rVb, rO[seg]);
            }
        }
        __syncthreads();
    }

    // -------- epilogue: per-row o_scales (sm100 semantics) --------
    if (warpgroup_idx == 0) {
        // combine the two rows this thread owns (only rL is merged; rM is
        // already identical across the base-group 4 threads)
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 1);
        rL[0] += __shfl_xor_sync(0xffffffff, rL[0], 2);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 1);
        rL[1] += __shfl_xor_sync(0xffffffff, rL[1], 2);

        if (idx_in_warpgroup%4 == 0) {
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                const int row = get_AorC_row_idx(i, idx_in_warpgroup);
                plan.sL[row] = rL[i];
                plan.sM[row] = rM[i];
            }
        }

        const int row0 = get_AorC_row_idx(0, idx_in_warpgroup);
        const int row1 = get_AorC_row_idx(1, idx_in_warpgroup);
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
            const int row = i == 0 ? row0 : row1;
            const float cur_L = i == 0 ? rL[0] : rL[1];
            const float cur_M = i == 0 ? rM[0] : rM[1];
            const float o_scale = cur_L == 0.0f ? 0.0f : __fdividef(1.0f, cur_L + exp2f(rAttn_sink[i] - cur_M));
            if (idx_in_warpgroup%4 == 0)
                plan.sOScale[row] = o_scale;
        }
    }
    __syncthreads();

    // per-thread final o_scales for its 2 rows
    float o_scales[2];
    CUTE_UNROLL
    for (int i = 0; i < 2; ++i)
        o_scales[i] = plan.sOScale[get_AorC_row_idx(i, idx_in_warpgroup)];

    // ------ conjugate O RoPE (last 64 dims = WG1's PV segment s==3) ------
    if (warpgroup_idx == 1) {
        CUTE_UNROLL
        for (int b = 0; b < 8; ++b)
            CUTE_UNROLL
            for (int lut = 0; lut < 2; ++lut) {
                const int g = 4*b + 2*lut;
                const float2 x = {rO[3](g), rO[3](g+1)};
                const float2 y = apply_conjugate_rope(x, cached_cos[b], cached_sin[b]);
                rO[3](g) = y.x;
                rO[3](g+1) = y.y;
            }
    }

    // ------------------------------------------------------------------
    // O epilogue: per-32-tile absmax -> e4m3 cast + permuted out_fp8/out_sf
    // stores (same layout as the sm100 fused kernel; the fp8 element (h, d)
    // lives at byte offset wv_group*(8*512) + (d/32)*256 + (h%8)*32 + (d%32)
    // of token s_q_idx, and the sf byte at
    //   s_q + wv_group*stride_out_sf_wv_group
    //       + (sf_block/4)*stride_out_sf_head_dim + sf_block%4,
    //   sf_block = (h%8) + (d/32)*8.
    // Absmax is taken AFTER the RoPE (rotation changes the per-tile max), so
    // the conjugate rope above runs first.
    // ------------------------------------------------------------------
    const int t = idx_in_warpgroup;
    const int base = 16*(t/32) + (t%32)/4;
    const uint32_t start_head = (uint32_t)(head_block_idx*BLOCK_M);

    CUTE_NO_UNROLL
    for (int s = 0; s < 4; ++s) {
        // per-thread partial absmax over the 2 rows x 2 (32-col tiles) of this
        // 64-col segment; each base-group lane owns 8 cols of a tile.
        float tile_max[2][2];
        CUTE_UNROLL
        for (int k = 0; k < 2; ++k) { tile_max[0][k] = 0.0f; tile_max[1][k] = 0.0f; }
        CUTE_UNROLL
        for (int g = 0; g < 32; ++g) {
            const int lut = (g%4)/2;
            const int k = (g/4) / 4;   // 0 for cols 0..31, 1 for 32..63
            tile_max[lut][k] = max(tile_max[lut][k], fabsf(rO[s](g)));
        }
        // reduce across the base-group 4 threads (sequential butterfly: first
        // the bit-0 neighbour, then the bit-1 neighbour of the ACCUMULATED max)
        CUTE_UNROLL
        for (int r = 0; r < 2; ++r)
            CUTE_UNROLL
            for (int k = 0; k < 2; ++k) {
                tile_max[r][k] = max(tile_max[r][k], __shfl_xor_sync(0xffffffff, tile_max[r][k], 1));
                tile_max[r][k] = max(tile_max[r][k], __shfl_xor_sync(0xffffffff, tile_max[r][k], 2));
            }

        CUTE_UNROLL
        for (int lut = 0; lut < 2; ++lut) {
            const uint32_t head = start_head + base + 8*lut;
            const uint32_t h_in_wv = head % WV_GROUP_SIZE;
            const uint32_t wv_group = head / WV_GROUP_SIZE;

            CUTE_UNROLL
            for (int k = 0; k < 2; ++k) {
                // sm100 per-32-tile scale computation:
                //   output_abs_max = max(min_clamp, tile_max * o_scale)
                //   sf = output_abs_max / 448; cur_multiplier = o_scale * 2^-exp(sf)
                const float absmax = max(O_QUANT_CLAMP_MIN_VALUE, tile_max[lut][k] * o_scales[lut]);
                const float sf = absmax / 448.0f;
                const uint32_t sf_u = *reinterpret_cast<const uint32_t*>(&sf);
                const int32_t exp_sf = (int32_t)((sf_u - 1) >> 23) + (1 - 127);
                const uint32_t sf_inv_u = (127 - exp_sf) << 23;
                const float sf_inv = *reinterpret_cast<const float*>(&sf_inv_u);
                const float cur_multiplier = o_scales[lut] * sf_inv;

                // store this (row, 32-tile)'s sf byte once
                if (idx_in_warpgroup % 4 == 0) {
                    const uint32_t d32_global = warpgroup_idx*8 + s*2 + k;   // tile index over D_VO=512
                    const uint32_t sf_block = h_in_wv + d32_global * WV_GROUP_SIZE;
                    // leave pointer arithmetic in uint32 element units (matching the
                    // `out_sf` dtype), only the low byte of each sf_block is touched
                    uint8_t* sfp = (uint8_t*)(params.out_sf + s_q_idx
                        + (uint64_t)wv_group*params.stride_out_sf_wv_group
                        + (uint64_t)(sf_block/4)*params.stride_out_sf_head_dim) + (sf_block % 4);
                    *sfp = (uint8_t)(exp_sf + 127);
                }

                // cast + store this thread's 8 cols of the tile (pairs g, g+1)
                CUTE_UNROLL
                for (int b = 0; b < 4; ++b) {
                    const int g = 4*(4*k + b) + 2*lut;
                    const int col = 8*(4*k + b) + 2*(t%4);
                    const int d = warpgroup_idx*256 + s*64 + col;   // global dim
                    const float2 o2 = {rO[s](g)*cur_multiplier, rO[s](g+1)*cur_multiplier};
                    const __nv_fp8x2_storage_t packed = __nv_cvt_float2_to_fp8x2(o2, __NV_SATFINITE, __NV_E4M3);

                    const uint32_t head_dim_idx = d / 32;   // 0..15 (D_VO=512)
                    uint8_t* dst = (uint8_t*)params.out_fp8
                        + (uint64_t)s_q_idx*(CONFIG.H_Q*D_VO)
                        + (uint64_t)wv_group*(WV_GROUP_SIZE*D_VO)
                        + (uint64_t)h_in_wv*32 + (uint64_t)head_dim_idx*256 + (d % 32);
                    *(uint16_t*)dst = *(uint16_t*)&packed;
                }
            }
        }
    }

    // max_logits (prefill only) / lse (natural ln domain; rM/rL in log2 domain).
    if (warpgroup_idx == 0) {
        CUTE_UNROLL
        for (int i = 0; i < 2; ++i) {
            const int row = get_AorC_row_idx(i, idx_in_warpgroup);
            const uint32_t head = start_head + row;
            const float cur_L = plan.sL[row];
            const float cur_M = plan.sM[row];
            if (cur_L == 0.0f) {
                if constexpr (!IS_DECODE)
                    params.max_logits[(int64_t)s_q_idx*params.h_q + head] = -INFINITY;
                params.lse[(int64_t)s_q_idx*params.h_q + head] = INFINITY;
            } else {
                if constexpr (!IS_DECODE)
                    params.max_logits[(int64_t)s_q_idx*params.h_q + head] = cur_M * CUDART_LN2_F;
                params.lse[(int64_t)s_q_idx*params.h_q + head] = logf(cur_L) + cur_M * CUDART_LN2_F;
            }
        }
    }
#endif
}

//--------------------------------------------------------------------------
// Kernel entry / launch (defined once per instantiation .cu).
//--------------------------------------------------------------------------
// Templated on the *type* Kernel (not a Config{} value, as upstream sm100 does):
// a class-type non-type template argument breaks the host-side device-stub
// generation in cudafe1 (template-id parse error); a type argument keeps the
// stub clean.
template<typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1)
fused_norm_rope_attn_rope_cast_fwd_kernel(__grid_constant__ const typename Kernel::Params params) {
    Kernel::devfunc(params);
}

template<Config CONFIG>
void Kernel<CONFIG>::run(const Params &params) {
    static constexpr bool IS_DECODE = is_decode_v<CONFIG.FWD_MODE>;
    static constexpr bool HAS_FP4_KV = CONFIG.EXTRA_MODEL_TYPE == ModelType::V41_FP4;

    KU_ASSERT(params.h_q == (int)H_Q);
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.d_qk == (int)D_QK);
    KU_ASSERT(params.d_v == (int)D_VO);
    KU_ASSERT(params.enable_q_norm == (bool)ENABLE_Q_NORM);
    KU_ASSERT(!params.is_rope_neox_style);
    KU_ASSERT(params.rope_dim == (uint32_t)D_ROPE);
    KU_ASSERT(params.wv_group_size == (uint32_t)WV_GROUP_SIZE);
    KU_ASSERT(params.num_per_channels == (uint32_t)O_QUANT_TILE_SIZE);
    KU_ASSERT(params.use_tma_aligned_col_major_sf == true);
    KU_ASSERT(params.round_sf == true);
    KU_ASSERT(params.use_packed_ue8m0 == true);
    KU_ASSERT(params.n_wv_group == (uint32_t)(H_Q / WV_GROUP_SIZE));
    if constexpr (IS_DECODE) {
        KU_ASSERT(params.b == 1);
        KU_ASSERT(params.enable_split_kv == false);
        if (params.extra_kv != nullptr) {
            // original-region block sizing must keep the unified slot space
            // page-block-aligned (see dequant_k_block_fused)
            KU_ASSERT(params.topk % (HAS_FP4_KV ? 8 : 4) == 0);
        }
    }

    const int num_head_blocks = H_Q / BLOCK_M;
    dim3 grid_dim((long)num_head_blocks*params.s_q, 1, 1);
    auto kernel = &fused_norm_rope_attn_rope_cast_fwd_kernel<Kernel<CONFIG>>;
    cudaError_t sattr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedMemoryPlan));
    if (sattr != cudaSuccess)
        fprintf(stderr, "[sm120 fused core_attn run] setattr failed: %s\n", cudaGetErrorString(sattr));
    kernel<<<grid_dim, NUM_THREADS, sizeof(SharedMemoryPlan), params.stream>>>(params);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[sm120 fused core_attn run] launch failed: %s (grid=%d,%d,%d smem=%zu)\n",
                cudaGetErrorString(err), grid_dim.x, grid_dim.y, grid_dim.z, (size_t)sizeof(SharedMemoryPlan));
}

template<Config CONFIG>
void run_fused_norm_rope_attn_rope_cast_fwd_kernel_sm120(const ParamT<CONFIG.FWD_MODE>& params) {
    Kernel<CONFIG>::run(params);
}

}   // namespace sm120::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn
