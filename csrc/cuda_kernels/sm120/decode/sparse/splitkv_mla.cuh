#pragma once

#include "splitkv_mla.h"

#include <cuda_fp8.h>
#include <cstdio>
#include <math_constants.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>

#include <kerutils/kerutils.cuh>

#include "cuda_kernels/sm120/cp_async.cuh"
#include "cuda_kernels/utils.h"
#include "components/dequant.h"
#include "config.h"

using namespace cute;

namespace sm120::decode::sparse {

#ifndef FLASH_MLA_SM120_PHASE_TIME
#define FLASH_MLA_SM120_PHASE_TIME 0
#endif

static constexpr float MAX_INIT_VAL = -1e30;    // Prevent (-inf) - (-inf) = nan
using cutlass::arch::fence_view_async_shared;
using fp8_e8m0 = __nv_fp8_e8m0;

// `ldmatrix` is b8-only in the .trans form (probe 15), and one x2 hands lane l
// rows 4*(l&3)+{0..3} at columns (l>>2) and (l>>2)+8 -- 32 rows x 16 columns per
// warp, 16 B per lane. The address lanes 0-15 feed column group 0 (rows 0-15)
// and lanes 16-31 group 1 (rows 16-31).
__device__ __forceinline__ void ldm_b8x2(uint32_t const *addr, uint32_t (&r)[4]) {
    asm volatile(
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"((uint32_t)__cvta_generic_to_shared(addr)));
}

// Fill one PV B fragment from the fp8 plane: one `ldm_b8x2` per 32 tokens x 16
// dims, then a 4-lane exchange. Lane l holds rows 4*(l&3)+{0..3} while the
// fragment wants lane t to hold rows 2*(t%4)+{0,1} and +8, so the two source
// lanes are (t & ~3) | (j>>1) and (t & ~3) | (2 + (j>>1)), j = t%4, and the
// wanted bytes sit at (j&1)*2 of their word.
template<class TFrag, int TOPK, int SEG, int ROWPAD, int NSC>
__device__ __forceinline__ void pv_dequant_ldsm(TFrag &rVb, const uint8_t* seg_base,
                                                const bf16* sSc, int lane) {
    const int j = lane & 3, grp = lane & ~3;
    const int src[2] = {grp | (j >> 1), grp | (2 + (j >> 1))};
    const int sh = (j & 1)*16;   // bit offset of the byte pair
    CUTE_UNROLL
    for (int twin = 0; twin < TOPK; twin += 32) {
        CUTE_UNROLL
        for (int dwin = 0; dwin < SEG; dwin += 16) {
            uint32_t raw[4];
            ldm_b8x2((const uint32_t*)(seg_base + (twin + lane)*ROWPAD + dwin), raw);
            CUTE_UNROLL
            for (int q = 0; q < 4; ++q) {
                // q = a + 2*s: a picks the dim half (r0/r2 -> +0, r1/r3 -> +8),
                // s picks the token half (r0/r1 -> +0, r2/r3 -> +16).
                const int a = q & 1, s = q >> 1;
                const int tok0 = twin + 16*s + 2*j;
                const int i0 = 4*(dwin/8 + a) + 32*(twin/16 + s);
                uint32_t wa = __shfl_sync(0xffffffff, raw[q], src[0]);
                uint32_t wb = __shfl_sync(0xffffffff, raw[q], src[1]);
                __nv_fp8x2_e4m3 pa, pb;
                *(uint16_t*)&pa = (uint16_t)(wa >> sh);
                *(uint16_t*)&pb = (uint16_t)(wb >> sh);
                float2 fa = (float2)pa, fb = (float2)pb;
                rVb(i0)     = bf16(fa.x) * sSc[(tok0    )*NSC];
                rVb(i0 + 1) = bf16(fa.y) * sSc[(tok0 + 1)*NSC];
                rVb(i0 + 2) = bf16(fb.x) * sSc[(tok0 + 8    )*NSC];
                rVb(i0 + 3) = bf16(fb.y) * sSc[(tok0 + 8 + 1)*NSC];
            }
        }
    }
}

// Element-granular twin of cvt_fp8x8_bf16x8 (same fp8 -> float -> bf16 -> *scale
// order), for the fragment loads where the two k values of an mma B atom pair are
// the only contiguous ones.
__device__ __forceinline__ __nv_bfloat162 cvt_fp8x2_bf16x2(
    const __nv_fp8x2_e4m3 &v, const __nv_bfloat162 &scale) {
    return __float22bfloat162_rn((float2)v) * scale;
}

// bf16x8 -> e4m3x8 with a per-row scale (V32's Q quantization, once per batch).
__device__ __forceinline__ fp8x8 quant_bf16x8(const bf16x8 &v, float qs) {
    float2 a = __bfloat1622float2(v.a01), b = __bfloat1622float2(v.a23);
    float2 c = __bfloat1622float2(v.a45), d = __bfloat1622float2(v.a67);
    fp8x8 o;
    o.lo = __nv_fp8x4_e4m3(make_float4(a.x*qs, a.y*qs, b.x*qs, b.y*qs));
    o.hi = __nv_fp8x4_e4m3(make_float4(c.x*qs, c.y*qs, d.x*qs, d.y*qs));
    return o;
}

// Load the e4m3 B tile `sB8` into the bf16 fragment `rB`, scaling each element
// by its token's scale from `sSc`. `sSc` is laid out like `sB8` in the axis that
// carries the token (n for QK, k for PV). The mma B atom pairs k, so KCONTIG
// (QK: k is the dim axis, contiguous) can fetch a pair with one 2-byte read and
// share one scale; for PV k is the token axis and the pair is K_ROW_PAD apart.
// `rB` is taken by reference: a cute register tensor owns its storage, so a
// by-value parameter would be written into a copy and the caller would keep an
// uninitialized fragment.
template<bool KCONTIG, class TFrag, class TThr, class TSrc8, class TSrcSc>
__device__ __forceinline__ void dequant_b_frag(TFrag &rB, TThr const &thr,
                                               TSrc8 sB8, TSrcSc sSc, int k0 = 0) {
    Tensor src = thr.partition_B(sB8);
    Tensor srcSc = thr.partition_B(sSc);
    if constexpr (KCONTIG) {
        // One n (token) per atom and the tile is one quant tile wide, so one
        // scale serves the atom's two k-pairs. The 16 dims of the tile are 16 B
        // of one row, so a single aligned 16 B read covers every k the atom can
        // ask for (k0 and k0+8, k0 = 2*(lane%4)); the pairs are then cut out of
        // its low and high word instead of costing a load each.
        CUTE_UNROLL
        for (int a = 0; a < (int)cute::size(rB)/4; ++a) {
            int i = 4*a;
            __nv_bfloat162 s = __bfloat162bfloat162(*(const __nv_bfloat16*)&srcSc(i));
            CUTE_UNROLL
            for (int p = 0; p < 2; ++p) {
                int j = i + 2*p;
                *(__nv_bfloat162*)&rB(j) =
                    cvt_fp8x2_bf16x2(*(const __nv_fp8x2_e4m3*)&src(j), s);
            }
        }
    } else {
        // k is the token axis: element i is step i/32, k-slot i%4 (a pair of
        // 2 + 8, see the mma B atom), n-atom i%32/4. One scale serves each
        // k-slot across all 8 n-atoms, so hoist it out of the inner loop.
        static_assert(decltype(cute::size(rB))::value == 128, "expected the 8n x 64k B tile");
        CUTE_UNROLL
        for (int slot = 0; slot < 16; ++slot) {
            int i0 = (slot/4)*32 + slot%4;
            __nv_bfloat16 s = *(const __nv_bfloat16*)&srcSc(i0);
            CUTE_UNROLL
            for (int a = 0; a < 8; ++a) {
                int i = i0 + 4*a;
                *(__nv_bfloat16*)&rB(i) =
                    __float2bfloat16_rn((float)*(const __nv_fp8_e4m3*)&src(i)) * s;
            }
        }
    }
}

// Host-side parameters handed to the kernel through __grid_constant__: only the
// O store descriptor is needed (Q is read directly by the compute threads).
struct TMAParams { CUtensorMap tensor_map_O; };

struct MainloopArgs {
    int start_block_idx, end_block_idx;
    bool is_no_split;

    // The following fields are only valid for DeepSeek-V4
    int topk_length, extra_topk_length, num_orig_kv_blocks;
};

// ---------------------------------------------------------------------------
// Copy one topk block (64 tokens) of K from global memory into the shared K
// plane. All 256 threads take part; each thread owns one token and a 16-dim
// strip of that token's dims.
//
// Nothing is converted: the cache bytes (e4m3 nope + bf16 rope) move as they
// are, so this is a pure cp.async copy and the plane is half the size it was
// when the nope half was dequantized on the way in. The per-token scales land
// in sScales and the dequant happens at the fragment load instead.
// ---------------------------------------------------------------------------
template<ModelType MODEL_TYPE, int NUM_HEADS>
__device__ __forceinline__ void dequant_k_block(
    const SparseAttnDecodeParams &params,
    typename KernelTemplate<MODEL_TYPE, NUM_HEADS>::SharedMemoryPlan &plan,
    int batch_idx,
    int s_q_idx,
    int block_idx,
    const MainloopArgs &args,
    bool use_extra,           // V4 only: block lives in the extra KV cache
    [[maybe_unused]] bool is_first_extra_block
) {
    using Kernel = KernelTemplate<MODEL_TYPE, NUM_HEADS>;
    constexpr int HEAD_DIM_NOPE = Kernel::HEAD_DIM_NOPE;
    constexpr int HEAD_DIM_ROPE = Kernel::HEAD_DIM_ROPE;
    constexpr int NUM_SCALES = Kernel::NUM_SCALES;
    constexpr int TOPK_BLOCK_SIZE = Kernel::TOPK_BLOCK_SIZE;
    constexpr int K_ROW_PAD = Kernel::K_ROW_PAD;

    const int token = threadIdx.x & 63;
    const int tg = threadIdx.x >> 6;    // 0..3 -> one of the 4 dim strips handled per token

    int* gIndices = params.indices + batch_idx*params.stride_indices_b + s_q_idx*params.stride_indices_s_q;
    int* gExtraIndices = params.extra_indices + batch_idx*params.stride_extra_indices_b + s_q_idx*params.stride_extra_indices_s_q;

    int token_index;
    if constexpr (MODEL_TYPE == ModelType::V32) {
        token_index = __ldg(gIndices + block_idx*TOPK_BLOCK_SIZE + token);
    } else {
        if (!use_extra)
            token_index = __ldg(gIndices + block_idx*TOPK_BLOCK_SIZE + token);
        else
            token_index = __ldg(gExtraIndices + (block_idx-args.num_orig_kv_blocks)*TOPK_BLOCK_SIZE + token);
    }

    // For DeepSeek-V4, tokens beyond topk_length are invalid (and may be garbage)
    bool is_valid = token_index != -1;
    if constexpr (MODEL_TYPE == ModelType::V4) {
        int topk_length = use_extra ? args.extra_topk_length : args.topk_length;
        int rel_block_idx = use_extra ? block_idx - args.num_orig_kv_blocks : block_idx;
        is_valid = is_valid && (rel_block_idx*TOPK_BLOCK_SIZE + token < topk_length);
    }

    int page_block_size = use_extra ? params.extra_page_block_size : params.page_block_size;
    int64_t k_block_stride = use_extra ? params.stride_extra_kv_block : params.stride_kv_block;
    int64_t k_row_stride = use_extra ? params.stride_extra_kv_row : params.stride_kv_row;
    fp8* k_ptr = use_extra ? (fp8*)params.extra_kv : (fp8*)params.kv;

    // uint32 div/mod: when token_index == -1 (UINT_MAX), UINT_MAX % page_block_size
    // stays < page_block_size, so no illegal memory access can occur.
    int block_index = token_index == -1 ? 0 : (int)((uint32_t)token_index/(uint32_t)page_block_size);
    int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;

    // V4 note: a token's full region is 584B (asserted below), but the fp8-nope +
    // bf16-rope DATA row is only 576B wide; the 8B e8m0 scales for ALL tokens of
    // a block are clustered in a separate region right after the block's rows
    // (see gK_scales_base). params.stride_kv_row is the 584B token-region stride,
    // so for V4 the nope/rope row step is hardcoded (matches the DeepSeek-V4
    // format and the SM90 kernel).
    fp8* gK_base;
    if constexpr (MODEL_TYPE == ModelType::V4) {
        gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*(HEAD_DIM_NOPE + HEAD_DIM_ROPE*sizeof(bf16));
    } else {
        gK_base = k_ptr + block_index*k_block_stride + rel_idx_in_block*k_row_stride;
    }

    bf16 scales[NUM_SCALES];
    if constexpr (MODEL_TYPE == ModelType::V32) {
        static_assert(NUM_SCALES == 4);
        float scales_float[NUM_SCALES];
        *(float4*)(scales_float) = load_128b_from_gmem<float4, L1CacheHint::EVICT_LAST, L2PrefetchHint::B128>((float*)(gK_base+HEAD_DIM_NOPE));
        CUTE_UNROLL
        for (int i = 0; i < NUM_SCALES; ++i)
            scales[i] = (bf16)scales_float[i];
    } else {
        static_assert(NUM_SCALES == 8);
        fp8_e8m0* gK_scales_base = (fp8_e8m0*)(k_ptr + block_index*k_block_stride + page_block_size*(HEAD_DIM_NOPE+HEAD_DIM_ROPE*sizeof(bf16)) + rel_idx_in_block*NUM_SCALES*sizeof(fp8_e8m0));
        fp8_e8m0 scales_e8m0[NUM_SCALES];
        *(int64_t*)scales_e8m0 = __ldg((int64_t*)gK_scales_base);
        CUTE_UNROLL
        for (int i = 0; i < NUM_SCALES; i += 2) {
            *(__nv_bfloat162_raw*)(scales+i) = __nv_cvt_e8m0x2_to_bf162raw(*(__nv_fp8x2_storage_t*)(scales_e8m0+i));
        }
    }

    plan.is_kv_valid[token] = is_valid;
    if (!is_valid) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_SCALES; ++i)
            scales[i] = (bf16)0.0f;
    }
    CUTE_UNROLL
    for (int i = 0; i < NUM_SCALES; ++i)
        plan.sScales[token*NUM_SCALES + i] = scales[i];

    // Row addressing: dim d of token t lives at t*K_ROW_PAD + k_dim_byte(d). The
    // 16 B pad keeps the tokens apart for the fragment reads and keeps every
    // 16 B chunk of a row aligned.
    auto sk_addr = [&](int dim) -> uint8_t* {
        return plan.u.k.data() + token*K_ROW_PAD + Kernel::k_dim_byte(dim);
    };

    // ----- nope dims (e4m3, byte copy): 8 x (16 dims) per thread -----
    // Alignment: V32 rows are 656 B and V4 data rows 576 B, both 16 B-aligned
    // (and the V4 block stride is a multiple of 576), so 16 B copies are safe.
    // Invalid tokens take the zero-fill form of cp.async.
    fp8* gK_nope = gK_base;
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
        int dim16 = (tg*8 + i)*16;
        if (dim16 >= HEAD_DIM_NOPE) break;
        cp_async16(sk_addr(dim16), gK_nope + dim16, is_valid ? 16 : 0);
    }

    // ----- rope dims (bf16, byte copy): 16 dims per thread -----
    {
        int dim16 = tg*16;   // 0..48 (HEAD_DIM_ROPE == 64, 4 threads x 16)
        bf16* gK_rope;
        if constexpr (MODEL_TYPE == ModelType::V32) {
            // layout: [512 nope fp8][16B fp32 scales][64 rope bf16]
            gK_rope = (bf16*)(gK_base + HEAD_DIM_NOPE + NUM_SCALES*sizeof(float));
        } else {
            // layout: [448 nope fp8][64 rope bf16], scales are NOT in the row
            gK_rope = (bf16*)(gK_base + HEAD_DIM_NOPE);
        }
        CUTE_UNROLL
        for (int j = 0; j < 2; ++j)
            cp_async16(sk_addr(HEAD_DIM_NOPE + dim16 + j*8),
                       gK_rope + dim16 + j*8, is_valid ? 16 : 0);
    }
    // One commit group for the whole block; the block barrier publishes it.
    cp_commit();
    cp_wait<0>();
}

// ---------------------------------------------------------------------------
// Sparse MLA decode kernel, register mma.sync (no GMMA/tcgen05 on sm_120a).
// CTA = 2 warpgroups, 256 threads:
//   WG0: QK (Q re-read from gmem per 64-dim segment) -> mask/softmax -> rS/sS;
//        PV over V dims [0,256) with rS staying in registers (gemm_rs).
//   WG1: PV over V dims [256,512) with rS re-read from sS (gemm_ss).
// Single KV buffer: K is dequantized fp8 -> bf16 into (SEG,TOKEN,COL64) with a
// K_ROW_PAD stride, V is that buffer's nope part read dim-major by
// ldmatrix.trans (no copy, no transpose). Sync is plain __syncthreads() between
// the phases; dual buffering is still open (it needs the fp8 plane to fit).
// ---------------------------------------------------------------------------
template<ModelType MODEL_TYPE, int NUM_HEADS>
template<typename TMAParams>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS>::devfunc(const SparseAttnDecodeParams &params, const TMAParams &tma_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300)) || (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
    const int head_block_idx = NUM_M_BLOCKS == 1 ? 0 : blockIdx.x;
    const int s_q_idx = blockIdx.y;
    const int partition_idx = blockIdx.z;
    const int warpgroup_idx = cutlass::canonical_warp_group_idx();
    const int idx_in_warpgroup = threadIdx.x % 128;
    const int warp_idx = cutlass::canonical_warp_idx_sync();

    extern __shared__ char wksp_buf[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(wksp_buf);
    uint8_t* sKbytes = plan.u.k.data();   // raw cache bytes: [nope e4m3][rope bf16]
    Tensor sS = make_tensor(make_smem_ptr(plan.s.data()), SmemLayoutS{});
    Tensor sOBuf = make_tensor(make_smem_ptr(plan.u.oBuf.data()), SmemLayoutOBuf{});
    float* sM = plan.sM;
    float* sL = plan.sL;
    float* sScale = plan.sScale;

    // split-KV is optional (enable_split_kv only when topk+extra_topk>640,
    // like sm100; sm90 always split). Without it num_sm_parts stays 0 and no
    // tile_scheduler_metadata exists: each (head_block, s_q) CTA covers the full KV
    // range, so synthesize a full-range schedule instead of dereferencing nullptr.
    bool is_split = params.tile_scheduler_metadata_ptr != nullptr;
    DecodingSchedMeta sched_meta;
    if (is_split) {
        sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];
        if (sched_meta.begin_req_idx >= params.b) return;
    } else {
        sched_meta = DecodingSchedMeta{};
        sched_meta.begin_req_idx = 0;
        sched_meta.end_req_idx = params.b - 1;
        sched_meta.begin_block_idx = 0;
        sched_meta.is_first_req_splitted = false;
        sched_meta.is_last_req_splitted = false;
    }

    auto get_cur_req_info = [&](int batch_idx) -> MainloopArgs {
        MainloopArgs args;
        int total_topk_padded;
        if constexpr (MODEL_TYPE == ModelType::V32) {
            total_topk_padded = params.topk;
        } else {
            int topk_length = params.topk_length ? __ldg(params.topk_length + batch_idx) : params.topk;
            int orig_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
            int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
            total_topk_padded = orig_topk_padded + ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
            args.topk_length = topk_length;
            args.extra_topk_length = extra_topk_length;
            args.num_orig_kv_blocks = orig_topk_padded / TOPK_BLOCK_SIZE;
        }

        const int total_num_blocks = total_topk_padded / TOPK_BLOCK_SIZE;
        args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
        args.end_block_idx = batch_idx == sched_meta.end_req_idx ? (is_split ? sched_meta.end_block_idx : total_num_blocks) : total_num_blocks;
        args.is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

        return args;
    };

    // Q tile of this CTA from global memory (re-read per block/segment; no smem Q)
    auto gQ_full = make_tensor(
        make_gmem_ptr((bf16*)params.q),
        make_layout(
            make_shape(params.h_q, params.d_qk, params.s_q, params.b),
            make_stride(params.stride_q_h_q, _1{}, params.stride_q_s_q, params.stride_q_b)
        )
    );

    TiledMMA_QK tiled_mma_QK{};
    auto thr_mma_QK = tiled_mma_QK.get_slice(idx_in_warpgroup);
    TiledMMA_QK8 tiled_mma_QK8{};
    auto thr_mma_QK8 = tiled_mma_QK8.get_slice(idx_in_warpgroup);
    TiledMMA_PV_LocalP tiled_mma_PV{};
    auto thr_mma_PV = tiled_mma_PV.get_slice(idx_in_warpgroup);

    // PV's B operand is V read DIM-major (K axis = token, stride 1). ldmatrix.trans
    // reads the stored (token, dim) plane through a transposed view and returns
    // the transposed fragment directly, so the per-block in-place transpose is
    // gone. U16x4 (x2) covers exactly one 8n x 16k mma B atom, which is what
    // make_tiled_copy_B requires; retile_D reinterprets the fragment.
    auto ldsm_PV = make_tiled_copy_B(Copy_Atom<SM75_U16x4_LDSM_T, bf16>{}, tiled_mma_PV);
    auto thr_ldsm_PV = ldsm_PV.get_slice(idx_in_warpgroup);
    // element (d,t) lives at t*K_ROW_PAD + d bytes in the raw plane; the scale of
    // a whole 64-dim segment is one value per token (V4's last segment is rope,
    // in bf16, and takes the ldmatrix path instead).
    static constexpr auto kTransposedView8 =
        Layout<Shape<Int<D_QK_SEG>, Int<TOPK_BLOCK_SIZE>>, Stride<_1, Int<K_ROW_PAD>>>{};
    static constexpr auto kTransposedViewBf16 =
        Layout<Shape<Int<D_QK_SEG>, Int<TOPK_BLOCK_SIZE>>, Stride<_1, Int<K_ROW_PAD_BF16>>>{};
    static constexpr auto kScaleTransposedView =
        Layout<Shape<Int<D_QK_SEG>, Int<TOPK_BLOCK_SIZE>>, Stride<_0, Int<NUM_SCALES>>>{};

    float rL[2], rM[2];
    // PV output: 4 independent 64-col C fragments (one 64x64
    // C per V N-segment; a single (64,256) C cannot be N-sliced by local_tile).
    using ROType = decltype(partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}));
    ROType rO[4] = {
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
        partition_fragment_C(tiled_mma_PV, Shape<Int<BLOCK_M>, _64>{}),
    };

    // One PV segment: V is the fp8 nope plane, dequantized at the fragment load;
    // V4's last segment is the rope plane and keeps the ldmatrix.trans read.
    auto pv_seg = [&](auto const &rSv, int abs_seg, int local_seg) {
        if (abs_seg*D_QK_SEG + D_QK_SEG <= HEAD_DIM_NOPE) {
            Tensor sB = make_tensor(sKbytes + abs_seg*D_QK_SEG, kTransposedView8);
            Tensor rVb = thr_mma_PV.partition_fragment_B(sB);
            pv_dequant_ldsm<decltype(rVb), TOPK_BLOCK_SIZE, D_QK_SEG, K_ROW_PAD, NUM_SCALES>(
                rVb, sKbytes + abs_seg*D_QK_SEG,
                plan.sScales + abs_seg*D_QK_SEG/QUANT_TILE_SIZE, idx_in_warpgroup % 32);
            cute::gemm(tiled_mma_PV, rSv, rVb, rO[local_seg]);
        } else {
            Tensor sB = make_tensor((bf16*)(sKbytes + HEAD_DIM_NOPE), kTransposedViewBf16);
            Tensor rVb = thr_mma_PV.partition_fragment_B(sB);
            cute::copy(ldsm_PV, thr_ldsm_PV.partition_S(sB), thr_ldsm_PV.retile_D(rVb));
            cute::gemm(tiled_mma_PV, rSv, rVb, rO[local_seg]);
        }
    };

    Tensor rP = partition_fragment_C(tiled_mma_QK, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
    Tensor rS = make_tensor<bf16>(partition_shape_A(tiled_mma_PV, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{}));

    float rAttn_sink[2] = {-CUDART_INF_F, -CUDART_INF_F};
    if (params.attn_sink != nullptr) {
        for (int i = 0; i < 2; ++i) {
            int head_idx = head_block_idx*BLOCK_M + get_AorC_row_idx(i, idx_in_warpgroup);
            rAttn_sink[i] = __ldg((float*)params.attn_sink + head_idx) * CUDART_L2E_F;
        }
    }

#if FLASH_MLA_SM120_PHASE_TIME
    // Diagnostic only (compile with -DFLASH_MLA_SM120_PHASE_TIME=1): per-block
    // clock64() deltas for the page copy, QK+softmax, the barrier that separates
    // them from PV, and PV. Block 0 prints its totals once at the end.
    unsigned long long ph_copy = 0, ph_qk = 0, ph_bar = 0, ph_pv = 0, t_prev = 0;
    int pages = 0;
    if (threadIdx.x == 0) t_prev = clock64();
#endif
    #pragma unroll 1
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        MainloopArgs args = get_cur_req_info(batch_idx);

        rL[0] = rL[1] = 0.0f;
        rM[0] = rM[1] = MAX_INIT_VAL;
        CUTE_UNROLL
        for (int s = 0; s < 4; ++s)
            cute::fill(rO[s], 0.f);

        Tensor gQ = flat_divide(
            gQ_full(_, _, s_q_idx, batch_idx),
            Tile<Int<BLOCK_M>, Int<HEAD_DIM_K>>{}
        )(_, _, head_block_idx, _0{});   // (BLOCK_M, HEAD_DIM_K)

        // -------- Q to e4m3, once per batch --------
        // One scale per row over the whole nope half (the probe says granularity
        // is not the lever), so the scale is a single multiply folded into rP
        // after the QK loop and the plane needs no per-tile bookkeeping. The rope
        // dims stay bf16 and keep the bf16 mma.
        {
            {
                const int row = threadIdx.x & 63;
                CUTE_NO_UNROLL
                for (int tile = threadIdx.x >> 6; tile < HEAD_DIM_NOPE/QUANT_TILE_SIZE; tile += 4) {
                    const bf16* src = &gQ(row, tile*QUANT_TILE_SIZE);
                    float amax = 0.f;
                    CUTE_UNROLL
                    for (int i = 0; i < QUANT_TILE_SIZE; ++i)
                        amax = fmaxf(amax, fabsf((float)src[i]));
                    const bf16 deq = amax > 0.f ? (bf16)(amax/448.f) : (bf16)1.f;
                    plan.sQscales[row*NUM_SCALES + tile] = deq;
                    const float qs = 1.f/(float)deq;
                    fp8* dst = (fp8*)plan.q.data() + (uint64_t)row*Q_ROW_PAD + tile*QUANT_TILE_SIZE;
                    CUTE_UNROLL
                    for (int c = 0; c < QUANT_TILE_SIZE/8; ++c) {
                        bf16x8 v = load_128b_from_gmem<bf16x8, L1CacheHint::EVICT_LAST, L2PrefetchHint::B256>(src + 8*c);
                        fp8x8 o = quant_bf16x8(v, qs);
                        *(uint64_t*)(dst + 8*c) = *(uint64_t*)&o;
                    }
                }
            }
            __syncthreads();
        }

        CUTE_NO_UNROLL
        for (int block_idx = args.start_block_idx; block_idx < args.end_block_idx; ++block_idx) {
            // -------- phase 1: dequant K (all 256 threads) --------
            if constexpr (MODEL_TYPE == ModelType::V32) {
                dequant_k_block<MODEL_TYPE, NUM_HEADS>(params, plan, batch_idx, s_q_idx, block_idx, args, false, false);
            } else {
                if (block_idx < min(args.num_orig_kv_blocks, args.end_block_idx))
                    dequant_k_block<MODEL_TYPE, NUM_HEADS>(params, plan, batch_idx, s_q_idx, block_idx, args, false, false);
                else
                    dequant_k_block<MODEL_TYPE, NUM_HEADS>(params, plan, batch_idx, s_q_idx, block_idx, args, true, true);
            }
            __syncthreads();
#if FLASH_MLA_SM120_PHASE_TIME
            if (threadIdx.x == 0) {
                unsigned long long t = clock64();
                ph_copy += t - t_prev; t_prev = t; ++pages;
            }
#endif

            // -------- phase 2: QK, dim steps split across the warpgroups --------
            // WG1 idled through the whole QK phase. Both warpgroups now run the
            // same 64x64 tile over disjoint halves of the k steps (a step never
            // straddles nope/rope, HEAD_DIM_NOPE being a multiple of 16, so the
            // split point is free); WG1 publishes its partial to sP and WG0 adds
            // it before the mask/softmax - the only cross-warpgroup reduction.
            {
                constexpr int QK_SPLIT = (HEAD_DIM_K/16) / 2;
                const int k_begin = (warpgroup_idx == 0 ? 0 : QK_SPLIT) * 16;
                const int k_end = (warpgroup_idx == 0 ? QK_SPLIT : HEAD_DIM_K/16) * 16;
                cute::clear(rP);
                // The nope dims run the fp8 atom, 32 dims at a time: both
                // fragments are filled with raw 4-byte reads out of the planes
                // (see TiledMMA_QK8). The token scale is applied to rP, but only
                // when the 128-dim quant tile changes - and once more before the
                // rope dims, which carry no scale. The row's Q scale is one
                // multiply after the loop. The rope segment keeps the bf16 atom
                // and its Q from gmem.
                const int qk_lane = idx_in_warpgroup % 32;
                const int qk_row = 16*(idx_in_warpgroup/32) + qk_lane/4;
                const int qk_k0 = 4*(qk_lane % 4);
                auto apply_k_tile_scale = [&](int tile) {
                    const bf16* sc = plan.sScales + tile;
                    const float sq0 = (float)plan.sQscales[qk_row*NUM_SCALES + tile];
                    const float sq1 = (float)plan.sQscales[(qk_row + 8)*NUM_SCALES + tile];
                    CUTE_UNROLL
                    for (int nb = 0; nb < TOPK_BLOCK_SIZE/8; ++nb)
                        CUTE_UNROLL
                        for (int lut = 0; lut < 2; ++lut)
                            CUTE_UNROLL
                            for (int cp = 0; cp < 2; ++cp)
                                rP(4*nb + 2*lut + cp) *= (float)sc[
                                    (8*nb + 2*(idx_in_warpgroup%4) + cp)*NUM_SCALES] * (lut ? sq1 : sq0);
                };
                int pending_tile = -1;
                CUTE_NO_UNROLL
                for (int k = k_begin; k < k_end; k += 32) {
                    uint8_t* p = sKbytes + k_dim_byte(k);
                    if (k + 32 <= HEAD_DIM_NOPE) {
                        if ((int)(k/QUANT_TILE_SIZE) != pending_tile) {
                            if (pending_tile >= 0) apply_k_tile_scale(pending_tile);
                            pending_tile = k/QUANT_TILE_SIZE;
                        }
                        // A: registers are (row, k0+{0..3}) and the same at +16,
                        // for rows qk_row and qk_row+8.
                        Tensor sQk = make_tensor(
                            (fp8*)plan.q.data() + k,
                            Layout<Shape<Int<BLOCK_M>, _32>, Stride<Int<Q_ROW_PAD>, _1>>{});
                        Tensor rQa = thr_mma_QK8.partition_fragment_A(sQk);
                        const uint8_t* qa = plan.q.data() + (uint64_t)qk_row*Q_ROW_PAD + k + qk_k0;
                        *(uint32_t*)&rQa(0)  = *(const uint32_t*)(qa);
                        *(uint32_t*)&rQa(4)  = *(const uint32_t*)(qa + 8*Q_ROW_PAD);
                        *(uint32_t*)&rQa(8)  = *(const uint32_t*)(qa + 16);
                        *(uint32_t*)&rQa(12) = *(const uint32_t*)(qa + 8*Q_ROW_PAD + 16);
                        Tensor sB = make_tensor(
                            (fp8*)p, Layout<Shape<Int<TOPK_BLOCK_SIZE>, _32>, Stride<Int<K_ROW_PAD>, _1>>{});
                        Tensor rKb = thr_mma_QK8.partition_fragment_B(sB);
                        const uint8_t* kb = p + (uint64_t)(qk_lane/4)*K_ROW_PAD + qk_k0;
                        CUTE_UNROLL
                        for (int a = 0; a < TOPK_BLOCK_SIZE/8; ++a) {
                            *(uint32_t*)&rKb(8*a)     = *(const uint32_t*)(kb + 8*a*K_ROW_PAD);
                            *(uint32_t*)&rKb(8*a + 4) = *(const uint32_t*)(kb + 8*a*K_ROW_PAD + 16);
                        }
                        cute::gemm(tiled_mma_QK8, rQa, rKb, rP);
                    } else {
                        // rope: bf16, two 16-dim steps to keep the 32-dim loop stride.
                        // It has no scale of its own, so the pending one goes in first.
                        if (pending_tile >= 0) { apply_k_tile_scale(pending_tile); pending_tile = -1; }
                        CUTE_UNROLL
                        for (int kk = k; kk < k + 32; kk += 16) {
                            Tensor gQk = local_tile(gQ, Shape<Int<BLOCK_M>, _16>{}, make_coord(_0{}, kk/16));
                            Tensor rQa = thr_mma_QK.partition_fragment_A(gQk);
                            cute::copy(thr_mma_QK.partition_A(gQk), rQa);
                            Tensor sB = make_tensor(
                                (bf16*)(sKbytes + k_dim_byte(kk)),
                                Layout<Shape<Int<TOPK_BLOCK_SIZE>, _16>, Stride<Int<K_ROW_PAD_BF16>, _1>>{});
                            Tensor rKb = thr_mma_QK.partition_fragment_B(sB);
                            cute::copy(thr_mma_QK.partition_B(sB), rKb);
                            cute::gemm(tiled_mma_QK, rQa, rKb, rP);
                        }
                    }
                }
                if (pending_tile >= 0) apply_k_tile_scale(pending_tile);
            }
            __syncthreads();
            // rP registers carry the C-fragment (row, col) map of save_rS_to_sS
            const int prow_base = 16*(idx_in_warpgroup/32) + (idx_in_warpgroup%32)/4;
            if (warpgroup_idx == 1) {
                CUTE_UNROLL
                for (int nb = 0; nb < TOPK_BLOCK_SIZE/8; ++nb)
                    CUTE_UNROLL
                    for (int lut = 0; lut < 2; ++lut)
                        CUTE_UNROLL
                        for (int cp = 0; cp < 2; ++cp)
                            plan.sP[(prow_base + 8*lut)*TOPK_BLOCK_SIZE + 8*nb + 2*(idx_in_warpgroup%4) + cp] =
                                (__half)rP(4*nb + 2*lut + cp);
            }
            __syncthreads();
            if (warpgroup_idx == 0) {
                CUTE_UNROLL
                for (int nb = 0; nb < TOPK_BLOCK_SIZE/8; ++nb)
                    CUTE_UNROLL
                    for (int lut = 0; lut < 2; ++lut)
                        CUTE_UNROLL
                        for (int cp = 0; cp < 2; ++cp)
                            rP(4*nb + 2*lut + cp) +=
                                (float)plan.sP[(prow_base + 8*lut)*TOPK_BLOCK_SIZE + 8*nb + 2*(idx_in_warpgroup%4) + cp];

                // softmax on segment 0; every segment rescales its own 64 cols
                float scale_for_olds[2];
                CUTE_UNROLL
                for (int s = 0; s < 4; ++s)
                    scale_softmax(rP, rS, rO[s], params.sm_scale_div_log2, sScale, rM, rL, plan.is_kv_valid, block_idx, idx_in_warpgroup, scale_for_olds, s == 0);

                save_rS_to_sS(rS, sS, idx_in_warpgroup);
                fence_view_async_shared();
#if FLASH_MLA_SM120_PHASE_TIME
                if (threadIdx.x == 0) {
                    unsigned long long t = clock64();
                    ph_qk += t - t_prev; t_prev = t;
                }
#endif
            }
            __syncthreads();
#if FLASH_MLA_SM120_PHASE_TIME
            if (threadIdx.x == 0) {
                unsigned long long t = clock64();
                ph_bar += t - t_prev; t_prev = t;
            }
#endif

            // -------- phase 3: PV (each WG its half of V; V = K's nope, no dequant) --------
            if (warpgroup_idx == 0) {
                // gemm_ss: rS re-read from sS (layout-matched A fragment),
                // same as WG1. rS-in-registers (gemm_rs) cannot be used: scale_softmax
                // fills rS with C-fragment indices but cute::gemm decodes the A operand
                // with the MMA A-fragment layout, which pairs a different (row,token)
                // set; both warpgroups must go through sS.
                Tensor rSv = thr_mma_PV.partition_fragment_A(sS);
                cute::copy(thr_mma_PV.partition_A(sS), rSv);
                CUTE_NO_UNROLL
                for (int seg = 0; seg < 4; ++seg)
                    pv_seg(rSv, seg, seg);
            } else {
                // scale historical O by this block's scale_for_old, then PV over V dims [256,512)
                float cur_scales[2];
                *(float2*)cur_scales = *(float2*)(sScale + (idx_in_warpgroup/4)*2);
                CUTE_UNROLL
                for (int s = 0; s < 4; ++s)
                    CUTE_UNROLL
                    for (int g = 0; g < (int)size(rO[s]); ++g)
                        rO[s](g) *= cur_scales[(g%4)/2];

                Tensor rSv = thr_mma_PV.partition_fragment_A(sS);
                cute::copy(thr_mma_PV.partition_A(sS), rSv);
                CUTE_NO_UNROLL
                for (int seg = 0; seg < 4; ++seg)
                    pv_seg(rSv, 4 + seg, seg);
            }
            __syncthreads();
#if FLASH_MLA_SM120_PHASE_TIME
            if (threadIdx.x == 0) {
                unsigned long long t = clock64();
                ph_pv += t - t_prev; t_prev = t;
            }
#endif
        }

#if FLASH_MLA_SM120_PHASE_TIME
        if (blockIdx.x == 0 && blockIdx.z == 0 && threadIdx.x == 0)
            printf("PHASE pages=%d copy=%llu qk=%llu tr=%llu pv=%llu\n",
                   pages, ph_copy, ph_qk, ph_bar, ph_pv);
#endif
        // -------- epilogue (per batch) --------
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
                    sL[row] = rL[i];
                    sM[row] = rM[i];
                }
            }

            // final o_scales for THIS wg (and published to WG1 through sOScale)
            int row0 = get_AorC_row_idx(0, idx_in_warpgroup);
            int row1 = get_AorC_row_idx(1, idx_in_warpgroup);
            float o_scales[2];
            CUTE_UNROLL
            for (int i = 0; i < 2; ++i) {
                int row = i == 0 ? row0 : row1;
                float cur_L = (i == 0 ? rL[0] : rL[1]);
                float cur_M = (i == 0 ? rM[0] : rM[1]);
                if (args.is_no_split) {
                    o_scales[i] = cur_L == 0.0f ? 0.0f : __fdividef(1.0f, cur_L + exp2f(rAttn_sink[i] - cur_M));
                } else {
                    o_scales[i] = cur_L == 0.0f ? 0.0f : __fdividef(1.0f, cur_L);
                }
                if (idx_in_warpgroup%4 == 0)
                    plan.sOScale[row] = o_scales[i];
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
        if (args.is_no_split) {
            bf16* o_ptr = (bf16*)params.out + batch_idx*params.stride_o_b + s_q_idx*params.stride_o_s_q + start_head_idx*params.stride_o_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_h_q, 1)
            Tensor gO = make_tensor(make_gmem_ptr(o_ptr), make_layout(
                Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                make_stride(params.stride_o_h_q, _1{})
            ));
            // stage own O half into oBuf (unions with k; K/V already dead)
            // cute::copy(rO, partition_fragment_C(sMySeg)) is broken here:
            // the target fragment lands in LOCAL (%STL) instead of shared memory,
            // so the copy silently writes nowhere and TMA would ship K remnants.
            // Write each (row,col) explicitly through the SW128 tensor instead.
            // Fragment map: 2 rows {base, base+8} per thread,
            // base = 16*(t/32)+(t%32)/4; col = 8*(g/4)+2*(t%4)+(g%4)%2; lut=(g%4)/2.
            const int t = idx_in_warpgroup;
            const int base = 16*(t/32) + (t%32)/4;
            // one half of V at a time, own warpgroup's half staged by itself
            CUTE_UNROLL
            for (int h = 0; h < 2; ++h) {
                if (warpgroup_idx == h) {
                    CUTE_UNROLL
                    for (int s = 0; s < 4; ++s) {
                        Tensor sMySeg = local_tile(sOBuf, Shape<_64, _64>{}, make_coord(_0{}, s));
                        CUTE_UNROLL
                        for (int g = 0; g < 32; ++g) {
                            int lut = (g%4)/2, cp = (g%4)%2;
                            int row = base + 8*lut;
                            int col = 8*(g/4) + 2*(t%4) + cp;
                            sMySeg(row, col) = (bf16)(rO[s](g) * o_scales[lut]);
                        }
                    }
                }
                fence_view_async_shared();
                __syncthreads();

                if (threadIdx.x == 0) {
                    SM90_TMA_STORE_5D::copy(
                        &tma_params.tensor_map_O,
                        plan.u.oBuf.data(),
                        0, head_block_idx*64, h*(HEAD_DIM_V/2/OBUF_SW),
                        s_q_idx, batch_idx
                    );
                    cute::tma_store_arrive();
                    cute::tma_store_wait<0>();
                }
                __syncthreads();
            }

            if (warpgroup_idx == 0) {
                float* gSoftmaxLse = (float*)params.lse + batch_idx*params.stride_lse_b + s_q_idx*params.stride_lse_s_q + start_head_idx;	// (BLOCK_M) : (1)
                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLse[i] = cur_L == 0.0f ? INFINITY : logf(cur_L) + sM[i] / (float)M_LOG2E;
                }
            }
            cute::tma_store_wait<0>();
        } else {
            int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
            int split_idx = __ldg(params.num_splits_ptr+batch_idx) + n_split_idx;
            float* oaccum_ptr = (float*)params.o_accum + split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + start_head_idx*params.stride_o_accum_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
            Tensor gOAccum = make_tensor(make_gmem_ptr(oaccum_ptr), make_layout(
                Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                make_stride(params.stride_o_accum_h_q, _1{})
            ));
            // direct scatter to the fp32 global accumulator (no oAccumBuf on sm120);
            // 4 N-segments: each C fragment is a 64-col block (row/col map per segment)
            CUTE_UNROLL
            for (int s = 0; s < 4; ++s) {
                CUTE_UNROLL
                for (int idx = 0; idx < size(rO[s]); idx += 2) {
                    int row = (idx_in_warpgroup/32)*16 + (idx_in_warpgroup%32/4) + (idx%4 >= 2 ? 8 : 0);
                    int col = warpgroup_idx*256 + s*64 + (idx_in_warpgroup%4)*2 + idx/4*8;
                    *(float2*)(&gOAccum(row, col)) = float2 {
                        rO[s](idx) * o_scales[idx%4>=2],
                        rO[s](idx+1) * o_scales[idx%4>=2],
                    };
                }
            }

            if (warpgroup_idx == 0) {
                float* gSoftmaxLseAccum = (float*)params.lse_accum + split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + start_head_idx;	// (BLOCK_M) : (1)
                int i = threadIdx.x;
                if (i < num_valid_seq_q) {
                    float cur_L = sL[i];
                    gSoftmaxLseAccum[i] = cur_L == 0.0f ? -INFINITY : log2f(cur_L) + sM[i];
                }
            }
        }
        __syncthreads();
    }
#endif
}

template<typename Kernel, typename TMAParams>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1)
flash_fwd_splitkv_mla_fp8_sparse_kernel(__grid_constant__ const SparseAttnDecodeParams params, __grid_constant__ const TMAParams tma_params) {
    Kernel::devfunc(params, tma_params);
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
    KU_ASSERT(params.d_qk == HEAD_DIM_K);
    KU_ASSERT(params.d_v == HEAD_DIM_V);
    KU_ASSERT(params.h_q % BLOCK_M == 0);
    if constexpr (MODEL_TYPE == ModelType::V4) {
        constexpr int BYTES_PER_TOKEN = HEAD_DIM_NOPE + 2*HEAD_DIM_ROPE + 8;
        KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN, "Each page block in KV cache must be contiguous for head64 sparse fp8 decoding attention in DeepSeek-V4");  // Each block must be contiguous
        if (params.extra_kv != nullptr) {
            KU_ASSERT(params.stride_extra_kv_row == BYTES_PER_TOKEN, "Each page block in extra KV cache must be contiguous for head64 sparse fp8 decoding attention in DeepSeek-V4");  // Each block must be contiguous
        }
    } else {
        KU_ASSERT(params.extra_kv == nullptr, "V3.2 does not support extra KV cache");
        KU_ASSERT(params.topk_length == nullptr, "V3.2 does not support dynamic topk length");
        KU_ASSERT(params.stride_kv_row == 656);  // number of bytes per token (512 fp8 + 4 float32 + 64 bfloat16)
    }

    // Only the O store descriptor is needed (no Q TMA on sm120: Q is read directly
    // from global memory by the compute threads).
    CUtensorMap tensor_map_O;
    {
        // 5D TMA to store O
        uint64_t size[5] = {OBUF_SW, (unsigned long)params.h_q, HEAD_DIM_V/OBUF_SW, (unsigned long)params.s_q, (unsigned long)params.b};
        uint64_t stride[4] = {params.stride_o_h_q*sizeof(bf16), OBUF_SW*sizeof(bf16), params.stride_o_s_q*sizeof(bf16), params.stride_o_b*sizeof(bf16)};
        uint32_t box_size[5] = {OBUF_SW, BLOCK_M, HEAD_DIM_V/OBUF_SW/2, 1, 1};
        uint32_t elem_stride[5] = {1, 1, 1, 1, 1};
        CUresult res = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
            &tensor_map_O,
            CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            5,
            params.out,
            size,
            stride,
            box_size,
            elem_stride,
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            OBUF_SW == 64 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B :
                OBUF_SW == 32 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B :
                OBUF_SW == 16 ? CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B :
                CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        KU_ASSERT(res == CUresult::CUDA_SUCCESS);
    }

    TMAParams tma_params{tensor_map_O};

    auto kernel = &flash_fwd_splitkv_mla_fp8_sparse_kernel<KernelTemplate<MODEL_TYPE, NUM_HEADS>, TMAParams>;

    // opt-in the maximally allowed dynamic shared memory (101376B);
    // our plan (~83.5KB) fits within it.
    {
        cudaError_t sattr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);
        if (sattr != cudaSuccess) {
            fprintf(stderr, "[sm120 run] setattr failed: %s\n", cudaGetErrorString(sattr));
        }
    }

    // Non-split-KV leaves num_sm_parts == 0; clamp to 1 (one CTA covers the full
    // KV range, see devfunc's synthesized full-range schedule).
    int num_sm_parts = params.num_sm_parts > 0 ? params.num_sm_parts : 1;
    dim3 grid_dim(NUM_M_BLOCKS, params.s_q, num_sm_parts);
    kernel<<<grid_dim, NUM_THREADS, sizeof(SharedMemoryPlan), params.stream>>>(params, tma_params);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[sm120 run] launch failed: %s (grid=(%d,%d,%d) smem=%zu num_threads=%d)\n",
                cudaGetErrorString(err), grid_dim.x, grid_dim.y, grid_dim.z,
                (size_t)sizeof(SharedMemoryPlan), NUM_THREADS);
    }
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params) {
    KernelTemplate<MODEL_TYPE, NUM_HEADS>::run(params);
}

}
