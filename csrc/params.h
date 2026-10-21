#pragma once

#include <cstdint>
#include <type_traits>

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA

#include <cutlass/bfloat16.h>
#include <cute/container/alignment.hpp>
using bf16 = cutlass::bfloat16_t;

#else

#include <acl/acl_base_rt.h>
#ifdef __NPU_DEVICE__   // If either __NPU_DEVICE__ and __NPU_HOST__ is defined (which means that this file is included by a `.asc` file), we can use `bfloat16_t` as `bf16`. Otherwise (which means that this file is included by a `.cpp` file) we declare `bf16` as a class to pass compilation
    using bf16 = bfloat16_t;
#else
    #ifdef __NPU_HOST__
        using bf16 = bfloat16_t;
    #else
        struct bf16;
    #endif
#endif

#endif

#if defined(__NPU_DEVICE__) || defined(__ASCC_DEVICE__)
#define FLASH_MLA_GM __gm__
#else
#define FLASH_MLA_GM
#endif

#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
using stream_t = cudaStream_t;
#else
using stream_t = aclrtStream;
#endif

// The KV cache formats. V41 / V41_FP4 store each token's scales right after its raw data
// ("interleaved"); the others keep the raw rows and the scale rows in separate regions of a
// page block. See KVCacheFormat in cuda_kernels/kv_cache_format.h for the byte layouts.
//
// Upstream dropped V32 / V4 in the 2026.09.30 release; we keep them, so the kernels that read
// them live under cuda_kernels/legacy/ (namespace `legacy::`) next to the current V4.1 ones.
enum class ModelType {
    V32,        // DeepSeek V3.2 (d_qk=576)
    V4,         // DeepSeek V4 (d_qk=512)
    V41,        // DeepSeek V4.1 (d_qk=512, RoPE fp8, tile_size=32)
    V41_FP4,    // DeepSeek V4.1 (d_qk=512, fp4 e2m1, tile_size=16, e4m3 scales)
    V32_NVFP4_FP8ROPE   // DeepSeek V3.2 (d_qk=576) with an NVFP4 NoPE part and an fp8 RoPE part
};

enum class SparseAttnFwdMode {
    Prefill,            // Normal prefill mode
    Decode,             // To trigger decoding mode for kernels that support both prefill and decode
    DecodeWithSplitKV   // Decoding with split KV
};

template<SparseAttnFwdMode FWD_MODE>
inline constexpr bool is_decode_v = FWD_MODE == SparseAttnFwdMode::Decode || FWD_MODE == SparseAttnFwdMode::DecodeWithSplitKV;

// Parameters for the sparse attention forward pass kernel (prefill).
struct SparseAttnFwdParams {
    int s_q, s_kv, h_q, h_kv, d_qk, d_v, topk;
    float sm_scale;             // Attention score scaling factor, typically d_qk ** -0.5
    float sm_scale_div_log2;    // sm_scale * log2(e), precomputed for log-base-2 softmax

    // Input tensors
    FLASH_MLA_GM bf16* __restrict__ q;    // [s_q, h_q, d_qk]
    FLASH_MLA_GM bf16* __restrict__ kv;   // [s_kv, h_kv, d_qk]
    FLASH_MLA_GM int* __restrict__ indices;   // [s_q, h_kv, topk], KV token indices for sparse attention
    FLASH_MLA_GM float* __restrict__ attn_sink;   // [h_q], may be nullptr. Per-head attention bias
    FLASH_MLA_GM int* __restrict__ topk_length;    // [s_q], may be nullptr. Actual valid topk count per query

    // Strides (in number of elements, not bytes)
    int stride_q_s_q; int stride_q_h_q;
    int stride_kv_s_kv; int stride_kv_h_kv;
    int stride_indices_s_q; int stride_indices_h_kv;

    // Output tensors
    FLASH_MLA_GM bf16* __restrict__ out;   // [s_q, h_q, d_v], attention output
    FLASH_MLA_GM float* __restrict__ max_logits; // [s_q, h_q], per-head max attention logit (for numerical stability)
    FLASH_MLA_GM float* __restrict__ lse; // [s_q, h_q], log-sum-exp of attention scores

    SparseAttnFwdMode mode;     // Which forward mode to use (Prefill / Decode / DecodeWithSplitKV)
    int num_sm;                 // Number of SMs available for this kernel launch
    stream_t stream;
};

// Tile scheduling metadata for split-KV decoding. Each entry describes
// the work assignment for one SM partition (i.e. a contiguous range of requests and KV blocks).
#ifdef FLASH_MLA_IS_BUILD_ON_CUDA
struct CUTE_ALIGNAS(4*8) DecodingSchedMeta {
#else
struct DecodingSchedMeta {
#endif
    int begin_req_idx, end_req_idx;     // Request range [begin, end], both inclusive
    int begin_block_idx, end_block_idx; // KV block range [begin, end), inclusive-exclusive
    int begin_split_idx;                // Starting split index for output accumulation
    int is_first_req_splitted, is_last_req_splitted; // Whether the first/last request in this partition is split across partitions
    int _pad[1];
};
static_assert(sizeof(DecodingSchedMeta) == 32);
static constexpr int DecodingSchedMetaSize = sizeof(DecodingSchedMeta);

// Parameters for the tile scheduler metadata generation kernel.
// This kernel computes how to partition work across SM partitions for split-KV decoding.
struct GetDecodeSchedMetaParams {
    int b;                              // Batch size
    int s_q;                            // Number of query tokens per request
    int block_size_topk;                // TopK block size used in the decoding kernel
    int fixed_overhead_num_blocks;      // Fixed overhead (in blocks) per request for scheduling
    int num_sm_parts;                   // Number of SM partitions to schedule across
    int topk;                           // Number of selected KV tokens, or -1 for dense attention
    int extra_topk;                     // Extra topk count (0 if extra KV cache is disabled)
    int *__restrict__ topk_length, *__restrict__ extra_topk_length;  // [batch_size], may be nullptr

    // [batch_size]. Only read by the dense scheduler (topk == -1), i.e. by SM90 dense decoding
    int *__restrict__ seqlens_k_ptr;

    // Output
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr; // [num_sm_parts], contiguous
    int* __restrict__ num_splits_ptr; // [batch_size+1], cumulative split counts

    stream_t stream;
};

// Parameters for the sparse attention decoding kernel.
// Decoding uses a paged KV cache, possibly quantized in FP8 format.
struct SparseAttnDecodeParams {
    int b;                      // Batch size (number of requests)
    int s_q;                    // Number of query tokens per request (typically 1 for autoregressive decoding)
    int h_q, h_kv;
    int d_qk, d_v;
    float sm_scale;             // Attention score scaling factor, typically d_qk ** -0.5
    float sm_scale_div_log2;    // sm_scale * log2(e), precomputed for log-base-2 softmax
    int num_blocks;             // Total number of blocks in the paged KV cache
    int page_block_size;        // Number of KV tokens per page block
    int topk;                   // Number of selected KV tokens per query
    ModelType model_type;       // Format of `kv`, see KVCacheFormat in cuda_kernels/kv_cache_format.h
    ModelType extra_model_type; // Format of `extra_kv`: model_type, or V41_FP4 next to a V41 `kv` (see is_valid_kv_format_pair)

    // Input tensors
    FLASH_MLA_GM bf16* __restrict__ q;   // [b, s_q, h_q, d_qk]
    FLASH_MLA_GM bf16* __restrict__ kv;  // [num_blocks, page_block_size, h_kv, bytes_per_token], paged KV cache
    FLASH_MLA_GM int* __restrict__ indices;   // [b, s_q, topk], KV token indices for sparse attention
    FLASH_MLA_GM int* __restrict__ topk_length;  // [b], may be nullptr. Actual valid topk count per request
    FLASH_MLA_GM float* __restrict__ attn_sink;  // [h_q], may be nullptr. Per-head attention bias

    // Output tensors
    FLASH_MLA_GM float* __restrict__ lse;    // [b, s_q, h_q], log-sum-exp of attention scores
    FLASH_MLA_GM bf16* __restrict__ out;   // [b, s_q, h_q, d_v], attention output

    // Extra KV cache — for models with a secondary KV cache
    int extra_num_blocks, extra_page_block_size, extra_topk;
    FLASH_MLA_GM bf16* __restrict__ extra_kv;  // [extra_num_blocks, extra_page_block_size, h_kv, bytes_per_token]
    FLASH_MLA_GM int* __restrict__ extra_indices;   // [b, s_q, extra_topk]
    FLASH_MLA_GM int* __restrict__ extra_topk_length;  // [b], may be nullptr

    // Strides (in number of elements, not bytes)
    int stride_q_b, stride_q_s_q, stride_q_h_q;
    int stride_kv_block, stride_kv_row;
    int stride_indices_b, stride_indices_s_q;
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;
    int stride_extra_kv_block, stride_extra_kv_row;
    int stride_extra_indices_b, stride_extra_indices_s_q;

    int num_sm;                 // Number of SMs available for this kernel launch
    stream_t stream;

    // Split-KV parameters — when enabled, the KV sequence is partitioned across CTAs
    bool enable_split_kv;
    float* __restrict__ lse_accum;  // [num_splits, s_q, h_q], partial LSE per split
    float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v], partial output per split
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;
    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr; // [num_sm_parts], tile scheduling metadata
    int* __restrict__ num_splits_ptr; // [batch_size+1], cumulative split counts per request
    int num_sm_parts;               // Number of SM partitions for split-KV scheduling

    // Row masks (SM90 only) — lets one query tile hold several queries that share a key list.
    // Row r of the tile belongs to slot r / rows_per_slot; bit s of mask[i] says whether slot s attends key i.
    const uint8_t* __restrict__ indices_mask;        // [b, s_q, topk], may be nullptr; same strides as `indices`
    const uint8_t* __restrict__ extra_indices_mask;  // [b, s_q, extra_topk], may be nullptr; same strides as `extra_indices`
    int rows_per_slot;                               // 0 when no row mask is given
};

template<SparseAttnFwdMode FWD_MODE>
using SparseFwdArgT = std::conditional_t<is_decode_v<FWD_MODE>, SparseAttnDecodeParams, SparseAttnFwdParams>;

// =====================================================================================
// Parameters of the kernels under cuda_kernels/legacy/ (CUDA only)
//
// Dense MLA decoding (SM90, DeepSeek V3 / V3.1 / GLM5Next) has no counterpart in the
// current kernels: it reads an unquantized paged KV cache addressed by a block table and
// `seqlens_k` rather than by top-k indices. Its split-KV results go through the legacy
// combine kernel, which is also the only one that handles an fp16 output.
// =====================================================================================

struct DenseAttnDecodeParams {
    using index_t = int64_t;

    int b;              // batch size
    int s_q;
    int q_seq_per_hk;   // The number of q(s) per KV head, = h_q / h_k * s_q
    int d, d_v;         // K/V dimension
    int h_q, h_k;       // The number of Q/K heads
    int num_blocks;     // Number of blocks in total
    int q_head_per_hk;  // The number of q_head(s) per KV head, = h_q / h_k
    bool is_causal;
    float scale_softmax, scale_softmax_log2;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ o_ptr;
    float *__restrict__ softmax_lse_ptr;

    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t o_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t o_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t o_head_stride;

    int *__restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;
    int *__restrict__ seqlens_k_ptr;

    DecodingSchedMeta *__restrict__ tile_scheduler_metadata_ptr;
    int num_sm_parts;
    int *__restrict__ num_splits_ptr;

    int total_num_splits;
    float *__restrict__ softmax_lseaccum_ptr;
    float *__restrict__ oaccum_ptr;

    stream_t stream;
};

struct CombineParams {
    int b, s_q, h_q, d_v;

    float* __restrict__ lse;    // [b, s_q, h_q]
    void* __restrict__ out;   // [b, s_q, h_q, d_v]
    int stride_lse_b, stride_lse_s_q;
    int stride_o_b, stride_o_s_q, stride_o_h_q;

    float* __restrict__ lse_accum;  // [num_splits, s_q, h_q]
    float* __restrict__ o_accum;    // [num_splits, s_q, h_q, d_v]
    int stride_lse_accum_split, stride_lse_accum_s_q;
    int stride_o_accum_split, stride_o_accum_s_q, stride_o_accum_h_q;

    DecodingSchedMeta* __restrict__ tile_scheduler_metadata_ptr; // [num_sm_parts, ], contiguous
    int* __restrict__ num_splits_ptr; // [batch_size+1, ], contiguous
    int num_sm_parts;

    float* attn_sink;  // [h_q], may be nullptr

    stream_t stream;
};

#undef FLASH_MLA_GM
