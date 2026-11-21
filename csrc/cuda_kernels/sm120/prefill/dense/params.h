#pragma once

#include <cuda_runtime_api.h>

namespace sm120::prefill::dense {

// Dense prefill (FMHA) launch params. This is the sm120 register-mma
// path corresponding to fmha_cutlass_fwd_sm100 (the sm100 tcgen05 FMHA); the
// tensor semantics mirror the python flash_attn_varlen_func contract:
//   q: [total_q, h_q, d_qk] bf16, contiguous
//   k: [total_kv, h_kv, d_qk] bf16, contiguous
//   v: [total_kv, h_kv, d_v] bf16, contiguous  (d_v == HEAD_DIM_V == 128)
//   o: [total_q, h_q, d_v] bf16, contiguous
//   lse: [total_q, h_q] float (memory layout of torch.empty(h,total).T)
//   cu_seqlens_q/kv: [b+1] int32
// head index of a query row is h_q for o/lse but the KV head is h / (h_q/h_kv)
// (GQA). mask_mode: 0 = residual (no mask), 1 = causal.
// is_varlen is accepted for interface parity; the kernel derives every range
// from cu_seqlens either way.
struct DensePrefillParams {
    void* q;
    void* k;
    void* v;
    void* o;
    void* lse;                  // float, [total_q, h_q]
    const int32_t* cu_seqlens_q;    // [b+1]
    const int32_t* cu_seqlens_kv;   // [b+1]

    int32_t total_q;
    int32_t total_kv;
    int32_t h_q;
    int32_t h_kv;
    int32_t d_qk;
    int32_t d_v;
    int32_t b;

    int32_t mask_mode;          // 0 = residual, 1 = causal
    float softmax_scale;        // multiplication scale, not log2
    cudaStream_t stream;
};

}

