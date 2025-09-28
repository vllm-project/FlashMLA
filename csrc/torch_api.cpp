#include <Python.h>

#include <torch/nn/functional.h>

#include "pytorch_shim.h"


extern
std::vector<at::Tensor>
get_mla_decoding_metadata(
    at::Tensor &seqlens_k,
    const int num_q_tokens_per_head_k,
    const int h_k,
    const std::optional<int> h_q,
    const bool is_fp8_kvcache,
    const std::optional<int> topk
);

extern
std::vector<at::Tensor>
fwd_kvcache_mla(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size (when is_fp8 is False) or num_blocks x num_heads_k x (page_block_size*656) (when is_fp8 is True)
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits,                // batch_size + 1
    const bool &is_fp8,
    const std::optional<at::Tensor> &indices     // None, or batch_size x seqlen_q x topk
);

extern
std::vector<at::Tensor>
fwd_kvcache_mla_fp8(
    at::Tensor &q,                               // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,                    // num_blocks x page_block_size x num_heads_k x head_size (when is_fp8 is False) or num_blocks x num_heads_k x (page_block_size*656) (when is_fp8 is True)
    const int head_size_v,
    const at::Tensor &seqlens_k,                 // batch_size
    const at::Tensor &block_table,               // batch_size x max_num_blocks_per_seq
    const float softmax_scale,
    bool is_causal,
    const at::Tensor &tile_scheduler_metadata,   // num_sm_parts x TileSchedulerMetaDataSize
    const at::Tensor &num_splits,                // batch_size + 1
    const std::optional<at::Tensor> &descale_q,  // None or batch_size
    const std::optional<at::Tensor> &descale_k   // None or batch_size
);

extern
std::vector<at::Tensor> sparse_prefill_fwd(
    const at::Tensor &q,
    const at::Tensor &kv,
    const at::Tensor &indices,
    float sm_scale,
    int d_v
);


TORCH_LIBRARY(_flashmla_C, m) {
    m.def("get_mla_decoding_metadata", make_pytorch_shim(&get_mla_decoding_metadata));
    m.impl("get_mla_decoding_metadata", torch::kCUDA, make_pytorch_shim(&get_mla_decoding_metadata));

    m.def("fwd_kvcache_mla", make_pytorch_shim(&fwd_kvcache_mla));
    m.impl("fwd_kvcache_mla", torch::kCUDA, make_pytorch_shim(&fwd_kvcache_mla));

    m.def("sparse_prefill_fwd", make_pytorch_shim(&sparse_prefill_fwd));
    m.impl("sparse_prefill_fwd", torch::kCUDA, make_pytorch_shim(&sparse_prefill_fwd));
}

PyMODINIT_FUNC PyInit__flashmla_C() {
    static struct PyModuleDef module = {
        PyModuleDef_HEAD_INIT, "_flashmla_C", nullptr, 0, nullptr};
    return PyModule_Create(&module);
}
