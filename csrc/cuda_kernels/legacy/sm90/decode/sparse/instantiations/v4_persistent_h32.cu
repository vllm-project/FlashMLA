#include "../splitkv_mla.cuh"

namespace legacy::sm90::decode::sparse {

template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V4, 32>(const SparseAttnDecodeParams &params);

}
