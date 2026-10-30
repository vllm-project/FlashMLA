#include "../splitkv_mla.cuh"

namespace sm120::decode::sparse {

template void run_flash_splitkv_mla_fp8_sparse_kernel<ModelType::V4, 128>(const SparseAttnDecodeParams &params);

}
