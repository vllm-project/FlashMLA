#include "../kernel.cuh"

namespace legacy::sm100::decode::sparse::head64 {

template
void run_flash_splitkv_mla_fp8_sparse_kernel<Config{ModelType::V4, ModelType::V4, false}>(const SparseAttnDecodeParams &params);

}
