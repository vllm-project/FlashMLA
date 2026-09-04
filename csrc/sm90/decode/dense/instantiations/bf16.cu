#include "../splitkv_mla.cuh"
#include "../splitkv_mla.h"

namespace sm90 {

template void run_flash_splitkv_mla_kernel<cutlass::bfloat16_t, 576>(DenseAttnDecodeParams &params);
template void run_flash_splitkv_mla_kernel<cutlass::bfloat16_t, 512>(DenseAttnDecodeParams &params);

}
