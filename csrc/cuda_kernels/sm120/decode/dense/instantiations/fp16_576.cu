#include "../splitkv_mla.cuh"

namespace sm120::decode::dense {

template void run_flash_splitkv_mla_kernel<cutlass::half_t, 576>(const DenseAttnDecodeParams &params);

}
