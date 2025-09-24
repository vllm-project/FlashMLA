#pragma once

#include "params.h"

namespace sm100 {

void run_flash_splitkv_mla_fp8_sparse_kernel(DecodingParams &params, cudaStream_t stream);

}

