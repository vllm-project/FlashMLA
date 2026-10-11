#pragma once

#include "params.h"

namespace legacy::sm90::decode::sparse {

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp8_sparse_kernel(const SparseAttnDecodeParams &params);

}

