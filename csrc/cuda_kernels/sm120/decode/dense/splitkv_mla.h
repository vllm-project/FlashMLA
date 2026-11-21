#pragma once

#include "params.h"

namespace sm120::decode::dense {

template<typename InputT, int HEAD_DIM_K>
void run_flash_splitkv_mla_kernel(const DenseAttnDecodeParams &params);

}
