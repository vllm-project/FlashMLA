#pragma once

#include "params.h"

namespace sm90 {

template<typename InputT, int HEAD_DIM_K>
void run_flash_splitkv_mla_kernel(DenseAttnDecodeParams &params);

}
