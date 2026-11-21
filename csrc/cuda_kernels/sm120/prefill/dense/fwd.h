#pragma once

#include "params.h"

namespace sm120::prefill::dense {

template<int D_QK>
void run_dense_fwd_kernel(const DensePrefillParams &params);

}
