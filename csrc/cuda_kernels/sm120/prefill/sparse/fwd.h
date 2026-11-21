#pragma once

#include "params.h"

namespace sm120::prefill::sparse {

template<int D_QK>
void run_sparse_fwd_kernel(const SparseAttnFwdParams &params);

}
