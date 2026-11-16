#include "../phase1.h"
#include "../phase1.cuh"

namespace legacy::sm90::prefill::sparse_fwd {

template void run_fwd_phase1_kernel<576, false>(const SparseAttnFwdParams& params);

template void run_etap_split_kernel<576, false>(const EtapSplitParams& params);

}
