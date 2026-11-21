#include "../fwd.cuh"

namespace sm120::prefill::sparse {

template void run_sparse_fwd_kernel<512>(const SparseAttnFwdParams &params);
template void KernelTemplate<512>::run(const SparseAttnFwdParams &params);

}
