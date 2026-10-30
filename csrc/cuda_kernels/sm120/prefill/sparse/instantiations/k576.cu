#include "../fwd.cuh"

namespace sm120::prefill::sparse {

template void run_sparse_fwd_kernel<576>(const SparseAttnFwdParams &params);
template void KernelTemplate<576>::run(const SparseAttnFwdParams &params);

}
