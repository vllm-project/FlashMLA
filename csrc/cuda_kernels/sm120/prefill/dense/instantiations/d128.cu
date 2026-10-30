#include "../fwd.cuh"

namespace sm120::prefill::dense {

template void run_dense_fwd_kernel<128>(const DensePrefillParams &params);
template void KernelTemplate<128>::run(const DensePrefillParams &params);

}
