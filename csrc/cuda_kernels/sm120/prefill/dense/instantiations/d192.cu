#include "../fwd.cuh"

namespace sm120::prefill::dense {

template void run_dense_fwd_kernel<192>(const DensePrefillParams &params);
template void KernelTemplate<192>::run(const DensePrefillParams &params);

}
