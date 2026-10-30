// [SM120] minimal launch probe of the real kernel (grid (1,1,1)) to isolate
// "invalid configuration argument": config-vs-params vs kernel itself.
#include <cstdio>
#include <cuda_runtime.h>
#include <kernels/sm120/decode/sparse/config.h>
#include <kernels/sm120/decode/sparse/splitkv_mla.cuh>

using namespace sm120::decode::sparse;

int main() {
    using Kernel = KernelTemplate<ModelType::V32, 64>;
    using KFN = decltype(&flash_fwd_splitkv_mla_fp8_sparse_kernel<Kernel, TMAParams>);
    KFN kfn = &flash_fwd_splitkv_mla_fp8_sparse_kernel<Kernel, TMAParams>;

    cudaError_t e2 = cudaFuncSetAttribute((const void*)kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);
    printf("cudaFuncSetAttribute err=%s\n", cudaGetErrorString(e2));

    SparseAttnDecodeParams params;
    memset(&params, 0, sizeof(params));
    params.num_sm_parts = 1;
    params.s_q = 1;
    params.h_q = 64;
    TMAParams tma_params{};  // zeros: invalid descriptor, but launch should still be accepted

    kfn<<<dim3(1,1,1), 256, sizeof(Kernel::SharedMemoryPlan), 0>>>(params, tma_params);
    cudaError_t e3 = cudaPeekAtLastError();
    printf("launch peek error: %s\n", cudaGetErrorString(e3));
    cudaError_t e4 = cudaDeviceSynchronize();
    printf("sync error: %s\n", cudaGetErrorString(e4));
    return 0;
}
