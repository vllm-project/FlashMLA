// Probe 6: what the two synchronisation idioms become on sm_120a.
//
// Compile-only; the script disassembles the cubin. Expected, from the earlier
// ptxas/nvdisasm run: `bar.sync` -> BAR.SYNC ... DEFER_BLOCKING (imprecise),
// `mbarrier.try_wait.parity` -> an MBAR/TRY_WAIT form that stays precise.
// This gives §12.26 a measured replacement text.
#include "common.h"

__global__ void bar_kernel(int* out) {
  __shared__ int s;
  if (threadIdx.x == 0) s = 7;
  asm volatile("bar.sync 0;");
  out[threadIdx.x] = s;
}

__global__ void mbar_kernel(int* out) {
  __shared__ uint64_t mbar;
  __shared__ int s;
  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    s = 7;
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(&mbar)));
  }
  __syncthreads();
  mbar_wait(&mbar, 0);
  out[threadIdx.x] = s;
}

int main() {
  int devs = 0;
  cudaGetDeviceCount(&devs);
  if (devs == 0) {
    printf("no device; compile-only probe\n");
    return 0;
  }
  CK(cudaFree(0));
  int* out = (int*)dalloc(64 * sizeof(int));
  bar_kernel<<<1, 32>>>(out);
  mbar_kernel<<<1, 32>>>(out);
  CK(cudaDeviceSynchronize());
  printf("RESULT PASS probe06 (script checks disassembly)\n");
  return 0;
}
