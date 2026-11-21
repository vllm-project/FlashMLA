// Probe 13: what does CUDA's e4m3 conversion make of 0x7f?
// torch's float8_e4m3fn says nan; the row-10 cache is ~98% 0x7f in the nope, so
// the kernel's answer decides whether those bytes poison P.
#include "common.h"
#include <cuda_fp8.h>

__global__ void kernel() {
  if (threadIdx.x != 0) return;
  __nv_fp8x2_e4m3 x;
  *(uint16_t*)&x = 0x7f7f;
  float2 f = (float2)x;
  __half_raw h = __nv_cvt_fp8_to_halfraw(0x7f, __NV_E4M3);
  __half hh = *(__half*)&h;
  printf("fp8x2=%f,%f half=%f isnan=%d,%d\n", f.x, f.y, __half2float(hh),
         isnan(f.x), isnan(__half2float(hh)));
}

int main() {
  CK(cudaFree(0));
  kernel<<<1, 32>>>();
  cudaError_t e = cudaDeviceSynchronize();
  int rc = report("e4m3_0x7f", e == cudaSuccess, cudaGetErrorString(e));
  printf("%s probe13\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
