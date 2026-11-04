// Probe 5: does OUR toolchain keep `mma.sync.kind::f8f6f4` in SASS?
//
// DeepGEMM hard-#errors below CUDA 13 because pre-13 ptxas silently dropped
// `block_scale`. This probe builds against whatever $CUDA_HOME points at and
// asserts the instruction survives to SASS (QMMA/OMMA). The script greps
// `cuobjdump -sass`; the kernel also runs a zero-input sanity check when a
// device is present.
#include "common.h"

__global__ void f8f6f4_kernel(const uint32_t* a, const uint32_t* b, float* d) {
  uint32_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
  uint32_t b0 = b[0], b1 = b[1];
  float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  int t = threadIdx.x;
  d[t * 4 + 0] = c0;
  d[t * 4 + 1] = c1;
  d[t * 4 + 2] = c2;
  d[t * 4 + 3] = c3;
}

int main() {
  int devs = 0;
  cudaGetDeviceCount(&devs);
  if (devs == 0) {
    printf("no device; compile-only probe\n");
    return 0;
  }
  CK(cudaFree(0));
  uint32_t ha[4] = {0, 0, 0, 0}, hb[2] = {0, 0};
  uint32_t *a = (uint32_t*)dalloc(16), *b = (uint32_t*)dalloc(8);
  float* d = (float*)dalloc(32 * 16);
  h2d(a, ha, 16);
  h2d(b, hb, 8);
  f8f6f4_kernel<<<1, 32>>>(a, b, d);
  CK(cudaDeviceSynchronize());
  float h[128];
  d2h(h, d, sizeof(h));
  int bad = 0;
  for (int i = 0; i < 128; ++i) bad += (h[i] != 0.f);
  printf("%s probe05 (script also greps SASS for QMMA/OMMA)\n",
         bad == 0 ? "RESULT PASS" : "RESULT FAIL");
  return bad == 0 ? 0 : 1;
}
