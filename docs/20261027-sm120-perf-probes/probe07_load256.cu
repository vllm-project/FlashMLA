// Probe 7: 256-bit global load with L1::no_allocate, and the shared-store side.
//
// `ld.global.L1::no_allocate.v8.f32` needs 32 B alignment and PTX ISA 8.8
// (CUDA >= 12.9). The 256-bit *shared* store is not documented; build the
// default cell to find out, and -DNO_STV8 if ptxas rejects `st.shared.v8.b32`
// (the fallback is two 128-bit stores). The misaligned arm reads at +16 B.
#include "common.h"

constexpr int kLanes = 32;
constexpr int kBytes = kLanes * 32;

__device__ __forceinline__ void ld256(const void* p, float* v) {
  asm volatile("ld.global.L1::no_allocate.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=f"(v[0]), "=f"(v[1]), "=f"(v[2]), "=f"(v[3]), "=f"(v[4]),
                 "=f"(v[5]), "=f"(v[6]), "=f"(v[7])
               : "l"(p));
}

__device__ __forceinline__ void st_shared256(void* s, const float* v) {
#if defined(STV8)
  // ptxas 13.3 rejects this: "global state space expected for feature 256 bit"
  asm volatile("st.shared.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" ::"r"(smem_u32(s)),
               "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])),
               "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])),
               "r"(__float_as_uint(v[4])), "r"(__float_as_uint(v[5])),
               "r"(__float_as_uint(v[6])), "r"(__float_as_uint(v[7])));
#else
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(smem_u32(s)),
               "r"(__float_as_uint(v[0])), "r"(__float_as_uint(v[1])),
               "r"(__float_as_uint(v[2])), "r"(__float_as_uint(v[3])));
  asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(smem_u32((char*)s + 16)),
               "r"(__float_as_uint(v[4])), "r"(__float_as_uint(v[5])),
               "r"(__float_as_uint(v[6])), "r"(__float_as_uint(v[7])));
#endif
}

__global__ void kernel(const float* g, size_t misalign, float* out) {
  __shared__ alignas(32) float s[kLanes * 8];
  float v[8];
  int lane = threadIdx.x & 31;
  ld256((const char*)g + misalign + lane * 32, v);
  st_shared256(s + lane * 8, v);
  __syncthreads();
  for (int i = 0; i < 8; ++i) out[lane * 8 + i] = s[lane * 8 + i];
}

int main() {
  CK(cudaFree(0));
  float* host = (float*)malloc(kBytes + 32);
  for (int i = 0; i < kBytes / 4 + 8; ++i) host[i] = (float)i * 0.5f;
  float* g = (float*)dalloc(kBytes + 32);
  h2d(g, host, kBytes + 32);
  float* out = (float*)dalloc(kBytes);

  // 32 B is the required alignment: +16 must fault, not silently split.
  int rc = 0;
  const size_t arms[] = {0, 32};
  const char* names[] = {"aligned_32B", "aligned_plus32B"};
  for (int arm = 0; arm < 2; ++arm) {
    kernel<<<1, kLanes>>>(g, arms[arm], out);
    CK(cudaDeviceSynchronize());
    float h[kLanes * 8];
    d2h(h, out, kBytes);
    rc |= report(names[arm], memcmp(h, (char*)host + arms[arm], kBytes) == 0, "");
  }
  cudaGetLastError();
  kernel<<<1, kLanes>>>(g, 16, out);
  cudaError_t e = cudaDeviceSynchronize();
  char detail[96];
  snprintf(detail, sizeof(detail), "-> %s", cudaGetErrorString(e));
  rc |= report("misaligned_16B_faults", e == cudaErrorMisalignedAddress, detail);
  cudaGetLastError();
  printf("%s probe07\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
