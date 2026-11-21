// Probe 2: cp.async, its commit/wait groups, and the predicated zero-fill form.
//
// Part A: 64 lanes copy 16 B each; odd lanes carry a false predicate, so their
//         destination must come back zeroed instead of untouched.
// Part B: two commit groups; wait_group 1 must make group A visible and may
//         leave group B in flight.
#include "common.h"

constexpr int kThreads = 64;
constexpr int kBytes = kThreads * 16;

// The 4th operand is the SOURCE SIZE in bytes, not a predicate: 0 zero-fills
// the whole 16 B destination without reading. (A predicate there is accepted
// by ptxas but means src_size = 0/1, which copies one byte - measured.)
__device__ __forceinline__ void cp_async16(void* dst, const void* src,
                                           int src_size) {
  asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16, %2;"
               ::"r"(smem_u32(dst)),
               "l"(src), "r"(src_size));
}

__device__ __forceinline__ void cp_commit() {
  asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

__global__ void pred_kernel(const uint8_t* g, int* bad) {
  __shared__ uint8_t s[kBytes];
  int tid = threadIdx.x;
  cp_async16(s + tid * 16, g + tid * 16, (tid & 1) ? 0 : 16);
  cp_commit();
  cp_wait<0>();
  __syncthreads();
  for (int b = 0; b < 16; ++b) {
    uint8_t got = s[tid * 16 + b];
    uint8_t want = (tid & 1) ? 0 : g[tid * 16 + b];
    if (got != want) atomicAdd(bad, 1);
  }
}

__global__ void group_kernel(const uint8_t* g, uint8_t* out_a, uint8_t* out_b) {
  __shared__ uint8_t s[32];
  if (threadIdx.x == 0) {
    cp_async16(s + 0, g + 0, 16);
    cp_commit();
    cp_async16(s + 16, g + 16, 16);
    cp_commit();
    cp_wait<1>();
    // group A must be visible here; group B may not be
    *(uint4*)out_a = *(uint4*)s;
    cp_wait<0>();
    *(uint4*)out_b = *(uint4*)(s + 16);
  }
}

int main() {
  CK(cudaFree(0));
  uint8_t* host = (uint8_t*)malloc(kBytes);
  for (int i = 0; i < kBytes; ++i) host[i] = (uint8_t)(i * 7 + 3);
  uint8_t* g = (uint8_t*)dalloc(kBytes);
  h2d(g, host, kBytes);

  int* bad = (int*)dalloc(sizeof(int));
  CK(cudaMemset(bad, 0, sizeof(int)));
  pred_kernel<<<1, kThreads>>>(g, bad);
  CK(cudaDeviceSynchronize());
  int hbad = 0;
  d2h(&hbad, bad, sizeof(int));
  int rc = report("predicated_zero_fill", hbad == 0, hbad ? "mismatches>0" : "");

  uint8_t *a = (uint8_t*)dalloc(16), *b = (uint8_t*)dalloc(16);
  group_kernel<<<1, 32>>>(g, a, b);
  CK(cudaDeviceSynchronize());
  uint8_t ha[16], hb[16];
  d2h(ha, a, 16);
  d2h(hb, b, 16);
  rc |= report("commit_group_ordering", memcmp(ha, host, 16) == 0 &&
                                            memcmp(hb, host + 16, 16) == 0,
               "(group A after wait_group 1, B after wait_group 0)");

  printf("%s probe02\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
