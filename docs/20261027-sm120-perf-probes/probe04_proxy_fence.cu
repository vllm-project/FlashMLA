// Probe 4: generic-proxy WAR hazard - TMA (async proxy) writes the tile, then
// ldmatrix (generic proxy) reads it, with and without
// `fence.proxy.async.shared::cta` between the mbarrier wait and the read.
//
// DeepGEMM needed 14 of these fences on SM120; without one it failed 2 runs in
// 6000. Run each arm 6000 times and count differing outputs. The reference is a
// kernel that fills the same tile by plain stores and runs the same ldmatrix.
#include "common.h"

constexpr int kIter = 6000;
constexpr int kBytes = 2 * 16 * 16;  // 2 matrices of 16x16 bytes
__constant__ uint8_t kGolden[kBytes];

__device__ __forceinline__ void ldm(const void* addr, uint4& v) {
  asm volatile(
      "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "r"(smem_u32(addr)));
}

__global__ void ref_kernel(uint32_t* out) {
  __shared__ alignas(16) uint8_t s[kBytes];
  for (int i = threadIdx.x; i < kBytes; i += 32) s[i] = kGolden[i];
  __syncwarp();
  uint4 v;
  ldm(s + (threadIdx.x & 31) * 16, v);
  ((uint4*)out)[threadIdx.x & 31] = v;
}

__global__ void tma_kernel(const __grid_constant__ CUtensorMap map,
                           uint32_t* out, int use_fence) {
  __shared__ alignas(128) uint8_t s[kBytes];
  __shared__ uint64_t mbar;
  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    mbar_expect_tx(&mbar, kBytes);
    tma_2d_cta(&map, s, &mbar, 0, 0);
  }
  __syncthreads();
  mbar_wait(&mbar, 0);
  if (use_fence) fence_async_shared();
  uint4 v;
  ldm(s + (threadIdx.x & 31) * 16, v);
  ((uint4*)out)[threadIdx.x & 31] = v;
}

int main() {
  CK(cudaFree(0));

  uint8_t host[kBytes];
  for (int i = 0; i < kBytes; ++i) host[i] = (uint8_t)(i & 0xFF);
  CK(cudaMemcpyToSymbol(kGolden, host, kBytes));

  void* g = dalloc(kBytes);
  h2d(g, host, kBytes);
  // box = the whole 16 x 32 tile (one 16 B row per box row), 512 B per transfer
  CUtensorMap map = make_map_2d(g, 16, 32, 16, 16, 32,
                                CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                CU_TENSOR_MAP_SWIZZLE_NONE);

  uint32_t* ref = (uint32_t*)dalloc(512);
  ref_kernel<<<1, 32>>>(ref);
  CK(cudaDeviceSynchronize());

  uint8_t a[512], b[512];
  d2h(a, ref, 512);
  uint32_t* out = (uint32_t*)dalloc(512);

  int rc = 0;
  for (int arm = 0; arm < 2; ++arm) {
    int bad = 0;
    for (int it = 0; it < kIter; ++it) {
      tma_kernel<<<1, 32>>>(map, out, arm == 0 ? 0 : 1);
      CK(cudaDeviceSynchronize());
      d2h(b, out, 512);
      if (memcmp(a, b, 512) != 0) bad++;
    }
    char name[64];
    snprintf(name, sizeof(name), "arm_%s_%d_iters", arm ? "fence" : "nofence", kIter);
    rc |= report(name, bad == 0, bad ? "mismatching outputs" : "");
    if (bad) printf("  %d/%d runs differed\n", bad, kIter);
  }

  printf("%s probe04\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
