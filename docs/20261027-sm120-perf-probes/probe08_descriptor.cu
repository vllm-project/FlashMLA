// Probe 8: the three descriptor hazards.
//
// A: descriptor in 64 B-aligned device memory + `prefetch.tensormap` + TMA -
//    must work and land correct data.
// B: the same descriptor copied to base+8 -> prefetch must raise a fault.
// C: swizzle agreement. Descriptor swizzles 128 B; the read side must XOR by
//    (row & 7) at 16 B granularity. We compare the XOR read (must match global)
//    with a linear read (must differ), which is exactly the failure DeepGEMM
//    #433 shipped when the two sides were derived from different constants.
#include "common.h"

constexpr int kRows = 4;
constexpr int kCols = 32;  // u32 -> 128 B row

// 128 B swizzle in 16 B groups, 4 rows (rows 0..3 use subgroup 0..3)
__device__ __forceinline__ int sw_off(int row, int col) {
  int g = (col * 4) / 16;              // 16 B group index
  int w = (col * 4) % 16;              // byte within group
  int gs = g ^ (row & 7);
  return row * 128 + gs * 16 + w;
}

__global__ void kernel(CUtensorMap* const __grid_constant__ mapp, bool prefetch,
                       int* ok_prefetch, uint32_t* out_swz, uint32_t* out_lin) {
  __shared__ alignas(128) uint8_t tile[kRows * 128];
  __shared__ uint64_t mbar;
  if (threadIdx.x == 0) {
    if (prefetch) prefetch_tensormap(mapp);
    mbar_init(&mbar, 1);
    mbar_expect_tx(&mbar, kRows * 128);
    tma_2d_cta(mapp, tile, &mbar, 0, 0);
  }
  __syncthreads();
  mbar_wait(&mbar, 0);
  fence_async_shared();

  int t = threadIdx.x;
  for (int i = t; i < kRows * kCols; i += blockDim.x) {
    int row = i / kCols, col = i % kCols;
    out_swz[i] = *(uint32_t*)(tile + sw_off(row, col));
    out_lin[i] = ((uint32_t*)tile)[i];
  }
  if (t == 0) *ok_prefetch = 1;
}

int main() {
  CK(cudaFree(0));

  uint32_t host[kRows * kCols];
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kCols; ++c) host[r * kCols + c] = (uint32_t)(r * 1000 + c);

  void* g = dalloc(sizeof(host));
  h2d(g, host, sizeof(host));
  CUtensorMap map = make_map_2d(g, kCols, kRows, kCols * 4, kCols, kRows,
                                CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                CU_TENSOR_MAP_SWIZZLE_128B);

  // 64 B-aligned staging (cudaMalloc returns >= 256 B alignment)
  void* map_dev = dalloc(128);
  h2d(map_dev, &map, sizeof(map));
  void* map_bad = (char*)dalloc(192) + 8;
  h2d(map_bad, &map, sizeof(map));

  int* ok = (int*)dalloc(sizeof(int));
  uint32_t *o1 = (uint32_t*)dalloc(sizeof(host)), *o2 = (uint32_t*)dalloc(sizeof(host));

  int rc = 0;
  // A + C
  kernel<<<1, 64>>>((CUtensorMap*)map_dev, true, ok, o1, o2);
  CK(cudaDeviceSynchronize());
  uint32_t hs[kRows * kCols], hl[kRows * kCols];
  d2h(hs, o1, sizeof(hs));
  d2h(hl, o2, sizeof(hl));
  rc |= report("aligned_descriptor_prefetch", true, "no fault");
  rc |= report("swizzle_unswizzle_roundtrip", memcmp(hs, host, sizeof(host)) == 0, "");
  int diff = 0;
  for (int i = 0; i < kRows * kCols; ++i) diff += (hl[i] != host[i]);
  rc |= report("linear_read_mismatches_as_expected", diff > 0, "(wrong unswizzle is detectable)");

  // B
  cudaGetLastError();
  kernel<<<1, 64>>>((CUtensorMap*)map_bad, true, ok, o1, o2);
  cudaError_t e = cudaDeviceSynchronize();
  char detail[128];
  snprintf(detail, sizeof(detail), "-> %s", cudaGetErrorString(e));
  rc |= report("misaligned_descriptor_prefetch_faults", e != cudaSuccess, detail);

  printf("%s probe08\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
