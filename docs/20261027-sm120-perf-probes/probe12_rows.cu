// Probe 12: the real row-10 geometry on top of probe 11's finding (the gather4
// swizzle follows the absolute destination address).
//
// Same descriptor and destination layout as the kernel: UINT8, dim0 = 656 B
// (the V32 cache row), dim1 = 4096 tokens, box {128, 1}, SWIZZLE_128B, 40 KB
// smem plane = 5 slices x 8192 B, one gather of 4 tokens per (slice, group),
// x = {0,128,256,384,528} (528 = behind the row's fp32 scale field, like
// K_ROPE_X). The row coordinates come from a shuffled table, as in the kernel.
// Proves the descriptor accepts a stride that is not a multiple of 128 and that
// the swizzled image lands exactly where k_row_ptr() reads it.
#include "common.h"

constexpr int kRows = 4096;
constexpr int kRowB = 656;
constexpr int kSlices = 5;
constexpr int kSliceB = 8192;     // 64 tokens * 128 B
constexpr int kGroups = 16;       // 4-token groups per slice
constexpr int kToks = 64;

__device__ int slice_x(int s) { return s < 4 ? s * 128 : 528; }

__global__ void kernel(const __grid_constant__ CUtensorMap map, const int* rows, int* bad) {
  __shared__ alignas(1024) uint8_t plane[kSlices * kSliceB];
  __shared__ uint64_t mbar;
  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    mbar_expect_tx(&mbar, kSlices * kToks * 128);
    for (int s = 0; s < kSlices; ++s)
      for (int g = 0; g < kGroups; ++g) {
        const int4* r = (const int4*)(rows + 4 * g);
        tma_gather4_plain(&map, plane + s * kSliceB + g * 4 * 128, &mbar,
                          slice_x(s), r->x, r->y, r->z, r->w);
      }
  }
  __syncthreads();
  mbar_wait(&mbar, 0);

  constexpr int N = kSlices * kToks * 128 / 4;    // u32 of the image
  for (int flat = threadIdx.x; flat < N; flat += blockDim.x) {
    const int slice = flat / (kToks * 32);
    const int rem = flat % (kToks * 32);
    const int tok = rem / 32, w = rem % 32;        // w = u32 in the row
    const int chunk = w / 4, t = w % 4;
    const int row = rows[tok];
    const uint32_t want = (uint32_t)(row * 1000 + (slice_x(slice) + chunk * 16) / 4 + t);
    const uint32_t got =
        *(const uint32_t*)(plane + slice * kSliceB + tok * 128 + (((chunk ^ (tok & 7)) << 4) + 4 * t));
    if (got != want) atomicAdd(bad, 1);
  }
}

int main() {
  CK(cudaFree(0));

  const size_t n = (size_t)kRows * kRowB;
  uint32_t* host = (uint32_t*)malloc(n);
  for (int r = 0; r < kRows; ++r)
    for (int w = 0; w < kRowB / 4; ++w) host[(size_t)r * (kRowB / 4) + w] = (uint32_t)(r * 1000 + w);
  void* kv = dalloc(n);
  h2d(kv, host, n);

  int* rows = (int*)dalloc(kToks * sizeof(int));
  int hrows[kToks];
  for (int i = 0; i < kToks; ++i) hrows[i] = (i * 977 + 13) % kRows;   // shuffled, all distinct-ish
  h2d(rows, hrows, sizeof(hrows));

  CUtensorMap map = make_map_2d(kv, kRowB, kRows, kRowB, 128, 1,
                                CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                CU_TENSOR_MAP_SWIZZLE_128B);

  int* bad = (int*)dalloc(sizeof(int));
  CK(cudaMemset(bad, 0, sizeof(int)));

  int rc = 0;
  kernel<<<1, 1024>>>(map, rows, bad);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    rc |= report("gather4_real_rows", false, cudaGetErrorString(e));
  } else {
    int h = 0;
    d2h(&h, bad, sizeof(int));
    char det[64];
    snprintf(det, sizeof(det), "mismatches=%d", h);
    rc |= report("gather4_real_rows", h == 0, det);
  }
  printf("%s probe12\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
