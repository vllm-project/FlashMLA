// Probe 10: `tile::gather4` and the `.cta_group::1` token on sm_120a.
//
// PTX lists gather4 as sm_100a-or-higher *and* "supported on sm_120a"; the
// .cta_group::1 is what the SM100 dequant path carries but ptxas 13.3 rejects
// it at sm_120a; the default build is therefore the plain form and -DCTA_GROUP
// re-tests the token cell (expected ptxas FAIL).
// The descriptor is a UINT32 view with box {32, 1} = 128 B per gathered row.
// Measured constraint: the x coordinate must be a multiple of 4 u32 (16 B);
// col=2/3 faults with an illegal instruction, col=4 works.
#include "common.h"

constexpr int kRows = 4096;   // token rows
constexpr int kRowU32 = 32;   // 128 B
constexpr int kGroups = 4;    // rows per gather4

__global__ void kernel(const __grid_constant__ CUtensorMap map, int col,
                       int4 rows, int* bad) {
  __shared__ alignas(128) uint8_t tile[kGroups * 128];
  __shared__ uint64_t mbar;
  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    mbar_expect_tx(&mbar, kGroups * 128);
#ifdef CTA_GROUP
    tma_gather4(&map, tile, &mbar, col, rows.x, rows.y, rows.z, rows.w);
#else
    tma_gather4_plain(&map, tile, &mbar, col, rows.x, rows.y, rows.z, rows.w);
#endif
  }
  __syncthreads();
  mbar_wait(&mbar, 0);

  // each gathered row lands 128 B apart, rows in argument order
  const int rr[4] = {rows.x, rows.y, rows.z, rows.w};
  if (threadIdx.x < 128) {
    int g = threadIdx.x / 32, i = threadIdx.x % 32;
    uint32_t got = ((uint32_t*)tile)[g * kRowU32 + i];
    // columns past the 128 B row are TMA out-of-bounds: zero-filled
    uint32_t want = (col + i < kRowU32) ? (uint32_t)(rr[g] * 1000 + col + i) : 0u;
    if (got != want) atomicAdd(bad, 1);
  }
}

int main() {
  CK(cudaFree(0));

  uint32_t* host = (uint32_t*)malloc((size_t)kRows * kRowU32 * 4);
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kRowU32; ++c) host[(size_t)r * kRowU32 + c] = (uint32_t)(r * 1000 + c);
  void* kv = dalloc((size_t)kRows * kRowU32 * 4);
  h2d(kv, host, (size_t)kRows * kRowU32 * 4);

  CUtensorMap map = make_map_2d(kv, kRowU32, kRows, kRowU32 * 4, kRowU32, 1,
                                CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                CU_TENSOR_MAP_SWIZZLE_NONE);

  int* bad = (int*)dalloc(sizeof(int));
  CK(cudaMemset(bad, 0, sizeof(int)));

  int rc = 0;
  const struct { int col; int4 rows; } cases[] = {
      {4, {0, 1, 2, 3, }},
      {28, {4095, 1000, 7, 2500}},
  };
  for (int i = 0; i < 2; ++i) {
    kernel<<<1, 128>>>(map, cases[i].col, cases[i].rows, bad);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
      rc |= report("launch", false, cudaGetErrorString(e));
      break;
    }
    int h = 0;
    d2h(&h, bad, sizeof(int));
    char name[48];
    snprintf(name, sizeof(name), "gather4_case_%d_rows_land", i);
    char det[64];
    snprintf(det, sizeof(det), "mismatches=%d", h);
    rc |= report(name, h == 0, det);
    CK(cudaMemset(bad, 0, sizeof(int)));
  }

  printf("%s probe10\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
