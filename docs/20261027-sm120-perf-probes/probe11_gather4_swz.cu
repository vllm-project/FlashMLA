// Probe 11: does `tile::gather4` swizzle by the ABSOLUTE destination address or
// by the row index within the box?
//
// Row 10's K plane is `K_TMA_W`-byte slices of 64 tokens with one gather4 per
// 4-token group, so a group's four destination rows sit at absolute rows
// 4g..4g+3 of the slice. If the hardware XORs the 16 B chunks by the absolute
// address the phase is (4g+j)&7; if it XORs by the box row index the phase is
// always j; if it does not swizzle at all the phase is 0. Even groups cannot
// tell the first two apart, odd groups differ by 4, so all three expectations
// are counted.
//
// Descriptor: UINT8, dim0 = 128 B (the slice), dim1 = rows, box {128, 1},
// -DSWZ=1 selects SWIZZLE_128B, otherwise SWIZZLE_NONE.
#include "common.h"

constexpr int kRows = 4096;
constexpr int kRowB = 128;
constexpr int kGroups = 20;   // 80 rows, ten 8-row swizzle periods

__global__ void kernel(const __grid_constant__ CUtensorMap map, const int* rows,
                       int* bad_abs, int* bad_rel, int* bad_none) {
  __shared__ alignas(1024) uint8_t tile[kGroups * 4 * kRowB];
  __shared__ uint64_t mbar;
  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    mbar_expect_tx(&mbar, kGroups * 4 * kRowB);
    for (int g = 0; g < kGroups; ++g) {
      const int4* r = (const int4*)(rows + 4 * g);
      tma_gather4_plain(&map, tile + g * 4 * kRowB, &mbar, 0, r->x, r->y, r->z, r->w);
    }
  }
  __syncthreads();
  mbar_wait(&mbar, 0);

  // verify: byte (tok*128 + chunk*16 + t) of the tile must be the gmem byte
  // (row*128 + chunk*16 + t) placed at chunk ^ phase
  if (threadIdx.x < kGroups * 4 * kRowB / 4) {
    const int flat = threadIdx.x;             // one u32 of the image
    const int tok = flat / (kRowB / 4);
    const int w = flat % (kRowB / 4);         // u32 index in the row
    const int chunk = w / 4, t = w % 4;
    const int row = rows[tok];
    const uint32_t want = (uint32_t)(row * 1000 + chunk * 4 + t);
    const uint8_t* p = tile + tok * kRowB;
    const uint32_t a = *(const uint32_t*)(p + ((chunk ^ (tok & 7)) << 4) + 4 * t);
    const uint32_t b = *(const uint32_t*)(p + ((chunk ^ (tok & 3)) << 4) + 4 * t);
    const uint32_t c = *(const uint32_t*)(p + (chunk << 4) + 4 * t);
    if (a != want) atomicAdd(bad_abs, 1);
    if (b != want) atomicAdd(bad_rel, 1);
    if (c != want) atomicAdd(bad_none, 1);
  }
}

int main() {
  CK(cudaFree(0));

  uint32_t* host = (uint32_t*)malloc((size_t)kRows * kRowB);
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kRowB / 4; ++c) host[(size_t)r * (kRowB / 4) + c] = (uint32_t)(r * 1000 + c);
  void* kv = dalloc((size_t)kRows * kRowB);
  h2d(kv, host, (size_t)kRows * kRowB);

  int* rows = (int*)dalloc(kGroups * 4 * sizeof(int));
  int hrows[kGroups * 4];
  for (int i = 0; i < kGroups * 4; ++i) hrows[i] = i;
  h2d(rows, hrows, sizeof(hrows));

  CUtensorMap map = make_map_2d(kv, kRowB, kRows, kRowB, kRowB, 1,
                                CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                SWZ ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE);

  int* bad_abs = (int*)dalloc(sizeof(int));
  int* bad_rel = (int*)dalloc(sizeof(int));
  int* bad_none = (int*)dalloc(sizeof(int));
  CK(cudaMemset(bad_abs, 0, sizeof(int)));
  CK(cudaMemset(bad_rel, 0, sizeof(int)));
  CK(cudaMemset(bad_none, 0, sizeof(int)));

  int rc = 0;
  kernel<<<1, 1024>>>(map, rows, bad_abs, bad_rel, bad_none);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    rc |= report("gather4_swizzle", false, cudaGetErrorString(e));
  } else {
    int ha = 0, hr = 0, hn = 0;
    d2h(&ha, bad_abs, sizeof(int));
    d2h(&hr, bad_rel, sizeof(int));
    d2h(&hn, bad_none, sizeof(int));
    char det[128];
    snprintf(det, sizeof(det), "mismatch abs=%d boxrel=%d none=%d", ha, hr, hn);
    rc |= report("gather4_swizzle", ha == 0 || hr == 0 || hn == 0, det);
  }
  printf("%s probe11 swz=%d\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL", SWZ);
  return rc;
}
