// Probe 15: a byte-offset y axis. `tile::gather4` is 2D only (probe 14: ptxas
// rejects `.3d` with `.tile::gather4`), and V4's rows are 585 B apart inside a
// block that is padded to a multiple of 576, so no page block is a whole number
// of rows. Setting dim1's stride to 1 byte and passing the row's byte offset as
// the y coordinate makes the axis exact for every format.
#include "common.h"

constexpr int kBlocks = 64;
constexpr int kRowB = 585;
constexpr int kBlkStride = 8 * kRowB + 100;   // deliberately not 576-aligned
constexpr int kGroups = 8;
constexpr int kRowsPerGather = 4;

__global__ void kernel(const __grid_constant__ CUtensorMap map, const int* rows,
                       const int* cols, int* bad) {
  __shared__ alignas(1024) uint8_t plane[kGroups * kRowsPerGather * 128];
  __shared__ uint64_t mbar;
  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    mbar_expect_tx(&mbar, kGroups * kRowsPerGather * 128);
    for (int g = 0; g < kGroups; ++g) {
      const int4* r = (const int4*)(rows + 4 * g);
      // y = the byte offset of the row, x = a 128 B window inside its data
      tma_gather4_plain(&map, plane + g * 512, &mbar, cols[g],
                        r->x * kRowB, r->y * kRowB, r->z * kRowB, r->w * kRowB);
    }
  }
  __syncthreads();
  mbar_wait(&mbar, 0);

  for (int flat = threadIdx.x; flat < kGroups * 4 * 128 / 4; flat += blockDim.x) {
    const int g = flat / 128, rem = flat % 128;
    const int j = rem / 32, w = rem % 32;
    const int chunk = w / 4, t = w % 4;
    const int row = rows[4 * g + j], col = cols[g];
    const uint32_t want = (uint32_t)(row * 1000 + (col + chunk * 16) / 4 + t);
    const uint32_t got = *(const uint32_t*)(plane + g * 512 + j * 128 + (((chunk ^ (j & 7)) << 4) + 4 * t));
    if (got != want) atomicAdd(bad, 1);
  }
}

int main() {
  CK(cudaFree(0));

  // gmem: row r of block b starts at b*kBlkStride, rows identical per index so
  // the pattern depends on the flat row number only (rows are kBlocks*8 long)
  const size_t n = (size_t)kBlocks * kBlkStride;
  uint32_t* host = (uint32_t*)malloc(n);
  for (int b = 0; b < kBlocks; ++b)
    for (int w = 0; w < kBlkStride / 4; ++w) host[(size_t)b * (kBlkStride / 4) + w] = (uint32_t)(w);
  void* kv = dalloc(n);
  h2d(kv, host, n);

  int hrows[kGroups * 4], hcol[kGroups];
  for (int g = 0; g < kGroups; ++g) {
    hcol[g] = 128 * (g % 3);            // 0, 128, 256 (multiples of 16 B)
    for (int j = 0; j < 4; ++j) hrows[4 * g + j] = g * 8 + j;   // distinct rows
  }
  int* rows = (int*)dalloc(sizeof(hrows));
  int* cols = (int*)dalloc(sizeof(hcol));
  h2d(rows, hrows, sizeof(hrows));
  h2d(cols, hcol, sizeof(hcol));

  alignas(64) CUtensorMap map{};
  uint64_t gdim[2] = {kRowB, (uint64_t)n};        // dim1 extent in bytes
  uint64_t gstr[1] = {1};                          // one byte per y step
  uint32_t box[2] = {128, 1};
  uint32_t est[2] = {1, 1};
  CUK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, kv, gdim, gstr, box, est,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  int* bad = (int*)dalloc(sizeof(int));
  CK(cudaMemset(bad, 0, sizeof(int)));

  int rc = 0;
  kernel<<<1, 1024>>>(map, rows, cols, bad);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    rc |= report("gather4_byte_axis", false, cudaGetErrorString(e));
  } else {
    int h = 0;
    d2h(&h, bad, sizeof(int));
    char det[64];
    snprintf(det, sizeof(det), "mismatches=%d", h);
    rc |= report("gather4_byte_axis", h == 0, det);
  }
  printf("%s probe15\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
