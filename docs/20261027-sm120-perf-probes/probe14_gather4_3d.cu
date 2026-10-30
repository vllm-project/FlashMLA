// Probe 14: `tile::gather4` on a rank-3 map, so a page block can be its own
// axis instead of being assumed to be a whole number of rows.
//
// Row 10's V4 cache rows are 585 B apart while a page block is padded to a
// multiple of 576, so the block step is not an integer number of rows and no
// flat row axis can address it. The dense sm120 page copy already keeps the
// page and batch axes explicit in its tensor map; this probe asks whether
// gather4 accepts the same shape and in which order the coordinates go:
//   {x, y0, y1, y2, y3, z}  (expected)   vs   {x, z, y0..y3}  (-DZFIRST)
// Descriptor: UINT8, box {128,1,1}, dim1 = 8 rows, dim2 = 64 blocks, SWIZZLE_128B,
// and a block stride of 8*585 + 100 bytes so no flat row axis could work.
#include "common.h"

constexpr int kBlocks = 64;
constexpr int kToks = 8;
constexpr int kRowB = 585;          // V4-ish row, data is the first 512 B
constexpr int kSlice = 512;         // an x that only exists inside the data
constexpr int kBlkStride = kToks * kRowB + 100;
constexpr int kGroups = 8;          // gathers, each 4 rows of one block
constexpr int kSliceB = 8192;


// rank-3 gather4: the coordinates are {x, y0..y3} plus one per outer dim
__device__ __forceinline__ void tma3d_gather4(const void* map, void* smem,
                                              uint64_t* mbar, int32_t col,
                                              int r0, int r1, int r2, int r3, int z) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile::gather4.mbarrier::"
      "complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6, %7}], [%8];" ::
          "r"(smem_u32(smem)),
      "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(z), "r"(smem_u32(mbar))
      : "memory");
}

__global__ void kernel(const __grid_constant__ CUtensorMap map, const int* rows,
                       const int* blks, int* bad) {
  __shared__ alignas(1024) uint8_t plane[kGroups * 4 * 128];
  __shared__ uint64_t mbar;
  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    mbar_expect_tx(&mbar, kGroups * 4 * 128);
    for (int g = 0; g < kGroups; ++g) {
      const int4* r = (const int4*)(rows + 4 * g);
#ifdef ZFIRST
      tma3d_gather4(&map, plane + g * 512, &mbar, kSlice, blks[g], r->x, r->y, r->z, r->w);
#else
      tma3d_gather4(&map, plane + g * 512, &mbar, kSlice, r->x, r->y, r->z, r->w, blks[g]);
#endif
    }
  }
  __syncthreads();
  mbar_wait(&mbar, 0);

  for (int flat = threadIdx.x; flat < kGroups * 4 * 128 / 4; flat += blockDim.x) {
    const int g = flat / (4 * 32), rem = flat % (4 * 32);
    const int j = rem / 32, w = rem % 32;
    const int chunk = w / 4, t = w % 4;
    const int row = rows[4 * g + j], blk = blks[g];
    const uint32_t want = (uint32_t)(blk * 100000 + row * 1000 + (kSlice + chunk * 16) / 4 + t);
    const uint32_t got = *(const uint32_t*)(plane + g * 512 + j * 128 + (((chunk ^ (j & 7)) << 4) + 4 * t));
    if (got != want) atomicAdd(bad, 1);
  }
}

int main() {
  CK(cudaFree(0));

  // every block's own byte pattern, so a wrong z is unmistakable
  const size_t n = (size_t)kBlocks * kBlkStride;
  uint32_t* host = (uint32_t*)malloc(n);
  for (int b = 0; b < kBlocks; ++b)
    for (int w = 0; w < kBlkStride / 4; ++w)
      host[(size_t)b * (kBlkStride / 4) + w] = (uint32_t)(b * 100000 + w);
  void* kv = dalloc(n);
  h2d(kv, host, n);

  int hrows[kGroups * 4], hblk[kGroups];
  for (int g = 0; g < kGroups; ++g) {
    hblk[g] = g * 7 + 1;
    for (int j = 0; j < 4; ++j) hrows[4 * g + j] = (g + j * 3) % kToks;
  }
  int* rows = (int*)dalloc(sizeof(hrows));
  int* blks = (int*)dalloc(sizeof(hblk));
  h2d(rows, hrows, sizeof(hrows));
  h2d(blks, hblk, sizeof(hblk));

  alignas(64) CUtensorMap map{};
  uint64_t gdim[3] = {kRowB, kToks, kBlocks};
  uint64_t gstr[2] = {kRowB, (uint64_t)kBlkStride};
  uint32_t box[3] = {128, 1, 1};
  uint32_t est[3] = {1, 1, 1};
  CUK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, kv, gdim, gstr, box, est,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  int* bad = (int*)dalloc(sizeof(int));
  CK(cudaMemset(bad, 0, sizeof(int)));

  int rc = 0;
  kernel<<<1, 1024>>>(map, rows, blks, bad);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    rc |= report("gather4_rank3", false, cudaGetErrorString(e));
  } else {
    int h = 0;
    d2h(&h, bad, sizeof(int));
    char det[64];
    snprintf(det, sizeof(det), "mismatches=%d", h);
    rc |= report("gather4_rank3", h == 0, det);
  }
  printf("%s probe14\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
