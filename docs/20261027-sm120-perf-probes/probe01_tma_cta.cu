// Probe 1: TMA 2D end to end at run time.
//
// Three build cells (see run_all.sh):
//   default            uint8 tile, `.shared::cta`, must PASS at run time
//   -DSUBBYTE          uint4b tile, cta form, must PASS (dst must be 128 B aligned)
//   -DCLUSTER -DSUBBYTE  same tile through `.shared::cluster`, expected ptxas FAIL
//
// Verifies: the tile lands, the byte/nibble pattern is intact, and
// mbarrier expect_tx / try_wait.parity actually gates the copy.
#include "common.h"

constexpr int kRows = 64;
constexpr int kCols = 256;  // elements; 128 B row for both uint8 and 16U4

__global__
#ifdef CLUSTER
    __cluster_dims__(1, 1, 1)
#endif
        void
        tma_kernel(const __grid_constant__ CUtensorMap map, int row, int* bad) {
  __shared__ alignas(256) uint8_t tile[128];
  __shared__ uint64_t mbar;

  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    // Sub-byte notes: boxDim stays in elements (256 = one 128 B row) but the
    // smem destination must be 128 B aligned - a 64 B offset faults.
    mbar_expect_tx(&mbar, 128);
#ifdef CLUSTER
    tma_2d_cluster(&map, tile, &mbar, 0, row);
#else
    tma_2d_cta(&map, tile, &mbar, 0, row);
#endif
  }
  __syncthreads();
  mbar_wait(&mbar, 0);

  if (threadIdx.x < 32) {
#ifdef SUBBYTE
    for (int b = 0; b < 4; ++b) {
      int byte = threadIdx.x * 4 + b;
      uint8_t got = tile[byte];
      uint8_t lo = (uint8_t)((row + 2 * byte) & 0xF);
      uint8_t hi = (uint8_t)((row + 2 * byte + 1) & 0xF);
      uint8_t want = (uint8_t)(lo | (hi << 4));
      if (got != want) atomicAdd(bad, 1);
    }
#else
    for (int b = 0; b < 4; ++b) {
      int byte = threadIdx.x * 4 + b;
      if (tile[byte] != (uint8_t)((row * kCols + byte) & 0xFF)) atomicAdd(bad, 1);
    }
#endif
  }
}

int main() {
  CK(cudaFree(0));
  CUK(cuInit(0));

  int dev = 0;
  cudaDeviceProp prop{};
  CK(cudaGetDevice(&dev));
  CK(cudaGetDeviceProperties(&prop, dev));
  printf("device: %s cc %d.%d\n", prop.name, prop.major, prop.minor);

  int* bad = (int*)dalloc(sizeof(int));
  const int rows[] = {0, 7, 63};
  int total = 0;

#ifdef SUBBYTE
  // 4-bit data: 64 rows x 128 B, nibble value = (row + col) & 0xF
  const int row_bytes = 128;
  uint8_t* host = (uint8_t*)malloc(kRows * row_bytes);
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kCols; c += 2) {
      uint8_t lo = (uint8_t)((r + c) & 0xF);
      uint8_t hi = (uint8_t)((r + c + 1) & 0xF);
      host[r * row_bytes + c / 2] = (uint8_t)(lo | (hi << 4));
    }
  void* kv = dalloc(kRows * row_bytes);
  h2d(kv, host, kRows * row_bytes);
  CUtensorMap map =
      make_map_2d(kv, kCols, kRows, row_bytes, kCols, 1,
                  CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
                  CU_TENSOR_MAP_SWIZZLE_NONE);
#else
  const int row_bytes = kCols;
  uint8_t* host = (uint8_t*)malloc(kRows * row_bytes);
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kCols; ++c) host[r * row_bytes + c] = (uint8_t)((r * kCols + c) & 0xFF);
  void* kv = dalloc(kRows * row_bytes);
  h2d(kv, host, kRows * row_bytes);
  CUtensorMap map =
      make_map_2d(kv, kCols, kRows, row_bytes, kCols, 1,
                  CU_TENSOR_MAP_DATA_TYPE_UINT8, CU_TENSOR_MAP_SWIZZLE_NONE);
#endif

  for (int i = 0; i < 3; ++i) {
    CK(cudaMemset(bad, 0, sizeof(int)));
    printf("launch row %d\n", rows[i]);
    tma_kernel<<<1, 64>>>(map, rows[i], bad);
    CK(cudaDeviceSynchronize());
    int h = 0;
    d2h(&h, bad, sizeof(int));
    total += report(rows[i] == rows[i] ? "row_landed" : "", h == 0, "");
    if (h) printf("  %d mismatched 4-byte groups\n", h);
  }

  printf("%s probe01\n", total == 0 ? "RESULT PASS" : "RESULT FAIL");
  return total == 0 ? 0 : 1;
}
