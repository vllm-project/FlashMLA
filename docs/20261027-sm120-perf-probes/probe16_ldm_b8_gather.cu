// Probe 16: verify the b8 ldmatrix gather used by the PV dequant, end to end.
//
// probe03 printed the raw instruction table; this checks the *use* of it: the PV
// path reads a (dim, token) fp8 tile with one `ldmatrix.m16n16.x2.trans.b8` per
// 32 tokens x 16 dims and then exchanges the result across each group of 4 lanes
// so every lane ends up holding the four (dim, token) pairs its mma B fragment
// asks for, namely dim = 8a + t/4 and tokens 2*(t%4) + {0,1}, +8.
//
// The tile is 16 tokens x 16 dims, filled with a unique value per element, so
// the matrix-0 half of one window is checked exhaustively; matrix 1 (tokens
// 16..31) is read by the same addressing and is covered by probe03.
#include "common.h"

constexpr int kDims = 16;
constexpr int kToks = 16;
constexpr int kRowPad = kDims + 16;

__device__ __forceinline__ void ldm_b8x2(const uint32_t* addr, uint32_t (&r)[4]) {
  asm volatile(
      "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"((uint32_t)__cvta_generic_to_shared(addr)));
}

__global__ void run(uint32_t* bad_out) {
  __shared__ alignas(16) uint8_t s[32 * kRowPad];
  for (int i = threadIdx.x; i < 32 * kRowPad; i += 32) s[i] = 0xEE;
  for (int tok = 0; tok < kToks; ++tok)
    for (int d = threadIdx.x; d < kDims; d += 32)
      s[tok * kRowPad + d] = (uint8_t)(tok * kDims + d);
  __syncwarp();

  int lane = threadIdx.x & 31;
  const int j = lane & 3, grp = lane & ~3;
  const int src[2] = {grp | (j >> 1), grp | (2 + (j >> 1))};
  const int sh = (j & 1)*16;   // bit offset of the wanted byte pair

  uint32_t raw[4];
  ldm_b8x2((const uint32_t*)(s + lane*kRowPad), raw);

  int bad = 0;
  for (int q = 0; q < 2; ++q) {   // q = a, matrix 0 only (tokens 0..15)
    uint32_t wa = __shfl_sync(0xffffffff, raw[q], src[0]);
    uint32_t wb = __shfl_sync(0xffffffff, raw[q], src[1]);
    const int tok0 = 2*j;
    const int dim = 8*q + lane/4;
    uint8_t got[4] = {(uint8_t)(wa >> sh), (uint8_t)(wa >> (sh + 8)),
                      (uint8_t)(wb >> sh), (uint8_t)(wb >> (sh + 8))};
    int want[4] = {tok0*kDims + dim, (tok0 + 1)*kDims + dim,
                   (tok0 + 8)*kDims + dim, (tok0 + 9)*kDims + dim};
    for (int e = 0; e < 4; ++e)
      if (got[e] != (uint8_t)want[e]) ++bad;
    if (lane < 4)
      printf("lane %d q%d dim %d: got %02x %02x %02x %02x want %02x %02x %02x %02x\n",
             lane, q, dim, got[0], got[1], got[2], got[3],
             want[0], want[1], want[2], want[3]);
  }
  bad_out[lane] = (uint32_t)(bad ? 1 : 0);
}

int main() {
  CK(cudaFree(0));
  uint32_t* out = (uint32_t*)dalloc(32 * 4);
  run<<<1, 32>>>(out);
  CK(cudaDeviceSynchronize());
  uint32_t h[32];
  d2h(h, out, 128);
  int bad = 0;
  for (int i = 0; i < 32; ++i) bad += (int)h[i];
  int rc = report("gather", bad == 0, "every lane's 4 (dim, token) pairs match");
  printf("%s probe16\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
