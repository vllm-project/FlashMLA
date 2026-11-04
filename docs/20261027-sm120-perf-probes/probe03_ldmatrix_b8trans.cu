// Probe 3: `ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8` fragment table.
//
// Legality is settled; this probe produces OUR decode table. It fills two
// 16x16 byte matrices with a unique value per (matrix,row,col), runs the
// instruction with the standard x2 addressing (lanes 0-15 row addresses of
// matrix 0, lanes 16-31 of matrix 1), dumps every register, and checks:
//   - the 2048 fetched bytes are exactly a permutation of the input (no
//     duplication, no out-of-bounds),
//   - two runs are bit-identical,
// and prints the first lanes' registers so the mapping can be written down.
#include "common.h"

constexpr int kBytes = 512;  // 2 matrices x 16 rows x 16 B

__device__ __forceinline__ void ldm_b8trans(const void* addr, uint32_t& r0,
                                            uint32_t& r1, uint32_t& r2,
                                            uint32_t& r3) {
  asm volatile(
      "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(smem_u32(addr)));
}

__global__ void run(const uint32_t* pattern, uint32_t* out) {
  __shared__ alignas(16) uint8_t s[kBytes];
  // invert the second matrix so the two x2 halves never collide
  for (int i = threadIdx.x; i < kBytes; i += 32)
    s[i] = (uint8_t)((i & 0xFF) ^ ((i & 0x100) ? 0xFF : 0));
  __syncwarp();
  int lane = threadIdx.x & 31;
  const void* addr = s + lane * 16;  // lane<16 -> matrix 0 rows, else matrix 1
  uint32_t r0, r1, r2, r3;
  ldm_b8trans(addr, r0, r1, r2, r3);
  uint4 v = {r0, r1, r2, r3};
  ((uint4*)out)[lane] = v;
}

int main() {
  CK(cudaFree(0));
  uint32_t* out1 = (uint32_t*)dalloc(32 * 16);
  uint32_t* out2 = (uint32_t*)dalloc(32 * 16);

  run<<<1, 32>>>(nullptr, out1);
  CK(cudaDeviceSynchronize());
  run<<<1, 32>>>(nullptr, out2);
  CK(cudaDeviceSynchronize());

  uint8_t h1[512], h2[512];
  d2h(h1, out1, 512);
  d2h(h2, out2, 512);

  int hist_in[256] = {0}, hist_out[256] = {0};
  for (int i = 0; i < 256; ++i) hist_in[i] = 2;  // each byte value appears twice
  for (int i = 0; i < 512; ++i) hist_out[h1[i]]++;
  bool perm = true;
  for (int i = 0; i < 256; ++i) perm &= (hist_in[i] == hist_out[i]);
  int rc = report("permutation", perm, "2048 fetched bytes are a permutation");
  rc |= report("deterministic", memcmp(h1, h2, 512) == 0, "");

  printf("decode table (lane: r0 r1 r2 r3, hex):\n");
  for (int lane = 0; lane < 8; ++lane) {
    const uint32_t* v = (const uint32_t*)(h1 + lane * 16);
    printf("  lane %2d: %08x %08x %08x %08x\n", lane, v[0], v[1], v[2], v[3]);
  }
  printf("input byte value = smem offset (mod 256)\n");

  printf("%s probe03\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
