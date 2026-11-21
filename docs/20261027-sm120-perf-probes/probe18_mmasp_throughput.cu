// Probe 18: is the 2:4 sparse fp8 atom worth anything on sm_120a?
//
// `mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32` covers a k64 output
// tile in one instruction; the dense path needs two m16n8k32 for the same
// tile. Same output, half the A bytes (A arrives in the compressed form) and
// half the issues. Row 18 showed PV is LDS/LDSM-bound, so the answer here
// decides whether row 19 (V held 2:4 in the cache) has a pipe to run on.
//
// No memory is touched: A/B/C live in registers and four independent
// accumulators keep the dependency chain from becoming the limit. Reports
// cycles per k64 tile for one warp and for eight warps (the kernel's CTA
// width) - the second number is the one that matters.
#include "common.h"

#define ACCS 4
#define ITERS 200000

__device__ __forceinline__ void fill_c(float (&c)[ACCS][4]) {
#pragma unroll
  for (int i = 0; i < ACCS; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) c[i][j] = (float)(threadIdx.x + i + j);
}

// One k64 tile = two dense k32 mma.
__global__ void k_dense(unsigned long long* out) {
  float c[ACCS][4];
  uint32_t a0 = 0x38383838u, a1 = a0, a2 = a0, a3 = a0;
  uint32_t b0 = 0x38383838u, b1 = b0;
  fill_c(c);
  __syncthreads();
  unsigned long long t0 = threadIdx.x == 0 ? clock64() : 0;
  for (int it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int i = 0; i < ACCS; ++i)
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
          : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  unsigned long long t1 = threadIdx.x == 0 ? clock64() : 0;
  if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
  if (threadIdx.x == 0) out[2] = (unsigned long long)c[0][0];
}

// One k64 tile = one sparse k64 mma. A is 4 regs (2:4 compressed), B 4, the
// metadata byte-pair rides after C, the trailing 0x0 is the selector.
__global__ void k_sparse(unsigned long long* out) {
  float c[ACCS][4];
  uint32_t a0 = 0x38383838u, a1 = a0, a2 = a0, a3 = a0;
  uint32_t b0 = 0x38383838u, b1 = b0, b2 = b0, b3 = b0;
  uint32_t e0 = 0x11111111u;  // keep-elements-0-and-1 of every group of four
  fill_c(c);
  __syncthreads();
  unsigned long long t0 = threadIdx.x == 0 ? clock64() : 0;
  for (int it = 0; it < ITERS; ++it) {
#pragma unroll
    for (int i = 0; i < ACCS; ++i)
      asm volatile(
          "mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
          : "+f"(c[i][0]), "+f"(c[i][1]), "+f"(c[i][2]), "+f"(c[i][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
            "r"(b3), "r"(e0));
  }
  unsigned long long t1 = threadIdx.x == 0 ? clock64() : 0;
  if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
  if (threadIdx.x == 0) out[2] = (unsigned long long)c[0][0];
}


// Numeric arm: A = B = 1.0, C = 0, one iteration. The dense k64 tile sums 64
// ones; the sparse tile keeps 2 of every 4, so it must sum 32 - which also
// pins the metadata convention (0x11 nibbles = keep elements 0 and 1).
__global__ void k_check(float* out) {
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  uint32_t a0 = 0x38383838u, a1 = a0, a2 = a0, a3 = a0;
  uint32_t b0 = 0x38383838u, b1 = b0, b2 = b0, b3 = b0;
  uint32_t e0 = 0x11111111u;
  {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    if (threadIdx.x == 0) out[0] = c[0];
    c[0] = c[1] = c[2] = c[3] = 0.f;
    asm volatile(
        "mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%0,%1,%2,%3}, %12, 0x0;"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(e0));
    if (threadIdx.x == 0) out[1] = c[0];
  }
}

static void run(const char* name, void (*kern)(unsigned long long*), int warps,
                double clk_ghz) {
  unsigned long long* d = (unsigned long long*)dalloc(3 * sizeof(*d));
  CK(cudaMemset(d, 0, 3 * sizeof(*d)));
  int threads = warps * 32;
  kern<<<1, threads>>>(d);
  CK(cudaDeviceSynchronize());
  unsigned long long cyc = 0;
  d2h(&cyc, d, sizeof(cyc));
  // Tiles: ACCS per iteration per warp, and the dense arm issues two mma each.
  double tiles = (double)ITERS * ACCS * warps;
  double per_tile = (double)cyc / tiles;
  // 16x8x64 MACs per tile, 2 flop per MAC.
  double ns = per_tile / clk_ghz;
  double tflops = 2.0 * 16 * 8 * 64 / ns / 1.0e3;
  printf("PROBE18 %-7s warps=%d cycles=%llu cycles/tile=%.3f  %.1f TFLOPS\n",
         name, warps, cyc, per_tile, tflops);
  CK(cudaFree(d));
}

int main() {
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp p{};
  CK(cudaGetDeviceProperties(&p, dev));
  int clk_khz = 0;
  CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
  printf("PROBE18 device %s sm_%d%d clock %.2f GHz\n", p.name, p.major, p.minor,
         clk_khz / 1e6);
  double ghz = clk_khz / 1e6;
  // warm
  run("dense", k_dense, 1, ghz);
  run("sparse", k_sparse, 1, ghz);
  run("dense", k_dense, 8, ghz);
  run("sparse", k_sparse, 8, ghz);
  float* dchk = (float*)dalloc(2 * sizeof(float));
  k_check<<<1, 32>>>(dchk);
  CK(cudaDeviceSynchronize());
  float chk[2] = {0.f, 0.f};
  d2h(chk, dchk, sizeof(chk));
  printf("PROBE18 check dense=%.1f sparse=%.1f (expect 64 / 32)\n", chk[0], chk[1]);
  CK(cudaFree(dchk));
  printf("PROBE18 %s\n", chk[0] == 64.f && chk[1] == 32.f ? "PASS" : "FAIL");
  return 0;
}
