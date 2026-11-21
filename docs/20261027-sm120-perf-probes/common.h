// Shared helpers for the SM120 (sm_120a) performance probes.
// Pure CUDA + driver API, no CUTLASS. Every probe is a single translation unit
// with its own main(); build with:
//   nvcc -arch=sm_120a -O3 -lineinfo probeNN_*.cu -lcuda -o probeNN
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x)                                                                  \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "%s:%d CUDA %s\n", __FILE__, __LINE__,                   \
              cudaGetErrorString(e_));                                         \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

#define CUK(x)                                                                 \
  do {                                                                         \
    CUresult r_ = (x);                                                         \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char* s_ = "?";                                                    \
      cuGetErrorString(r_, &s_);                                               \
      fprintf(stderr, "%s:%d CUDA driver %d (%s)\n", __FILE__, __LINE__, (int)r_, \
              s_);                                                             \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

static inline void* dalloc(size_t n) {
  void* p = nullptr;
  CK(cudaMalloc(&p, n));
  return p;
}

static inline void h2d(void* dst, const void* src, size_t n) {
  CK(cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice));
}

static inline void d2h(void* dst, const void* src, size_t n) {
  CK(cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

// ---- mbarrier -------------------------------------------------------------
__device__ __forceinline__ void mbar_init(uint64_t* mbar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(mbar)),
               "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* mbar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smem_u32(mbar)),
               "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, uint32_t phase) {
  asm volatile(
      "{\n"
      ".reg .pred P%=;\n"
      "LAB%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P%=, [%0], %1;\n"
      "@!P%= bra LAB%=;\n"
      "}\n" ::"r"(smem_u32(mbar)),
      "r"(phase));
}

__device__ __forceinline__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ---- TMA ------------------------------------------------------------------
// `map` is a __grid_constant__ parameter; x is derived from generic smem.
__device__ __forceinline__ void tma_2d_cta(const void* map, void* smem,
                                           uint64_t* mbar, int32_t x, int32_t y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_u32(smem)),
      "l"(map), "r"(smem_u32(mbar)), "r"(x), "r"(y)
      : "memory");
}

__device__ __forceinline__ void tma_2d_cluster(const void* map, void* smem,
                                               uint64_t* mbar, int32_t x,
                                               int32_t y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_u32(smem)),
      "l"(map), "r"(smem_u32(mbar)), "r"(x), "r"(y)
      : "memory");
}

__device__ __forceinline__ void tma_gather4(const void* map, void* smem,
                                            uint64_t* mbar, int32_t col,
                                            int r0, int r1, int r2, int r3) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::"
      "complete_tx::bytes.cta_group::1 [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::
          "r"(smem_u32(smem)),
      "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(smem_u32(mbar))
      : "memory");
}

__device__ __forceinline__ void tma_gather4_plain(const void* map, void* smem,
                                                  uint64_t* mbar, int32_t col,
                                                  int r0, int r1, int r2,
                                                  int r3) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::"
      "complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];" ::
          "r"(smem_u32(smem)),
      "l"(map), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(smem_u32(mbar))
      : "memory");
}

__device__ __forceinline__ void prefetch_tensormap(const void* map) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(map));
}

// ---- tensor map -----------------------------------------------------------
// Rank-2, innermost dim first, byte strides. Box must fit the swizzle width.
static inline CUtensorMap make_map_2d(void* base, uint64_t dim0, uint64_t dim1,
                                      uint64_t stride1_bytes, uint32_t box0,
                                      uint32_t box1, CUtensorMapDataType type,
                                      CUtensorMapSwizzle swz) {
  alignas(64) CUtensorMap m{};
  uint64_t gdim[2] = {dim0, dim1};
  uint64_t gstr[1] = {stride1_bytes};
  uint32_t box[2] = {box0, box1};
  uint32_t est[2] = {1, 1};
  CUK(cuTensorMapEncodeTiled(&m, type, 2, base, gdim, gstr, box, est,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, swz,
                             CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return m;
}

// ---- reporting ------------------------------------------------------------
static inline int report(const char* name, bool ok, const char* detail = "") {
  printf("%s %s %s\n", ok ? "PASS" : "FAIL", name, detail);
  return ok ? 0 : 1;
}
