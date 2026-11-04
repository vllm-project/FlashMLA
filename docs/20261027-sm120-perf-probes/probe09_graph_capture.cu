// Probe 9: the TMA kernel under CUDA graph capture and capture replay.
//
// The SM120 field reports (vLLM #56700 / #56892) are full of graph-capture
// failures around these ops. Capture the probe-1 load, instantiate, replay it
// twice, and require correct data on every replay.
#include "common.h"

constexpr int kRows = 64;
constexpr int kCols = 256;

__global__ void tma_kernel(const __grid_constant__ CUtensorMap map, int row,
                           int* bad) {
  __shared__ alignas(128) uint8_t tile[128];
  __shared__ uint64_t mbar;
  if (threadIdx.x == 0) {
    mbar_init(&mbar, 1);
    mbar_expect_tx(&mbar, 128);
    tma_2d_cta(&map, tile, &mbar, 0, row);
  }
  __syncthreads();
  mbar_wait(&mbar, 0);
  if (threadIdx.x < 32) {
    for (int b = 0; b < 4; ++b) {
      int byte = threadIdx.x * 4 + b;
      if (tile[byte] != (uint8_t)((row * kCols + byte) & 0xFF)) atomicAdd(bad, 1);
    }
  }
}

int main() {
  CK(cudaFree(0));

  uint8_t* host = (uint8_t*)malloc(kRows * kCols);
  for (int r = 0; r < kRows; ++r)
    for (int c = 0; c < kCols; ++c) host[r * kCols + c] = (uint8_t)((r * kCols + c) & 0xFF);
  void* kv = dalloc(kRows * kCols);
  h2d(kv, host, kRows * kCols);
  CUtensorMap map = make_map_2d(kv, kCols, kRows, kCols, 128, 1,
                                CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                CU_TENSOR_MAP_SWIZZLE_NONE);

  int* bad = (int*)dalloc(sizeof(int));
  CK(cudaMemset(bad, 0, sizeof(int)));

  cudaStream_t s;
  CK(cudaStreamCreate(&s));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  tma_kernel<<<1, 64, 0, s>>>(map, 5, bad);
  CK(cudaStreamEndCapture(s, &graph));

  size_t nnodes = 0;
  CK(cudaGraphGetNodes(graph, nullptr, &nnodes));
  int rc = report("capture_produces_graph", nnodes == 1, "");

  cudaError_t ie = cudaGraphInstantiate(&exec, graph, 0);
  rc |= report("graph_instantiate", ie == cudaSuccess, cudaGetErrorString(ie));

  if (ie == cudaSuccess) {
    for (int replay = 0; replay < 2; ++replay) {
      CK(cudaGraphLaunch(exec, s));
      CK(cudaStreamSynchronize(s));
      int h = 0;
      d2h(&h, bad, sizeof(int));
      char name[40];
      snprintf(name, sizeof(name), "replay_%d_correct", replay);
      rc |= report(name, h == 0, "");
    }
  }

  printf("%s probe09\n", rc == 0 ? "RESULT PASS" : "RESULT FAIL");
  return rc;
}
