// probe11l: print the FULL (row, col) map of the QK A-fragment for the sm120
// TMMA (SM80_16x8x16_F32BF16BF16F32_TN, AtomLayoutMNK (_4,_1,_1), 128 threads,
// tile (64,64)). Fill smem A(r,c) = r*64+c, partition_fragment_A, dump.
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cstdio>
#include <vector>

using namespace cute;
using cutlass::bfloat16_t;
using bf16 = bfloat16_t;

using TMMA = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{},
    Layout<Shape<_4, _1, _1>>{},
    Layout<Shape<_1, _1, _1>>{}
));

__global__ void probe(float* outA) {
    __shared__ __align__(1024) bf16 sMat[64*64];
    for (int i = threadIdx.x; i < 64*64; i += blockDim.x)
        sMat[i] = (bf16)(i);
    __syncthreads();
    TMMA tmma{};
    auto thr = tmma.get_slice(threadIdx.x);
    auto sT = make_tensor(make_smem_ptr(sMat), Layout<Shape<_64,_64>, Stride<_64,_1>>{});
    Tensor rA = thr.partition_fragment_A(sT);
    cute::copy(thr.partition_A(sT), rA);
    for (int i = 0; i < size(rA); ++i)
        outA[threadIdx.x*size(rA) + i] = (float)rA(i);
}

int main() {
    float *dA;
    cudaMalloc(&dA, 128*64*sizeof(float));
    probe<<<1,128>>>(dA);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    std::vector<float> hA(128*64);
    cudaMemcpy(hA.data(), dA, 128*64*sizeof(float), cudaMemcpyDeviceToHost);

    for (int t = 0; t < 8; ++t) {
        printf("t=%02d (base=%d): ", t, 16*(t/32)+(t%32)/4);
        for (int g = 0; g < 32; ++g) {
            int v = (int)hA[t*32+g];
            printf("(%d,%d) ", v/64, v%64);
            if (g%4==3) printf("| ");
        }
        printf("\n");
    }
    return 0;
}
