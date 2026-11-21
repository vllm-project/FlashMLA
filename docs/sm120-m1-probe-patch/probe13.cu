// probe13: numeric marker method for the TMMA A/B fragment linear-index
// semantics. sA[(row,k)] = row*100+k ; sB[(token,dim)] = token*100+dim.
// partition_fragment_{A,B} + copy, then read back each element value to
// recover (row,k) / (token,dim) for every linear index in the fragment.
// This tells us how the identity rS (written by scale_softmax with C-g
// indices) must be laid out to match what cute::gemm expects for A.
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm80.hpp>
#include <cstdio>
using namespace cute;
using cutlass::bfloat16_t;
using bf16 = bfloat16_t;

using TMMA = decltype(make_tiled_mma(
    SM80_16x8x16_F32BF16BF16F32_TN{},
    Layout<Shape<_4, _1, _1>>{},
    Layout<Shape<_1, _1, _1>>{}
));

__global__ void probe(float* outA, float* outB) {
    __shared__ bf16 sA[64*64], sB[64*64];
    for (int i = threadIdx.x; i < 64*64; i += blockDim.x) {
        int r = i/64, c = i%64;
        sA[i] = (bf16)(r*100 + c);          // (row, k)
        sB[i] = (bf16)(r*100 + c);          // (token, dim)
    }
    __syncthreads();
    auto dA = make_tensor(make_smem_ptr(sA), Layout<Shape<_64, _64>, Stride<_64, _1>>{});
    auto dB = make_tensor(make_smem_ptr(sB), Layout<Shape<_64, _64>, Stride<_64, _1>>{});

    TMMA t{};
    auto thr = t.get_slice(threadIdx.x);
    auto fA = thr.partition_fragment_A(dA);
    cute::copy(thr.partition_A(dA), fA);
    auto fB = thr.partition_fragment_B(dB);
    cute::copy(thr.partition_B(dB), fB);

    // thread 0..3 dump
    if (threadIdx.x < 4) {
        int t0 = threadIdx.x;
        float* oa = outA + t0*34;   // 32 A values + "shape" sentinel
        for (int i = 0; i < size(fA); ++i) oa[i] = (float)fA(i);
        oa[32] = 9999; oa[33] = 0;
        float* ob = outB + t0*34;
        for (int i = 0; i < size(fB); ++i) ob[i] = (float)fB(i);
        ob[32] = 9999; ob[33] = 0;
    }
}

int main() {
    float *dA, *dB; cudaMalloc(&dA, 8*34*sizeof(float)); cudaMalloc(&dB, 8*34*sizeof(float));
    std::vector<float> hA(8*34), hB(8*34);
    probe<<<1,128>>>(dA, dB);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(hA.data(), dA, 8*34*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hB.data(), dB, 8*34*sizeof(float), cudaMemcpyDeviceToHost);
    for (int t0 = 0; t0 < 4; ++t0) {
        printf("thread %d A[32] = (row,k):", t0);
        for (int i = 0; i < 32; ++i) printf(" %d:%d", i, (int)hA[t0*34+i]);
        printf("\n");
        printf("thread %d B[32] = (token,dim):", t0);
        for (int i = 0; i < 32; ++i) printf(" %d:%d", i, (int)hB[t0*34+i]);
        printf("\n");
    }
    return 0;
}
