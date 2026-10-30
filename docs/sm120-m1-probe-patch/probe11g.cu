// probe11g: is QK correct when the A operand is read from GLOBAL memory (the one
// path the kernel uses but probe11 never validated -- probe11 read A from smem)?
// Encoding: Q (64,576) with Q(r,k)=k+1; K smem NKContig (64,576) K(c,k)=c+1 ->
//   P(r,c) = sum_k (k+1)*(c+1) = (c+1)*2080.
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

__global__ void probe(float* outP, bf16* gQ) {
    constexpr int HEAD_DIM_K = 576;
    extern __shared__ char dyn[];
    bf16* sK = reinterpret_cast<bf16*>(dyn);
    for (int i = threadIdx.x; i < 64*HEAD_DIM_K; i += blockDim.x)
        sK[i] = (bf16)((i/64)+1);          // K(c,k)=c+1, row-major (N,K)
    __syncthreads();

    TMMA tmma{};
    auto thr = tmma.get_slice(threadIdx.x);

    using SKK = Layout<Shape<Int<64>, Int<HEAD_DIM_K>>, Stride<Int<HEAD_DIM_K>,_1>>;
    auto sKt = make_tensor(make_smem_ptr(sK), SKK{});
    auto gQt = make_tensor(make_gmem_ptr(gQ), Layout<Shape<_64, Int<HEAD_DIM_K>>, Stride<Int<HEAD_DIM_K>,_1>>{});

    Tensor rP = partition_fragment_C(tmma, Shape<_64,_64>{});
    cute::clear(rP);
    CUTE_NO_UNROLL
    for (int seg = 0; seg < HEAD_DIM_K/64; ++seg) {
        auto gQseg = local_tile(gQt, Shape<_64,_64>{}, make_coord(_0{}, seg));
        auto sKseg = local_tile(sKt, Shape<Int<64>,_64>{}, make_coord(_0{}, seg));
        Tensor rQa = thr.partition_fragment_A(gQseg);
        Tensor rKb = thr.partition_fragment_B(sKseg);
        cute::copy(thr.partition_A(gQseg), rQa);
        cute::copy(thr.partition_B(sKseg), rKb);
        cute::gemm(tmma, rQa, rKb, rP);
    }
    for (int i = 0; i < size(rP); ++i)
        outP[threadIdx.x*size(rP) + i] = rP(i);
}

int main() {
    std::vector<bf16> hQ(64*576);
    for (int i = 0; i < 64*576; ++i) hQ[i] = (bf16)((i%576)+1);
    bf16 *dQ;
    cudaMalloc(&dQ, 64*576*sizeof(bf16));
    cudaMemcpy(dQ, hQ.data(), 64*576*sizeof(bf16), cudaMemcpyHostToDevice);

    float *dP; cudaMalloc(&dP, 128*32*sizeof(float));
    std::vector<float> hG(128*32);

    size_t smem = 64*576*sizeof(bf16);
    cudaFuncSetAttribute((const void*)probe, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    probe<<<1,128, smem>>>(dP, dQ);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("gmem-A KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(hG.data(), dP, 128*32*sizeof(float), cudaMemcpyDeviceToHost);

    constexpr float KSUM = 2080.0f;
    int bad = 0; float mn=1e30f, mx=-1e30f;
    for (int i = 0; i < 128*32; ++i) {
        if (hG[i] <= 0 || fmodf(hG[i], KSUM) != 0.0f || hG[i] > 64*KSUM) bad++;
        mn = fminf(mn, hG[i]); mx = fmaxf(mx, hG[i]);
    }
    printf("gmem-A QK: bad=%d min=%g max=%g (expect min=2080 max=64*2080=%g)\n", bad, mn, mx, 64.f*KSUM);
    return 0;
}
