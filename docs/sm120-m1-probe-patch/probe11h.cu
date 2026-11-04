// probe11h: isolate the failed variable. probe11g changed BOTH A source (gmem)
// and B stride (576 vs 64). Here B is a standalone (64,64) NKContig stride (64,1)
// exactly like probe11; only A comes from global memory (64 x 64, row-major).
// P(r,c) = sum_k (k+1)*(c+1) = (c+1)*2080  (Q(r,k)=k+1, K(c,k)=c+1)
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
    __shared__ __align__(1024) bf16 sK[64*64];
    for (int i = threadIdx.x; i < 64*64; i += blockDim.x)
        sK[i] = (bf16)((i/64)+1);          // K(c,k)=c+1
    __syncthreads();

    TMMA tmma{};
    auto thr = tmma.get_slice(threadIdx.x);

    auto gQt = make_tensor(make_gmem_ptr(gQ), Layout<Shape<_64,_64>, Stride<_64,_1>>{});
    using NK = Layout<Shape<_64,_64>, Stride<Int<64>,_1>>;
    auto sKt = make_tensor(make_smem_ptr(sK), NK{});

    Tensor rQa = thr.partition_fragment_A(gQt);
    Tensor rKb = thr.partition_fragment_B(sKt);
    cute::copy(thr.partition_A(gQt), rQa);
    cute::copy(thr.partition_B(sKt), rKb);
    Tensor rP = partition_fragment_C(tmma, Shape<_64,_64>{});
    cute::clear(rP);
    cute::gemm(tmma, rQa, rKb, rP);
    for (int i = 0; i < size(rP); ++i)
        outP[threadIdx.x*size(rP) + i] = rP(i);
}

int main() {
    std::vector<bf16> hQ(64*64);
    for (int i = 0; i < 64*64; ++i) hQ[i] = (bf16)((i%64)+1);   // Q(r,k)=k+1
    bf16 *dQ;
    cudaMalloc(&dQ, 64*64*sizeof(bf16));
    cudaMemcpy(dQ, hQ.data(), 64*64*sizeof(bf16), cudaMemcpyHostToDevice);

    float *dP; cudaMalloc(&dP, 128*32*sizeof(float));
    std::vector<float> h(128*32);
    probe<<<1,128>>>(dP, dQ);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h.data(), dP, 128*32*sizeof(float), cudaMemcpyDeviceToHost);

    constexpr float KSUM = 2080.0f;
    int bad = 0; float mn=1e30f, mx=-1e30f;
    for (int i = 0; i < 128*32; ++i) {
        if (h[i] <= 0 || fmodf(h[i], KSUM) != 0.0f || h[i] > 64*KSUM) bad++;
        mn = fminf(mn, h[i]); mx = fmaxf(mx, h[i]);
    }
    printf("gmem-A(64x64) stride64-B QK: bad=%d min=%g max=%g (expect 0, 2080, 133120)\n", bad, mn, mx);
    return 0;
}
