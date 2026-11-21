// probe11i: isolate the failed variable #2. probe11g showed gmem-A + B-stride-576
// fails; probe11h showed gmem-A + B-stride-64 works. Now test smem-A + B stride-576
// with 9-seg local_tile (the kernel's exact QK shape).
// P(r,c) = sum_k (k+1)*(c+1) = (c+1)*2080 (Q(r,k)=k+1, K(c,k)=c+1)
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

__global__ void probe(float* outP) {
    constexpr int HEAD_DIM_K = 576;
    extern __shared__ char dyn[];
    bf16* smem = reinterpret_cast<bf16*>(dyn);
    bf16* sQ = smem;                      // (64,576)
    bf16* sK = smem + 64*HEAD_DIM_K;      // (64,576)
    for (int i = threadIdx.x; i < 64*HEAD_DIM_K; i += blockDim.x) {
        sQ[i] = (bf16)((i%HEAD_DIM_K)+1); // Q(r,k)=k+1
        sK[i] = (bf16)((i/64)+1);         // K(c,k)=c+1
    }
    __syncthreads();

    TMMA tmma{};
    auto thr = tmma.get_slice(threadIdx.x);

    using SKK = Layout<Shape<Int<64>, Int<HEAD_DIM_K>>, Stride<Int<HEAD_DIM_K>,_1>>;
    auto sQt = make_tensor(make_smem_ptr(sQ), SKK{});
    auto sKt = make_tensor(make_smem_ptr(sK), SKK{});

    Tensor rP = partition_fragment_C(tmma, Shape<_64,_64>{});
    cute::clear(rP);
    CUTE_NO_UNROLL
    for (int seg = 0; seg < HEAD_DIM_K/64; ++seg) {
        auto sQseg = local_tile(sQt, Shape<_64,_64>{}, make_coord(_0{}, seg));
        auto sKseg = local_tile(sKt, Shape<Int<64>,_64>{}, make_coord(_0{}, seg));
        Tensor rQa = thr.partition_fragment_A(sQseg);
        Tensor rKb = thr.partition_fragment_B(sKseg);
        cute::copy(thr.partition_A(sQseg), rQa);
        cute::copy(thr.partition_B(sKseg), rKb);
        cute::gemm(tmma, rQa, rKb, rP);
    }
    for (int i = 0; i < size(rP); ++i)
        outP[threadIdx.x*size(rP) + i] = rP(i);
}

int main() {
    float *dP; cudaMalloc(&dP, 128*32*sizeof(float));
    std::vector<float> h(128*32);
    size_t smem = 2*64*576*sizeof(bf16);
    cudaFuncSetAttribute((const void*)probe, cudaFuncAttributeMaxDynamicSharedMemorySize, 101376);
    if (smem > 101376) { printf("smem %zu exceeds 101376, cannot test\n", smem); return 1; }
    probe<<<1,128, smem>>>(dP);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h.data(), dP, 128*32*sizeof(float), cudaMemcpyDeviceToHost);

    constexpr float KSUM = 2080.0f;
    int bad = 0; float mn=1e30f, mx=-1e30f;
    for (int i = 0; i < 128*32; ++i) {
        if (h[i] <= 0 || fmodf(h[i], KSUM) != 0.0f || h[i] > 64*KSUM) bad++;
        mn = fminf(mn, h[i]); mx = fmaxf(mx, h[i]);
    }
    printf("smem-A + stride-576 local_tile B QK: bad=%d min=%g max=%g (expect 0, 2080, 133120)\n", bad, mn, mx);
    return 0;
}
