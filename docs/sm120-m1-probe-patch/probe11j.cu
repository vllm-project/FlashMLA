// probe11j: B is (64,576) stride 576, read via 9x local_tile (64,64) K-contiguous,
// A comes from global as a plain (64,64) (proven OK by probe11h). If this FAILS,
// the B stride-576 local_tile read is the broken piece.
// Encoding: A(r2,k)=k+1 (reused per seg), B(c,k)=c+1 independent of seg.
// P(r,c) = sum over 9 segs of sum_k (k+1)*(c+1) = 9 * 2080 * (c+1) = 18720*(c+1).
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

__global__ void probe(float* outP, bf16* gA) {
    constexpr int HEAD_DIM_K = 576;
    extern __shared__ char dyn[];
    bf16* sK = reinterpret_cast<bf16*>(dyn);
    for (int i = threadIdx.x; i < 64*HEAD_DIM_K; i += blockDim.x)
        sK[i] = (bf16)((i/64)+1);          // K(c,k)=c+1
    __syncthreads();

    TMMA tmma{};
    auto thr = tmma.get_slice(threadIdx.x);

    auto gAt = make_tensor(make_gmem_ptr(gA), Layout<Shape<_64,_64>, Stride<_64,_1>>{});
    using SKK = Layout<Shape<Int<64>, Int<HEAD_DIM_K>>, Stride<Int<HEAD_DIM_K>,_1>>;
    auto sKt = make_tensor(make_smem_ptr(sK), SKK{});

    Tensor rQa = thr.partition_fragment_A(gAt);
    cute::copy(thr.partition_A(gAt), rQa);
    Tensor rP = partition_fragment_C(tmma, Shape<_64,_64>{});
    cute::clear(rP);
    CUTE_NO_UNROLL
    for (int seg = 0; seg < HEAD_DIM_K/64; ++seg) {
        auto sKseg = local_tile(sKt, Shape<Int<64>,_64>{}, make_coord(_0{}, seg));
        Tensor rKb = thr.partition_fragment_B(sKseg);
        cute::copy(thr.partition_B(sKseg), rKb);
        cute::gemm(tmma, rQa, rKb, rP);
    }
    for (int i = 0; i < size(rP); ++i)
        outP[threadIdx.x*size(rP) + i] = rP(i);
}

int main() {
    std::vector<bf16> hA(64*64);
    for (int i = 0; i < 64*64; ++i) hA[i] = (bf16)((i%64)+1);   // A(r,k)=k+1
    bf16 *dA;
    cudaMalloc(&dA, 64*64*sizeof(bf16));
    cudaMemcpy(dA, hA.data(), 64*64*sizeof(bf16), cudaMemcpyHostToDevice);

    float *dP; cudaMalloc(&dP, 128*32*sizeof(float));
    std::vector<float> h(128*32);
    size_t smem = 64*576*sizeof(bf16);
    cudaFuncSetAttribute((const void*)probe, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    probe<<<1,128, smem>>>(dP, dA);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h.data(), dP, 128*32*sizeof(float), cudaMemcpyDeviceToHost);

    constexpr float KSUM = 9.0f*2080.0f;   // 18720
    int bad = 0; float mn=1e30f, mx=-1e30f;
    int colok[64] = {0}; int coltot[64] = {0};
    for (int t = 0; t < 128; ++t) {
        int base = 16*(t/32) + (t%32)/4;
        for (int g = 0; g < 32; ++g) {
            float v = h[t*32+g];
            int col = 8*(g/4) + 2*(t%4) + (g%4)%2;
            int ok = (v > 0 && fmodf(v, KSUM) == 0.0f && v <= 64*KSUM);
            colok[col] += ok; coltot[col]++;
            if (!ok) { if (bad < 20) printf("t%03d g%02d col%02d val=%g (expect %g)\n", t, g, col, v, KSUM*(col+1)); bad++; }
            mn = fminf(mn, v); mx = fmaxf(mx, v);
        }
    }
    int goodcols = 0;
    for (int c = 0; c < 64; ++c) if (colok[c] == coltot[c] && coltot[c]>0) goodcols++;
    printf("bad=%d min=%g max=%g good-col-count=%d/64 (expect 0,18720,%g,64)\n", bad, mn, mx, goodcols, 64.f*KSUM);
    printf("first col tallies:"); for (int c=0;c<16;++c) printf(" %d/%d", colok[c], coltot[c]); printf("\n");
    return 0;
}
