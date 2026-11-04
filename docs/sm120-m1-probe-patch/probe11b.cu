// probe11b: decide the correct PV C-fragment idiom for N=256 split into 4x64
// segments on sm_120a register mma.sync (TMMA = SM80_16x8x16, AtomLayout (4,1,1)).
// Two candidates:
//   A) 4 independent (64,64) C fragments, B via local_tile of K-contig smem V.
//   C) whole-N single gemm (probe11 style) as numeric reference.
// Encoding: S (M,K) row-major with S(r,k)=k+1; V (N,K) K-contig, V(n,k)=n+1.
// Expect O(r,n) = sum_k (k+1)*(n+1) = (n+1)*1088.

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

// D: single (64,256) C fragment, gemm called per 64-N-block via local_tile on C
__global__ void k_pvD(float* out) {
    constexpr int NV = 256;
    __shared__ __align__(1024) bf16 sS_[64*64];
    extern __shared__ char dyn[];
    bf16* sV = reinterpret_cast<bf16*>(dyn);

    for (int i = threadIdx.x; i < 64*64; i += blockDim.x)
        sS_[i] = (bf16)((i%64)+1);
    for (int i = threadIdx.x; i < NV*64; i += blockDim.x)
        sV[i] = (bf16)((i/64)+1);
    __syncthreads();

    TMMA tmma{};
    auto thr = tmma.get_slice(threadIdx.x);

    auto sSt = make_tensor(make_smem_ptr(sS_), Layout<Shape<_64,_64>, Stride<_64,_1>>{});
    using NKV = Layout<Shape<Int<NV>,_64>, Stride<Int<64>,_1>>;
    auto sVt = make_tensor(make_smem_ptr(sV), NKV{});

    Tensor rS_a = thr.partition_fragment_A(sSt);
    cute::copy(thr.partition_A(sSt), rS_a);

    Tensor rO = partition_fragment_C(tmma, Shape<_64, Int<NV>>{});
    cute::clear(rO);
    for (int s = 0; s < 4; ++s) {
        auto sVseg = local_tile(sVt, Shape<_64, _64>{}, make_coord(s, _0{}));
        Tensor rV_b = thr.partition_fragment_B(sVseg);
        cute::copy(thr.partition_B(sVseg), rV_b);
        cute::gemm(tmma, rS_a, rV_b, local_tile(rO, Shape<_64, _64>{}, make_coord(_0{}, s)));
    }
    for (int i = 0; i < size(rO); ++i)
        out[threadIdx.x*size(rO) + i] = rO(i);
}

template<int WHICH>  // 0 = A (4 independent C), 1 = C (whole-N)
__global__ void k_pv(float* out) {
    constexpr int NV = 256;
    __shared__ __align__(1024) bf16 sS_[64*64];
    extern __shared__ char dyn[];
    bf16* sV = reinterpret_cast<bf16*>(dyn);

    for (int i = threadIdx.x; i < 64*64; i += blockDim.x)
        sS_[i] = (bf16)((i%64)+1);          // S(r,k)=k+1
    for (int i = threadIdx.x; i < NV*64; i += blockDim.x)
        sV[i] = (bf16)((i/64)+1);           // V(n,k)=n+1
    __syncthreads();

    TMMA tmma{};
    auto thr = tmma.get_slice(threadIdx.x);

    auto sSt = make_tensor(make_smem_ptr(sS_), Layout<Shape<_64,_64>, Stride<_64,_1>>{});
    using NKV = Layout<Shape<Int<NV>,_64>, Stride<Int<64>,_1>>;
    auto sVt = make_tensor(make_smem_ptr(sV), NKV{});

    Tensor rS_a = thr.partition_fragment_A(sSt);
    cute::copy(thr.partition_A(sSt), rS_a);

    if constexpr (WHICH == 1) {
        // C: whole-N single gemm (reference; huge B fragment, may spill)
        Tensor rV_b = thr.partition_fragment_B(sVt);
        cute::copy(thr.partition_B(sVt), rV_b);
        Tensor rO = partition_fragment_C(tmma, Shape<_64, Int<NV>>{});
        cute::clear(rO);
        cute::gemm(tmma, rS_a, rV_b, rO);
        for (int i = 0; i < size(rO); ++i)
            out[threadIdx.x*size(rO) + i] = rO(i);
    } else {
        // A: 4 independent (64,64) C fragments, accumulated separately
        // concatenated dump: thread t, segment s, element i -> out[(s*32+i)*128 + t]
        CUTE_UNROLL
        for (int s = 0; s < 4; ++s) {
            Tensor rO = partition_fragment_C(tmma, Shape<_64, _64>{});
            cute::clear(rO);
            auto sVseg = local_tile(sVt, Shape<_64, _64>{}, make_coord(s, _0{}));
            Tensor rV_b = thr.partition_fragment_B(sVseg);
            cute::copy(thr.partition_B(sVseg), rV_b);
            cute::gemm(tmma, rS_a, rV_b, rO);
            for (int i = 0; i < size(rO); ++i)
                out[(s*size(rO) + i)*128 + threadIdx.x] = rO(i);
        }
    }
}

int verify(float* h, int nregsA) {
    // decode (row,col): O(r,c) = (c+1)*1088, unique -> c = v/1088 - 1, r = (c+1)*1088 annotation
    int bad = 0; float mn=1e30f, mx=-1e30f;
    // A-idiom dump layout: out[(s*32+i)*128 + t]; element universe = 4*32*128 = 256 per thread total
    long long sumcheck = 0;
    constexpr float KSUM = 2080.0f;  // sum_{k=0..63} (k+1)
    for (int idx = 0; idx < 4*32*128; ++idx) {
        float v = h[idx];
        if (v <= 0 || fmodf(v, KSUM) != 0.0f) bad++;
        mn = fminf(mn, v); mx = fmaxf(mx, v);
    }
    printf("  bad=%d min=%g max=%g (expect min=2080 max=256*2080=%g)\n", bad, mn, mx, 256.0f*KSUM);
    return bad;
}

int main() {
    float *dOut;
    cudaMalloc(&dOut, 128*1024*sizeof(float));
    std::vector<float> h(128*1024);

    printf("== whole-N single gemm (reference) ==\n");
    k_pv<1><<<1,128, 256*64*2>>>(dOut);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("I1 KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h.data(), dOut, 128*32*sizeof(float), cudaMemcpyDeviceToHost); // first 128*32 floats = thread0..127 first 32 elems? layout: out[t*size + i]
    verify(h.data(), 32);

    printf("== 4 independent C fragments ==\n");
    k_pv<0><<<1,128, 256*64*2>>>(dOut);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("I2 KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h.data(), dOut, 4*32*128*sizeof(float), cudaMemcpyDeviceToHost);
    verify(h.data(), 32);

    printf("== D: single (64,256) C + local_tile N-slice ==\n");
    k_pvD<<<1,128, 256*64*2>>>(dOut);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("I3 KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    // whole-N dump layout: out[t*128*? ] -> out[t*128 + i] covers 128 elems/thread
    cudaMemcpy(h.data(), dOut, 128*128*sizeof(float), cudaMemcpyDeviceToHost);
    int bad = 0; float mn=1e30f, mx=-1e30f; constexpr float KSUM = 2080.0f;
    for (int t = 0; t < 128; ++t)
        for (int i = 0; i < 128; ++i) {
            float v = h[t*128 + i];
            if (v <= 0 || fmodf(v, KSUM) != 0.0f) bad++;
            mn = fminf(mn, v); mx = fmaxf(mx, v);
        }
    printf("  bad=%d min=%g max=%g (expect min=2080 max=%g)\n", bad, mn, mx, 256.0f*KSUM);
    printf("done\n");
    return 0;
}
