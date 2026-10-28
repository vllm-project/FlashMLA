// probe11d: is the g-index of the QK C-fragment (rP) layout-identical to the g-index
// of the PV A-fragment (rS)? scale_softmax writes rS(g) = softmax(rP(g)), and PV
// gemms with rS as A -- so same g must address the same row. Otherwise every softmax
// value lands on the wrong row and O is scrambled while the per-row max/sum (LSE)
// stays correct -- exactly the observed symptom (lse ok, out wrong).
// Method: fill a (64,64) smem A with A(r,k)=r*64+k and dump the A fragment; fill a
// (64,64) smem C with C(r,n)=r*64+n and dump the C fragment. Compare rows per g.

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

__global__ void probe(float* outA, float* outC) {
    __shared__ __align__(1024) bf16 sMat[64*64];
    for (int i = threadIdx.x; i < 64*64; i += blockDim.x)
        sMat[i] = (bf16)(i);   // i = r*64 + c  (both A and C use same encoding)
    __syncthreads();

    TMMA tmma{};
    auto thr = tmma.get_slice(threadIdx.x);
    auto sT = make_tensor(make_smem_ptr(sMat), Layout<Shape<_64,_64>, Stride<_64,_1>>{});

    Tensor rS_a = thr.partition_fragment_A(sT);
    cute::copy(thr.partition_A(sT), rS_a);
    for (int i = 0; i < size(rS_a); ++i)
        outA[threadIdx.x*size(rS_a) + i] = (float)rS_a(i);

    Tensor rC = partition_fragment_C(tmma, Shape<_64,_64>{});
    // C not directly a gemm product here; emulate by reading sT through the C partition
    // is not possible -- instead, use a qk-style gemm with identity-ish inputs to fill rC.
    // Simpler: construct via copy from a smem tensor using partition_fragment_C is invalid.
    // Use a real QK gemm with A(r,k)=r (row), B favorite: choose B(N,K) with B(n,k)=n.
    // Then P(r,n) = sum_k (r)*(n) = 64*r*n; but we need row *and* col decode -> encode
    // row in A= r+1, col in B=n+1 -> P = 64*(r+1)*(n+1) on the C fragment, decodeable.
    __shared__ __align__(1024) bf16 sAq[64*64];
    __shared__ __align__(1024) bf16 sBq[64*64];
    for (int i = threadIdx.x; i < 64*64; i += blockDim.x) {
        sAq[i] = (bf16)((i/64)+1);        // A(r,k)=r+1
        sBq[i] = (bf16)((i/64)+1);        // B(n,k)=n+1 (NKContig row-major (N,K))
    }
    __syncthreads();
    auto sAt = make_tensor(make_smem_ptr(sAq), Layout<Shape<_64,_64>, Stride<_64,_1>>{});
    using NK = Layout<Shape<_64,_64>, Stride<Int<64>,_1>>;
    auto sBt = make_tensor(make_smem_ptr(sBq), NK{});
    Tensor rA = thr.partition_fragment_A(sAt);
    Tensor rB = thr.partition_fragment_B(sBt);
    cute::copy(thr.partition_A(sAt), rA);
    cute::copy(thr.partition_B(sBt), rB);
    cute::clear(rC);
    cute::gemm(tmma, rA, rB, rC);
    for (int i = 0; i < size(rC); ++i)
        outC[threadIdx.x*size(rC) + i] = rC(i);
}

int main() {
    float *dA, *dC;
    cudaMalloc(&dA, 128*32*sizeof(float));
    cudaMalloc(&dC, 128*32*sizeof(float));
    probe<<<1,128>>>(dA, dC);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("KERNEL ERROR: %s\n", cudaGetErrorString(e)); return 1; }
    std::vector<float> hA(128*32), hC(128*32);
    cudaMemcpy(hA.data(), dA, 128*32*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hC.data(), dC, 128*32*sizeof(float), cudaMemcpyDeviceToHost);

    // decode A element: value v = r*64 + k  (row = v/64)  -- same encoding as rS would get at (r,k)
    // decode C element: value 64*(r+1)*(n+1) -> find (r,n): n = v/64/(r+1) - 1  unknown r; instead
    //   since r = row = (c%4>=2)*8 + base, and value v = 64*(r+1)*(n+1): iterate r over the two
    //   candidate rows {base, base+8}, check divisibility.
    int badA = 0, badC = 0;
    for (int t = 0; t < 128; ++t) {
        int base = 16*(t/32) + (t%32)/4;
        for (int g = 0; g < 32; ++g) {
            // --- A decode ---
            float va = hA[t*32+g];
            int rowA = (int)(va/64.0f);
            // --- C decode ---
            float vc = hC[t*32+g];
            int colC = -1, rowC = -1;
            int rsel = (g%4) >= 2 ? 8 : 0;
            int r0 = base + rsel;
            for (int rr = 0; rr < 2; ++rr) {
                int r = base + (rr ? 8 : 0);
                if (r != r0) continue;  // g gives us the row candidate directly
                if (vc <= 0) break;
                float nn = vc/(64.0f*(r+1)) - 1.0f;
                if (nn >= -0.5f && nn == (float)(int)(nn+0.5f)) { rowC = r; colC = (int)(nn+0.5f); }
            }
            // row alignment check: same g must map to same row
            if (rowA != rowC) {
                if (badA < 8)
                    printf("T%03d g%02d: A_row=%d C_row=%d C_col=%d (vA=%g vC=%g)\n", t, g, rowA, rowC, colC, va, vc);
                badA++;
            }
            if (vc > 0 && colC < 0) badC++;
        }
    }
    printf("row-mismatches=%d  undecodable-C=%d  (0/0 expected if A and C fragment rows are g-aligned)\n", badA, badC);
    return 0;
}
