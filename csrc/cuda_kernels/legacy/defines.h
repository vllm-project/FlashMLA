#pragma once

// Additions the legacy kernels need on top of the shared definitions: the `fp8` spelling and
// the small vector types that the 2026.09.30 release dropped from cuda_kernels/defines.h.

#include <cuda_bf16.h>

#include "cuda_kernels/defines.h"

using fp8 = fp8_e4m3;

struct int32x8_t {
    int a0, a1, a2, a3, a4, a5, a6, a7;
};

struct float8 {
    float2 a01, a23, a45, a67;
};

struct bf16x8 {
    __nv_bfloat162 a01;
    __nv_bfloat162 a23;
    __nv_bfloat162 a45;
    __nv_bfloat162 a67;
};
