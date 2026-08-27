#pragma once

#include <cute/tensor.hpp>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include "defines.h"

namespace sm100 {

using namespace cute;

CUTE_DEVICE
int int4_max(int4 t) {
    return max(max(t.x, t.y), max(t.z, t.w));
}

CUTE_DEVICE
int int4_min(int4 t) {
    return min(min(t.x, t.y), min(t.z, t.w));
}

// Convert 2x fp8_e4m3 to 2x bf16 with scaling
CUTE_DEVICE
nv_bfloat162 fp8x2_to_bf16x2_with_scale(__nv_fp8x2_e4m3 data, nv_bfloat16 scale) {
    // TODO Use native conversion for CUDA >= 13.1
    float2 data_float2 = (float2)data;
    nv_bfloat162 data_bf16x2 = __float22bfloat162_rn(data_float2);
    return nv_bfloat162 {
        data_bf16x2.x * scale,
        data_bf16x2.y * scale
    };
}

// Convert 2x fp8_e4m3 to 2x bf16 (no scaling). Exact: e4m3 values are a subset of bf16.
CUTE_DEVICE
nv_bfloat162 fp8x2_to_bf16x2(__nv_fp8x2_e4m3 data) {
    return __float22bfloat162_rn((float2)data);
}

// Convert 1x fp8_e4m3 (a scale factor) to bf16. Exact: e4m3 values are a subset of bf16.
CUTE_DEVICE
nv_bfloat16 fp8_e4m3_to_bf16(uint8_t data) {
    __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)data, __NV_E4M3);
    return __float2bfloat16_rn(__half2float(*(__half*)&h));
}

// Convert 8x fp4_e2m1 (packed in a uint32, low nibble = even element) to 8x bf16 with scaling.
// The e2m1*scale product is exactly representable in bf16 (<= 5 mantissa bits), so the
// bf16 multiply below is exact.
CUTE_DEVICE
void fp4x8_to_bf16x8_with_scale(uint32_t data, nv_bfloat16 scale, nv_bfloat162 out[4]) {
    nv_bfloat162 scale2 = {scale, scale};
    CUTE_UNROLL
    for (int i = 0; i < 4; ++i) {
        // Native cvt.rn.f16x2.e2m1x2 on sm_100f
        __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)(data >> (8*i)), __NV_E2M1);
        float2 f2 = __half22float2(*(__half2*)&h2);
        out[i] = __hmul2(__float22bfloat162_rn(f2), scale2);
    }
}

}
