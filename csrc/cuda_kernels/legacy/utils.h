#pragma once

// The assertion macros the legacy kernels use. The 2026.09.30 release replaced them with
// kerutils' KU_* equivalents and dropped them from cuda_kernels/utils.h, which still provides
// `println`, `ceil_div`, TRAP_ONLY_DEVICE_ASSERT and RingBufferState.

#include <cstdio>
#include <cstdlib>

#include "cuda_kernels/utils.h"

#define CHECK_CUDA(call)                                                                                  \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                              \
        }                                                                                                 \
    } while(0)

#define CHECK_CUDA_KERNEL_LAUNCH() CHECK_CUDA(cudaGetLastError())


#define FLASH_ASSERT(cond)                                                                                \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            fprintf(stderr, "Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                 \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)


#define FLASH_DEVICE_ASSERT(cond)                                                                         \
    do {                                                                                                  \
        if (not (cond)) {                                                                                 \
            printf("Assertion failed (%s:%d): %s\n", __FILE__, __LINE__, #cond);                          \
            asm("trap;");                                                                                 \
        }                                                                                                 \
    } while(0)
