// cp.async helpers shared by the SM120 decode kernels.
#pragma once

#include <cstdint>

namespace sm120 {

// 16 B page copy. The 4th operand is the SOURCE SIZE, not a predicate: 0
// zero-fills the destination without reading, which is what the out-of-range
// rows of a partial last page need (probe 02 measured both forms).
// .ca rather than .cg: the 16 B-per-lane pattern is served well from L1, and
// .cg measured 13% slower on the dense shape.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int src_size) {
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16, %2;"
                 :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(src_size));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }

template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }

}  // namespace sm120
