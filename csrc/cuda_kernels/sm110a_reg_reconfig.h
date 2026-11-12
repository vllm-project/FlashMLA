#pragma once

// The pinned CUTLASS turns warpgroup_reg_alloc/dealloc (setmaxnreg) into no-ops
// on sm_110, which the SM100 kernels dispatched on Thor rely on. Enable it for
// the sm_110a and sm_110f device passes only, as newer CUTLASS does. Include this
// before any CUTLASS header; drop it once the CUTLASS submodule lists SM110 in
// cutlass/arch/reg_reconfig.h.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1100 && !defined(CUDA_CTA_RECONFIG_ACTIVATED) && \
    (defined(__CUDA_ARCH_FEAT_SM110_ALL) || \
     (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && __CUDA_ARCH_FAMILY_SPECIFIC__ == 1100))
#define CUDA_CTA_RECONFIG_ACTIVATED 1
#endif
