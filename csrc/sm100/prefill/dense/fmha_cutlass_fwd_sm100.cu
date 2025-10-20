// SM100-specific file - only compile for SM100+ architectures
#include "interface.h"
#include <stdexcept>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) || !defined(__CUDA_ARCH__)

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#include "common/mask.cuh"
#include "common/utils.hpp"

#include "fmha_cutlass_fwd_sm100.cuh"

template <class Mask, class Varlen, class Element, class ElementOut, class Mla>
void call_run_fmha_fwd([[maybe_unused]] Mask mask, [[maybe_unused]] Varlen is_varlen,
                       [[maybe_unused]] Element in, [[maybe_unused]] ElementOut out,
                       [[maybe_unused]] Mla mla, at::Tensor workspace_buffer, at::Tensor q,
                       at::Tensor k, at::Tensor v, at::Tensor cumulative_seqlen_q,
                       at::Tensor cumulative_seqlen_kv, at::Tensor o, at::Tensor lse,
                       float softmax_scale, int max_seqlen_q, int max_seqlen_kv) {
  static constexpr bool IsVarlen = std::is_same_v<Varlen, true_type>;
  static constexpr bool IsMla = std::is_same_v<Mla, true_type>;
  static constexpr bool IsCausalMask = std::is_same_v<Mask, CausalMask<false>>;
  using Option =
      std::conditional_t<IsCausalMask || (IsVarlen), Option<Tag::kIsPersistent, false_type>,
                         Option<Tag::kIsPersistent, true_type>>;

  run_fmha_fwd<Element, ElementOut, IsVarlen, IsMla, Mask, Option>(
      workspace_buffer, q, k, v, cumulative_seqlen_q, cumulative_seqlen_kv, o, lse,
      softmax_scale, max_seqlen_q, max_seqlen_kv);
}

void FMHACutlassSM100FwdRun(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k,
                            at::Tensor v, at::Tensor cumulative_seqlen_q,
                            at::Tensor cumulative_seqlen_kv, at::Tensor o, at::Tensor lse,
                            int mask_mode_code, float sm_scale, int max_seqlen_q,
                            int max_seqlen_kv, bool is_varlen) {
  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  CHECK(q.scalar_type() == k.scalar_type());
  auto scalar_type_in = q.scalar_type();
  auto scalar_type_out = o.scalar_type();
  int head_dim_qk = q.size(-1);
  int head_dim_vo = v.size(-1);
  MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  if (scalar_type_in == at::ScalarType::BFloat16 &&
      scalar_type_out == at::ScalarType::BFloat16) {
    using Element = cutlass::bfloat16_t;
    using ElementOut = cutlass::bfloat16_t;

    auto apply_config = [&](auto fn) {
      if (mask_mode == MaskMode::kCausal) {
        if (is_varlen) {
          fn(CausalMask<false>{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(CausalMask<false>{}, cute::false_type{}, Element{}, ElementOut{});
        }
      } else {
        if (is_varlen) {
          fn(ResidualMask{}, cute::true_type{}, Element{}, ElementOut{});
        } else {
          fn(ResidualMask{}, cute::false_type{}, Element{}, ElementOut{});
        }
      }
    };

    apply_config([&](auto mask, auto varlen, auto in, auto out) {
      if (head_dim_qk == 192 && head_dim_vo == 128) {
        call_run_fmha_fwd(mask, varlen, in, out, true_type{}, workspace_buffer, q, k, v,
                          cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale,
                          max_seqlen_q, max_seqlen_kv);
      } else if (head_dim_qk == 128 && head_dim_vo == 128) {
        call_run_fmha_fwd(mask, varlen, in, out, false_type{}, workspace_buffer, q, k, v,
                          cumulative_seqlen_q, cumulative_seqlen_kv, o, lse, sm_scale,
                          max_seqlen_q, max_seqlen_kv);
      } else {
        std::cout << "No kernel instantiated for head_dim_qk=" << head_dim_qk
                  << " head_dim_vo=" << head_dim_vo << std::endl;
      }
    });

  } else {
    FLASH_MLA_ASSERT(false);
  }
}

#else // !SM100+ architecture

void FMHACutlassSM100FwdRun(at::Tensor workspace_buffer, at::Tensor q, at::Tensor k,
                           at::Tensor v, at::Tensor cumulative_seqlen_q,
                           at::Tensor cumulative_seqlen_kv, at::Tensor o, at::Tensor lse,
                           int mask_mode_code, float sm_scale, int max_seqlen_q,
                           int max_seqlen_kv, bool is_varlen) {
    throw std::runtime_error("FlashMLA dense prefill requires SM100+ architecture. This build was compiled without SM100 support.");
}

void FMHACutlassSM100BwdRun(at::Tensor workspace_buffer, at::Tensor d_o, at::Tensor q, at::Tensor k,
                           at::Tensor v, at::Tensor o, at::Tensor lse,
                           at::Tensor cumulative_seqlen_q, at::Tensor cumulative_seqlen_kv,
                           at::Tensor dq, at::Tensor dk, at::Tensor dv,
                           int mask_mode_code, float softmax_scale, int max_seqlen_q, int max_seqlen_kv, bool is_varlen) {
    throw std::runtime_error("FlashMLA dense prefill backward requires SM100+ architecture. This build was compiled without SM100 support.");
}

#endif // SM100+ architecture check
