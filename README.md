# FlashMLA

## Introduction

FlashMLA is DeepSeek's library of optimized attention kernels, powering the [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) and [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) models. This repository contains the following implementations:

**Sparse Attention Kernels**

*These kernels power DeepSeek Sparse Attention (DSA), as introduced in [this paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).*

- Token-level sparse attention for the prefill stage
- Token-level sparse attention for the decoding stage, with FP8 KV cache

**Dense Attention Kernels**

- Dense attention for the prefill stage
- Dense attention for the decoding stage

## Changes in the vLLM fork

This repository is [vLLM](https://github.com/vllm-project/vllm)'s fork of [deepseek-ai/FlashMLA](https://github.com/deepseek-ai/FlashMLA). It tracks upstream (currently synced through [deepseek-ai/FlashMLA@07a1089](https://github.com/deepseek-ai/FlashMLA/commit/07a1089857b63e74e3133630c02b083b75e8d4b2), which includes the DeepSeek V4.1 kernels) and adds the following on top of it. vLLM compiles these sources directly through its `cmake/external_projects/flashmla.cmake`; set `FLASH_MLA_SRC_DIR` to build vLLM against a local checkout.

- **PyTorch stable ABI.** The API layer in `csrc/api/` uses the libtorch stable ABI (`torch/csrc/stable`, `STABLE_TORCH_LIBRARY`) instead of `torch/extension.h` and pybind11. Operators are registered under `torch.ops._flashmla_C`, and the module defines `PyInit__flashmla_C` so that vLLM can import it as `vllm._flashmla_C`. The extension is built against CPython's limited API, so a single `abi3` wheel works for every CPython >= 3.10; it requires PyTorch >= 2.10 at runtime. `flash_mla/flash_mla_interface.py` calls the operators through `torch.ops._flashmla_C` and is vendored into vLLM as `vllm/third_party/flashmla/flash_mla_interface.py`.
- **Registered operators.** `sparse_decode_fwd`, `dense_decode_fwd`, `sparse_prefill_fwd`, `dense_prefill_fwd`, `fused_norm_rope_attn_rope_cast_fwd`, `fused_norm_rope_attn_rope_cast_decode`, `permute_q_b_proj` and `permute_wv_proj`. `dense_prefill_bwd` is registered only when the extension is compiled with `FLASH_MLA_ENABLE_DENSE_BWD` (set by the standalone `setup.py`; vLLM's inference-only build leaves it out).
- **Optional output buffers.** `flash_mla_with_kvcache(..., out=...)` (sparse and dense decoding) and `flash_mla_sparse_fwd(..., out=...)` write into a caller-provided tensor instead of allocating a new one.
- **Dense FP8 KV-cache decoding on SM90.** `csrc/extension/sm90/dense_fp8/` contains vLLM's Hopper MLA decoding kernel for FP8 KV caches. vLLM builds it as a separate `_flashmla_extension_C` module (`fwd_kvcache_mla_fp8`, `get_mla_decoding_metadata_dense_fp8`); it is not part of the standalone `setup.py` build.
- **NVFP4 KV cache for sparse decoding on SM100.** Besides the 656-byte FP8 layout, sparse decoding with `head_dim == 576` accepts a 352-byte-per-token layout: 256 bytes of e2m1 NoPE values, 64 bytes of unscaled e4m3 RoPE values and 32 e4m3 scale factors (one per 16 NoPE values). The layout is detected from `k_cache.shape[-1]`. See `KVCacheLayout.V32_NVFP4_FP8ROPE` in `tests/quant.py` for the exact wire format and `csrc/kernels/sm100/decode/sparse/nvfp4_head64/` for the kernel; 64 and 128 query heads are supported.
- **Robustness fixes.** Thread-safe cached device properties and stable-ABI stream and tensor helpers in `csrc/kerutils/include/kerutils/supplemental/`, plus fixes to the dense FP8 decoding metadata (for example `num_sm_parts` is clamped to at least 1).
- **Tests.** `tests/test_api_registration.py`, `tests/test_output_buffer_api.py` and `tests/test_nvfp4_quant.py`, plus NVFP4 cases in `tests/test_flash_mla_sparse_decoding.py`.

### Detailed diff against upstream

Everything below is the complete set of files that differ from upstream; regenerate it with `git diff --stat 07a1089857b63e74e3133630c02b083b75e8d4b2 HEAD -- . ':!README.md'`. Files that are not listed are identical to upstream.

| Area | Files | Difference from upstream |
| :--- | :--- | :--- |
| Operator registration | `csrc/api/api.cpp`, `csrc/api/interfaces.h` (new), `csrc/api/dense_fwd.cpp` and `csrc/api/dense_bwd.cpp` (removed) | The pybind11 `PYBIND11_MODULE` and per-file `register_*` shims are replaced by `STABLE_TORCH_LIBRARY` / `STABLE_TORCH_LIBRARY_IMPL` with explicit operator schemas. All interface functions are declared once in `interfaces.h`. `dense_prefill_bwd` is only registered under `FLASH_MLA_ENABLE_DENSE_BWD`, and `PyInit__flashmla_C` lets vLLM import the library as a Python module. |
| API implementations | `csrc/api/sparse_decode.cpp`, `csrc/api/dense_decode.cpp`, `csrc/api/sparse_prefill.cpp`, `csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp`, `csrc/api/common.h` | `at::Tensor`, `TORCH_CHECK`, `torch::empty`, `at::cuda::CUDAGuard` and `at::cuda::getCurrentCUDAStream` are replaced by `torch::stable::Tensor`, `STD_TORCH_CHECK`, `torch::stable::new_empty`, `torch::stable::accelerator::DeviceGuard` and the stable stream helper. Scalar arguments are widened to `int64_t` / `double` as stable schemas require. Sparse decode, dense decode and sparse prefill take an optional `out_` buffer. `Arch` reads the cached device properties. Sparse decode detects the NVFP4 layout from bytes per token (`detect_kv_cache_format_for_headdim_576`), advertises `NVFP4_FP8ROPE_KVCACHE_FORMAT` on the SM100 head-64 and head-64x2 implementations, and always uses split-KV scheduling for it. |
| Dense MHA prefill entry points | `csrc/kernels/sm100/prefill/dense/interface.h`, `fmha_cutlass_fwd_sm100.cu` / `.cuh`, `fmha_cutlass_bwd_sm100.cu` / `.cuh`, `common/utils.hpp` | `FMHACutlassSM100FwdRun` and `FMHACutlassSM100BwdRun` take `torch::stable::Tensor` and `int64_t` / `double` scalars so they can be registered directly as stable operators. |
| Stable-ABI helpers | `csrc/kerutils/include/kerutils/supplemental/cuda_stream.h`, `device_prop.h`, `torch_tensors.h` | `get_current_cuda_stream(tensor)` through the AOTI shim, a thread-safe per-device `cudaDeviceProp` cache (`std::once_flag`) replacing `at::cuda::getCurrentDeviceProperties()`, and the `KU_CHECK_*` / `get_optional_tensor_ptr` helpers rewritten over `torch::stable::Tensor`. |
| NVFP4 KV cache (SM100 sparse decode) | `csrc/kernels/params.h`, `csrc/kernels/kv_cache_format.h`, `csrc/kernels/sm100/decode/sparse/nvfp4_head64/config.h`, `kernel.cuh`, `kernel.h`, `instantiations/v32_nvfp4_fp8rope.cu`, `csrc/kernels/sm100/helpers.h` | New `ModelType::V32_NVFP4_FP8ROPE`; `KVCacheFormat` describes the 352-byte record (256 B e2m1 NoPE, 64 B e4m3 RoPE, 32 B e4m3 scales) and `kv_cache_bytes_per_token` returns it. The kernel is a dedicated head-64 decode kernel, derived from the pre-V4.1 SM100 head-64 kernel and extended with e2m1 dequantization; `helpers.h` gains a bf16-scale overload of `fp8x2_to_bf16x2_with_scale` that it uses. |
| SM90 dense FP8 decoding extension | `csrc/extension/torch_api.cpp`, `csrc/extension/sm90/dense_fp8/*` | vLLM-only sources for the `_flashmla_extension_C` module (`fwd_kvcache_mla_fp8`, `get_mla_decoding_metadata_dense_fp8`), already ported to the stable ABI. Not compiled by `setup.py`; vLLM's CMake builds them. |
| Python package | `flash_mla/__init__.py`, `flash_mla/flash_mla_interface.py`, `flash_mla/fused_norm_rope_attn_rope_cast.py` | `__init__` loads `_flashmla_C*.so` with `torch.ops.load_library`; call sites use `torch.ops._flashmla_C` instead of the `flash_mla.cuda` pybind module; `flash_mla_with_kvcache` and `flash_mla_sparse_fwd` accept `out=` and document the NVFP4 layout. |
| Build | `setup.py`, `.gitignore` | The extension is named `flash_mla._flashmla_C`; it is compiled with `-DTORCH_TARGET_VERSION=0x020a000000000000 -DUSE_CUDA -DFLASH_MLA_ENABLE_DENSE_BWD`, `py_limited_api=True` and `bdist_wheel.py_limited_api = cp310`; the NVFP4 instantiation is added and the removed `dense_fwd.cpp` / `dense_bwd.cpp` are dropped from the source list; `.venv/` is ignored. |
| Tests | `tests/lib.py`, `tests/quant.py`, `tests/test_flash_mla_sparse_decoding.py`, `tests/test_api_registration.py`, `tests/test_output_buffer_api.py`, `tests/test_nvfp4_quant.py` | `KVCacheLayout.V32_NVFP4_FP8ROPE` quantization and dequantization (including the scale-byte permutation) and per-layout byte accounting for the bandwidth numbers; NVFP4 correctness, corner and performance cases; new tests for operator registration, `out=` forwarding and the NVFP4 wire format. |

<details>
<summary>File-level diffstat against deepseek-ai/FlashMLA@07a1089</summary>

```text
 .gitignore                                                                        |    1 +
 csrc/api/api.cpp                                                                  |   55 ++++--
 csrc/api/common.h                                                                 |   69 ++++---
 csrc/api/dense_bwd.cpp                                                            |    9 -
 csrc/api/dense_decode.cpp                                                         |  151 ++++++++-------
 csrc/api/dense_fwd.cpp                                                            |    9 -
 csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp                                   |  281 ++++++++++++++--------------
 csrc/api/interfaces.h                                                             |  102 ++++++++++
 csrc/api/sparse_decode.cpp                                                        |  183 ++++++++++--------
 csrc/api/sparse_prefill.cpp                                                       |   71 +++----
 csrc/extension/sm90/dense_fp8/flash_fwd_mla_fp8_sm90.cu                           |   10 +
 csrc/extension/sm90/dense_fp8/flash_fwd_mla_kernel.h                              |  709 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 csrc/extension/sm90/dense_fp8/flash_fwd_mla_metadata.cu                           |   77 ++++++++
 csrc/extension/sm90/dense_fp8/flash_mla.h                                         |   85 +++++++++
 csrc/extension/sm90/dense_fp8/fp8_transpose_v.h                                   |   89 +++++++++
 csrc/extension/sm90/dense_fp8/named_barrier.h                                     |   21 +++
 csrc/extension/sm90/dense_fp8/pybind.cpp                                          |  246 +++++++++++++++++++++++++
 csrc/extension/sm90/dense_fp8/softmax.h                                           |  202 ++++++++++++++++++++
 csrc/extension/sm90/dense_fp8/static_switch.h                                     |   70 +++++++
 csrc/extension/sm90/dense_fp8/utils.h                                             |  279 ++++++++++++++++++++++++++++
 csrc/extension/torch_api.cpp                                                      |   47 +++++
 csrc/kernels/kv_cache_format.h                                                    |   19 +-
 csrc/kernels/params.h                                                             |    5 +-
 csrc/kernels/sm100/decode/sparse/head64/config.h                                  |    2 +-
 csrc/kernels/sm100/decode/sparse/nvfp4_head64/config.h                            |  270 +++++++++++++++++++++++++++
 csrc/kernels/sm100/decode/sparse/nvfp4_head64/instantiations/v32_nvfp4_fp8rope.cu |    8 +
 csrc/kernels/sm100/decode/sparse/nvfp4_head64/kernel.cuh                          | 1103 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 csrc/kernels/sm100/decode/sparse/nvfp4_head64/kernel.h                            |   10 +
 csrc/kernels/sm100/helpers.h                                                      |    8 +
 csrc/kernels/sm100/prefill/dense/common/utils.hpp                                 |    1 -
 csrc/kernels/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu                        |   27 +--
 csrc/kernels/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cuh                       |   50 ++---
 csrc/kernels/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu                        |   29 +--
 csrc/kernels/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cuh                       |   39 ++--
 csrc/kernels/sm100/prefill/dense/interface.h                                      |   20 +-
 csrc/kerutils/include/kerutils/supplemental/cuda_stream.h                         |   19 ++
 csrc/kerutils/include/kerutils/supplemental/device_prop.h                         |   56 ++++++
 csrc/kerutils/include/kerutils/supplemental/torch_tensors.h                       |   29 +--
 flash_mla/__init__.py                                                             |   10 +
 flash_mla/flash_mla_interface.py                                                  |   25 ++-
 flash_mla/fused_norm_rope_attn_rope_cast.py                                       |    4 +-
 setup.py                                                                          |   21 ++-
 tests/lib.py                                                                      |   10 +-
 tests/quant.py                                                                    |   78 +++++++-
 tests/test_api_registration.py                                                    |   24 +++
 tests/test_flash_mla_sparse_decoding.py                                           |   49 +++++
 tests/test_nvfp4_quant.py                                                         |   28 +++
 tests/test_output_buffer_api.py                                                   |   87 +++++++++
 48 files changed, 4306 insertions(+), 491 deletions(-)
```

</details>

## News

- **2026.09.10 Release of DeepSeek v4.1's Attention Kernels**: We've released attention kernels for [DeepSeek-V4.1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), including both prefill and decoding (with FP8 or FP4 KV cache). We've also released a [fused-norm-rope-attn-rope-cast kernel](#fused-norm--rope--attn--rope--cast-kernel) which fuses Q-norm (only used in V4, not V4.1), Q-RoPE, core attention, O-RoPE (conjugate), and cast-to-fp8, while retaining the same performance.
- **2025.09.29 Release of Sparse Attention Kernels**: With the launch of [DeepSeek-V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp), we are releasing the corresponding token-level sparse attention kernels. These kernels power the model's DeepSeek Sparse Attention (DSA) and achieve up to 640 TFlops during prefilling and 410 TFlops during decoding. We also release a deep-dive blog for our new FP8 sparse decoding kernel. Check it out [here](docs/20250929-hopper-fp8-sparse-deep-dive.md).
- **2025.08.01 Kernels for MHA on SM100**: Thanks to [NVIDIA's PR](https://github.com/deepseek-ai/FlashMLA/pull/76) for MHA forward / backward kernels on SM100!
- **2025.04.22 Deep-Dive Blog**: We'd love to share the technical details behind the new FlashMLA kernel! Check out our deep-dive write-up [here](docs/20250422-new-kernel-deep-dive.md).
- **2025.04.22 Performance Update**: We're excited to announce the new release of Flash MLA, which delivers 5% ~ 15% performance improvement for compute-bound workloads, achieving up to 660 TFlops on NVIDIA H800 SXM5 GPUs. The interface of the new version is fully compatible with the old one. Simply upgrade to the new version for an immediate performance boost! 🚀🚀🚀

## Performance

#### Test & benchmark MLA decoding (Sparse & Dense):

```bash
python tests/test_flash_mla_dense_decoding.py
python tests/test_flash_mla_sparse_decoding.py
```

The dense MLA decoding kernel achieves up to 3000 GB/s in memory-bound configuration and 660 TFLOPS in computation-bound configuration on H800 SXM5 with CUDA 12.8. The token-level sparse MLA decoding kernel (which uses an FP8 KV cache while performing the matrix multiplication in bfloat16) achieves 410 TFLOPS in compute-bound configuration on H800 SXM5 with CUDA 12.8, and achieves up to 700 TFlops on B200.

#### Test & benchmark MHA prefill (Dense):

```bash
python tests/test_fmha_sm100.py
```

It achieves up to 1460 TFlops in forward and 1000 TFlops in backward computation on B200, as reported by NVIDIA.

#### Test & benchmark MLA prefill (Sparse):

```bash
python tests/test_flash_mla_sparse_prefill.py
```

It achieves up to 640 TFlops in forward computation on H800 SXM5 with CUDA 12.8, and achieves up to 1450 TFlops on B200, CUDA 12.9.

#### Test & benchmark the fused norm RoPE attn RoPE cast kernel (Sparse):

```bash
python tests/test_fused_norm_rope_attn_rope_cast.py
```

[TileLang](https://github.com/tile-ai/tilelang), [Tile-Kernels](https://github.com/deepseek-ai/TileKernels/), and [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) are required for running this test script.

This kernel fuses Q-norm (only used in V4, not V4.1), Q-RoPE, core attention, O-RoPE (conjugate) and cast-to-FP8 into a single kernel, saving times for those small kernels. Although it fuses many small operations, this kernel still keeps the same or even slightly higher TFlops (at the cost of having to permute the Q_b and Wv weights in advance). It achieves up to 1430 TFlops during prefill and 670 TFlops during decoding on B200.


## Requirements

- SM90 / SM100 (See the support matrix below)
- CUDA 12.8 and above (CUDA 12.9+ is required for SM100 kernels)
- PyTorch 2.0 and above

Support matrix:

| Kernel | GPU Architecture | MLA Mode [1] | Supported Models |
| :---: | :---: | :---: | :---: |
| Dense Decoding | SM90 | MQA | DeepSeek V3 / V3.1 |
| Sparse Decoding | SM90 & SM100 | MQA | DeepSeek V3.2 / V4 / V4.1 [2] |
| Dense Prefill | SM100 | MHA | DeepSeek V3 / V3.1 / V3.2 |
| Sparse Prefill | SM90 & SM100 | MQA | DeepSeek V3.2 / V4 / V4.1 |
| Fused Norm RoPE Attn RoPE Cast | SM100 | MQA | DeepSeek V4 / V4.1 |

[1]: Here "MLA Mode" refers to the mode used for MLA calculation. MQA stands for Multi-Query Attention mode (i.e. `head_dim_k` = 576 (for DeepSeek V3/V3.1/V3.2) or 512 (for DeepSeek V4/V4.1) with `head_dim_v` = 512), while MHA stands for Multi-Head Attention mode (i.e. `head_dim_k` = 192 / 128 with `head_dim_v` = 128). For a detailed explanation of these modes, please refer to the appendix of [DeepSeek V3.2's Paper](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).

[2] Sparse Decoding for DeepSeek V4.1 is only available on SM100

## Installation

```bash
git clone https://github.com/deepseek-ai/FlashMLA.git flash-mla
cd flash-mla
git submodule update --init --recursive
pip install -v .
```

## Usage

### MLA Decoding

To use the MLA decoding kernels, call get_mla_metadata once before the decoding loop to get the tile scheduler metadata. Then, call flash_mla_with_kvcache in each decoding step. For example:

```python
from flash_mla import get_mla_metadata, flash_mla_with_kvcache

tile_scheduler_metadata, num_splits = get_mla_metadata(
    cache_seqlens,
    s_q * h_q // h_kv,
    h_kv,
    h_q,
    is_fp8,
    topk,
)

for i in range(num_layers):
    ...
    o_i, lse_i = flash_mla_with_kvcache(
        q_i, kvcache_i, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits,
        is_causal, is_fp8_kvcache, indices,
    )
    ...
```

Where

- `s_q` is the number of q tokens per q sequence. If MTP (speculative decoding) is disabled, it should be 1.
- `h_kv` is the number of key-value heads.
- `h_q` is the number of query heads.

**FP8 KV Cache:**
If `is_fp8_kvcache` is set to `True`, the kernel reads the KV cache in the "FP8 with scale" format (described below). It dequantizes the cache to bfloat16 and performs attention computation in bfloat16. The output is also in bfloat16. In this repository, `is_fp8_kvcache=True` is only supported together with `indices` (i.e. sparse attention); the dense decoding kernel reads bf16 / fp16 KV caches. 

In the "FP8 with scale" format for DeepSeek V3.2 (`head_dim` = 576, sparse attention), a page block is `page_block_size` token-major rows of 656 Bytes each:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512 `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values. The first `float32` is the scale for the first 128 `float8_e4m3` values, the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This part is not quantized for accuracy.

For DeepSeek V4 / V4.1 (`head_dim` = 512), the format is detected from the last dimension of `k_cache` (i.e. the bytes per token): 584 (V4), 528 (V4.1) or 288 (V4.1 fp4). In all three, a page block stores `page_block_size` data rows first and `page_block_size` scale rows afterwards:
-   **V4**: 584 Bytes per token. The data row is 448 Bytes of quantized NoPE (`float8_e4m3`) followed by 128 Bytes, i.e. the 64 `bfloat16` RoPE values (not quantized). The scale row is 8 Bytes, of which the first 7 are `float8_e8m0` scales and the 8th byte is padding; each scale covers 64 consecutive `float8_e4m3` values of the NoPE part.
-   **V4.1**: 528 Bytes per token. The data row is 512 Bytes of `float8_e4m3`, i.e. the 64 RoPE dimensions are quantized as well and there is no `bfloat16` part. The scale row is 16 Bytes of `float8_e8m0`, each scale covering 32 consecutive `float8_e4m3` values.
-   **V4.1 fp4**: 288 Bytes per token. The data row is 256 Bytes containing 512 `e2m1` values, 2 values per byte (the even-indexed one in the low nibble). The scale row is 32 Bytes of `float8_e4m3`, each scale covering 16 consecutive `e2m1` values. This format is only valid for `extra_k_cache`, and only when `k_cache` is in the V4.1 format; otherwise `extra_k_cache` must have the same format as `k_cache`. In pratice we expect the sliding window (SWA) kv cache to be in FP8 and the compress attention (CA) kv cache to be in FP4.

See `tests/quant.py` for quantization and dequantization details.

**Sparse Attention (`indices` tensor):**
The `indices` tensor (if provided) enables token-level sparse attention by instructing the kernel to compute attention only for specified tokens.

-   **Shape:** `indices` should be a 3D tensor of shape `(batch_size, seq_len_q, topk)`.
-   **Format:** `indices_in_kvcache[i][j][k] = (the index of the page block where token t resides) * page_block_size + (the offset of token t within the page block)`, where `t` is the k-th token for the j-th query sequence in the i-th batch. Since the index of the page block has already been encoded into `indices_in_kvcache`, the kernel does not require the `block_table` parameter.
-   **Invalid entries:** Set invalid indices to `-1`.

**Return Values:**
The kernel returns `(out, lse)`, where:
-   `out` is the attention result.
-   `lse` is the log-sum-exp value of the attention scores for each query head.

See `tests/test_flash_mla_dense_decoding.py` and `tests/test_flash_mla_sparse_decoding.py` for complete examples.

### Sparse MLA Prefill

For the sparse MLA prefill kernel, call `flash_mla_sparse_fwd` directly with the following parameters:
-   `q`: Query tensor of shape `[s_q, h_q, d_qk]`
-   `kv`: Key-Value tensor of shape `[s_kv, h_kv, d_qk]`
-   `indices`: Indices tensor of shape `[s_q, h_kv, topk]`
-   `sm_scale`: A scalar value

**Note on batching:** This kernel does not support a batch dimension. For multi-batch inference, reshape the input tensors and adjust the `indices` parameter to simulate batch processing.

**Invalid indices:** Set invalid entries in `indices` to `-1` or any number `>= s_kv`.

**Return Values and Equivalent PyTorch Code:**
The kernel returns `(out, max_logits, lse)`. This is equivalent to the following PyTorch operations:

```python
Q: [s_q, h_q, d_qk], bfloat16
kv: [s_kv, h_kv, d_qk], bfloat16
indices: [s_q, h_kv, topk], int32

kv = kv.squeeze(1)  # [s_kv, d_qk], h_kv must be 1
indices = indices.squeeze(1)    # [s_q, topk]
focused_kv = kv[indices]    # For the i-th sequence (s_q), the corresponding KV tokens are selected from the KV cache based on indices[i, :]. This operation results in a tensor of shape [s_q, topk, d_qk].

P = (Q @ focused_kv.transpose(-1, -2)) * sm_scale * math.log2(math.e)    # [s_q, h_q, topk]
max_logits = P.max(dim=-1) # [s_q, h_q]
lse = log2sumexp2(P, dim=-1, base=2)   # [s_q, h_q]，"log2sumexp2" means that the exponentiation and logarithm are base-2
S = exp2(P - lse)      # [s_q, h_q, topk]
out = S @ focused_kv  # [s_q, h_q, d_qk]

return (out, max_logits, lse)
```

See `tests/test_flash_mla_sparse_prefill.py` for a complete example.

### Dense MHA Prefill

This kernel implements the standard dense Multi-Head Attention (MHA) forward and backward operations. It can be called using:
-   `flash_attn_varlen_func`
-   `flash_attn_varlen_qkvpacked_func`
-   `flash_attn_varlen_kvpacked_func`

The usage is similar to the `flash_attn` package. See `tests/test_fmha_sm100.py` for a complete example.

### Fused norm + RoPE + attn + RoPE + cast kernel

In the DeepSeek-V4.1 release, we also provide a fused kernel that combines Q-norm (only used in V4, not in V4.1), Q-RoPE, core attention, O-RoPE (conjugate) and the cast to FP8 into a single kernel. It removes the extra time spent on these small kernels while keeping the same or even slightly higher TFlops, at the cost of having to permute the Q_b and Wv weights in advance.

In DeepSeek-V4.1 attention, Q (`[hidden_size]`) is first projected to `[q_lora_rank]` (the Q_a projection) and then to `[num_attention_heads, head_dim]` (the Q_b projection). After core attention, the output (`[num_attention_heads, head_dim]`) is reshaped to `[o_groups, num_attention_heads // o_groups * head_dim]`, and each of its rows is projected to `[o_lora_rank]` (the Wv projection), giving an `[o_groups, o_lora_rank]` matrix. That matrix is reshaped to `[o_groups * o_lora_rank]` and finally projected to `[hidden_size]` (the Wo projection). This kernel requires the Q_b and Wv weights to be permuted.

To permute the Q_b weight:

```python
import torch
import tile_kernels
from flash_mla import fused_norm_rope_attn_rope_cast

h_q, d_q = 64, 512          # Q heads and Q head dimension
q_lora_rank = 1536
scale_gran = 128

# q_b_proj: [h_q * d_q, q_lora_rank], bfloat16
q_b_proj = torch.randn((h_q * d_q, q_lora_rank), dtype=torch.bfloat16, device='cuda')

# Quantize the weight to FP8 with per-token scale factors, in DeepGEMM's layout
q_b_proj_fp8, q_b_sf = tile_kernels.quant.per_token_cast(
    q_b_proj, 'e4m3', scale_gran,
    use_tma_aligned_col_major_sf=True, round_sf=True, use_packed_ue8m0=True,
)

# Permute the weight and its scale factors into the layout required by the fused kernel
q_b_proj_fp8, q_b_sf = fused_norm_rope_attn_rope_cast.permute_q_b_proj(
    (q_b_proj_fp8, q_b_sf), h_q, d_q,
)
```

To permute the Wv weight:

```python
import deep_gemm
import torch
import tile_kernels
from flash_mla import fused_norm_rope_attn_rope_cast

n_wv_group, wv_group_size, d_o = 8, 8, 512      # n_wv_group * wv_group_size == h_q
wv_proj_out_dim = 512                           # o_lora_rank
scale_gran = 32

# wv_proj: [n_wv_group, wv_proj_out_dim, wv_group_size * d_o], bfloat16
wv_proj = torch.randn((n_wv_group * wv_proj_out_dim, wv_group_size * d_o),
                      dtype=torch.bfloat16, device='cuda')

# Quantize the weight to FP8, and put its scale factors into the layout that DeepGEMM's einsum expects
wv_proj_fp8, wv_sf = tile_kernels.quant.per_token_cast(
    wv_proj, 'e4m3', scale_gran,
    use_tma_aligned_col_major_sf=False, round_sf=True, use_packed_ue8m0=False,
)
wv_sf = deep_gemm.transform_sf_into_required_layout(
    wv_sf.view(n_wv_group, wv_proj_out_dim, wv_group_size * d_o // scale_gran),
    wv_proj_out_dim, wv_group_size * d_o,
    num_groups=n_wv_group, recipe=(1, 1, scale_gran), is_sfa=False,
)
wv_proj_fp8 = wv_proj_fp8.view(n_wv_group, wv_proj_out_dim, wv_group_size * d_o)

# Permute the weight and its scale factors into the layout required by the fused kernel
wv_proj_fp8, wv_sf = fused_norm_rope_attn_rope_cast.permute_wv_proj(
    (wv_proj_fp8, wv_sf), wv_group_size, d_o,
)
```

And finally, to use the fused kernel:

```python
# q: [s_q, h_q, d_qk], bfloat16, i.e. the Q_b projection computed with the permuted weight above
out_fp8, out_sf, max_logits, lse = fused_norm_rope_attn_rope_cast.prefill(
    enable_q_norm,              # False for DeepSeek-V4.1
    rms_norm_eps,               # e.g. 1e-4
    token_positions,            # [s_q], int32
    False, 64, cos_sin_cache,   # non-neox RoPE with rope_dim = 64
    n_wv_group,                 # h_q // wv_group_size
    32,                         # num_per_channels
    True, True, True,           # use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0
    q, kv, indices,             # bf16 Q, bf16 KV [s_kv, h_kv, d_qk], int32 indices [s_q, h_kv, topk]
    sm_scale=sm_scale,
    attn_sink=attn_sink,        # optional, [h_q], float32
    topk_length=topk_length,    # optional, [s_q], int32
)

# For decoding, call `decode` instead, passing the paged quantized KV cache:
#   q: [s_q, h_q, d_qk], bf16
#   k_cache: [num_blocks, page_block_size, h_kv, bytes_per_token], fp8_e4m3
#   indices_in_kvcache: [s_q, topk], int32
out_fp8, out_sf, lse = fused_norm_rope_attn_rope_cast.decode(
    enable_q_norm, rms_norm_eps,
    token_positions, False, 64, cos_sin_cache,
    n_wv_group, 32, True, True, True,
    q, k_cache, indices_in_kvcache,
    sm_scale=sm_scale,
    attn_sink=attn_sink,
    topk_length=topk_length,
    extra_k_cache=extra_k_cache,                        # optional, same layout as k_cache
    extra_indices_in_kvcache=extra_indices_in_kvcache,  # optional, [s_q, extra_topk], int32
    extra_topk_length=extra_topk_length,                # optional, [s_q], int32
)

# The FP8 output is consumed directly by the Wv projection, using the permuted Wv weight
wv_out = torch.empty((s_q, n_wv_group, wv_proj_out_dim), dtype=torch.bfloat16, device='cuda')
deep_gemm.fp8_einsum("bhr,hdr->bhd", (out_fp8, out_sf), (wv_proj_fp8, wv_sf), wv_out, recipe=(1, 1, 32))
```

You may refer to the fused kernel's test script ([tests/test_fused_norm_rope_attn_rope_cast.py](tests/test_fused_norm_rope_attn_rope_cast.py)) for a complete example.

## Acknowledgement

FlashMLA is inspired by [FlashAttention 2&3](https://github.com/dao-AILab/flash-attention/) and [cutlass](https://github.com/nvidia/cutlass) projects.

## Community Support

### MetaX
For MetaX GPUs, visit the official website: [MetaX](https://www.metax-tech.com).

The corresponding FlashMLA version can be found at: [MetaX-MACA/FlashMLA](https://github.com/MetaX-MACA/FlashMLA)


### Moore Threads
For the Moore Threads GPU, visit the official website: [Moore Threads](https://www.mthreads.com/).

The corresponding FlashMLA version is available on GitHub: [MooreThreads/MT-flashMLA](https://github.com/MooreThreads/MT-flashMLA).


### Hygon DCU
For the Hygon DCU, visit the official website: [Hygon Developer](https://developer.sourcefind.cn/).

The corresponding FlashMLA version is available here: [OpenDAS/MLAttention](https://developer.sourcefind.cn/codes/OpenDAS/MLAttention).


### Intellifusion
For the Intellifusion NNP, visit the official website: [Intellifusion](https://www.intellif.com).

The corresponding FlashMLA version is available on Gitee: [Intellifusion/tyllm](https://gitee.com/Intellifusion_2025/tyllm/blob/master/python/tylang/flash_mla.py).


### Iluvatar Corex
For Iluvatar Corex GPUs, visit the official website: [Iluvatar Corex](https://www.iluvatar.com).

The corresponding FlashMLA version is available on GitHub: [Deep-Spark/FlashMLA](https://github.com/Deep-Spark/FlashMLA/tree/iluvatar_flashmla)


### AMD Instinct
For AMD Instinct GPUs, visit the official website: [AMD Instinct](https://www.amd.com/en/products/accelerators/instinct.html).

The corresponding FlashMLA version can be found at: [AITER/MLA](https://github.com/ROCm/aiter/blob/main/aiter/mla.py)

## Citation

```bibtex
@misc{flashmla2025,
      title={FlashMLA: Efficient Multi-head Latent Attention Kernels},
      author={Jiashi Li, Shengyu Liu},
      year={2025},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/FlashMLA}},
}
```
