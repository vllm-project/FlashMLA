import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

EXT_NAME_CUDA = "flash_mla._flashmla_C"

CUDA_API_SOURCES = [
    "csrc/api/api.cpp",
    "csrc/api/sparse_prefill.cpp",
    "csrc/api/sparse_decode.cpp",
    "csrc/api/dense_decode.cpp",
    "csrc/api/fused_norm_rope_attn_rope_cast_fwd.cpp",
]

# Heads the fused core_attn kernel is instantiated for: 128 (2-CTA cluster) and 64
HEADS_FOR_FUSED_CORE_ATTN = (64, 128)

# The current kernels: DeepSeek V4.1, whose KV cache stores each token's scales right after
# that token's raw data.
CUDA_KERNEL_SOURCES = [
    # Architecture-independent kernels for decoding
    "csrc/cuda_kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
    "csrc/cuda_kernels/smxx/decode/combine/combine.cu",

    # sm100 dense prefill (cutlass FMHA): forward and backward
    "csrc/cuda_kernels/sm100/prefill/dense/fmha_cutlass_fwd_sm100.cu",
    "csrc/cuda_kernels/sm100/prefill/dense/fmha_cutlass_bwd_sm100.cu",

    # sm100 sparse prefill
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd/head64/instantiations/phase1_h64_k512.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k512.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_k512.cu",

    # sm100 sparse decode, head64 and (through the small-topk prefill kernel) head128
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41_h64.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41_h64_no_split.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64.cu",
    "csrc/cuda_kernels/sm100/decode/sparse/head64/instantiations/v41fp4_h64_no_split.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41_splitkv.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_v41fp4_splitkv.cu",

    # sm100 fused norm + rope + attn + rope + cast: 6 instantiations per head count
    # (V4.1 fp8 prefill, V4.1 fp8 decode and V4.1 fp4 decode, each with and without q norm)
    *[f"csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/{model}_h{h}_{mode}_{norm}.cu"
      for h in HEADS_FOR_FUSED_CORE_ATTN
      for model, mode in (("v41", "prefill"), ("v41", "decode"), ("v41fp4", "decode"))
      for norm in ("norm", "nonorm")],
    "csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_q_b_proj/kernel.cu",
    "csrc/cuda_kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/permute_wv_proj/kernel.cu",
]

# The kernels upstream dropped in the 2026.09.30 release and we keep: DeepSeek V3.2 / V4 (whose
# page blocks hold the scales in a region of their own) and everything on Hopper.
LEGACY_KERNEL_SOURCES = [
    # Decoding helpers for dense MLA, which has no counterpart among the current kernels
    "csrc/cuda_kernels/legacy/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu",
    "csrc/cuda_kernels/legacy/smxx/decode/combine/combine.cu",

    # sm90 dense decode (DeepSeek V3 / V3.1, GLM5Next)
    "csrc/cuda_kernels/legacy/sm90/decode/dense/instantiations/fp16.cu",
    "csrc/cuda_kernels/legacy/sm90/decode/dense/instantiations/bf16.cu",

    # sm90 sparse decode
    "csrc/cuda_kernels/legacy/sm90/decode/sparse/instantiations/v4_persistent_h64.cu",
    "csrc/cuda_kernels/legacy/sm90/decode/sparse/instantiations/v4_persistent_h128.cu",
    "csrc/cuda_kernels/legacy/sm90/decode/sparse/instantiations/v32_persistent_h64.cu",
    "csrc/cuda_kernels/legacy/sm90/decode/sparse/instantiations/v32_persistent_h128.cu",

    # sm90 sparse prefill
    "csrc/cuda_kernels/legacy/sm90/prefill/sparse/instantiations/phase1_k512.cu",
    "csrc/cuda_kernels/legacy/sm90/prefill/sparse/instantiations/phase1_k512_topklen.cu",
    "csrc/cuda_kernels/legacy/sm90/prefill/sparse/instantiations/phase1_k576.cu",
    "csrc/cuda_kernels/legacy/sm90/prefill/sparse/instantiations/phase1_k576_topklen.cu",

    # sm100 sparse prefill, d_qk = 576 only: the current kernels cover d_qk = 512 for V4 too,
    # since the KV of a prefill is bf16 and the cache format does not reach these kernels
    "csrc/cuda_kernels/legacy/sm100/prefill/sparse/fwd/head64/instantiations/phase1_h64_k576.cu",
    "csrc/cuda_kernels/legacy/sm100/prefill/sparse/fwd/head128/instantiations/phase1_k576.cu",

    # sm100 sparse decode
    "csrc/cuda_kernels/legacy/sm100/decode/sparse/head64/instantiations/v32_h64.cu",
    "csrc/cuda_kernels/legacy/sm100/decode/sparse/head64/instantiations/v32_h64_no_split.cu",
    "csrc/cuda_kernels/legacy/sm100/decode/sparse/head64/instantiations/v4_h64.cu",
    "csrc/cuda_kernels/legacy/sm100/decode/sparse/head64/instantiations/v4_h64_no_split.cu",
    "csrc/cuda_kernels/legacy/sm100/decode/sparse/nvfp4_head64/instantiations/v32_nvfp4_fp8rope.cu",
    "csrc/cuda_kernels/legacy/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512.cu",
    "csrc/cuda_kernels/legacy/sm100/prefill/sparse/fwd_for_small_topk/head128/instantiations/phase1_decode_k512_splitkv.cu",

    # sm100 fused norm + rope + attn + rope + cast, V4 decode (V4 prefill is identical to V4.1's)
    *[f"csrc/cuda_kernels/legacy/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/instantiations/v4_h{h}_decode_{norm}.cu"
      for h in HEADS_FOR_FUSED_CORE_ATTN
      for norm in ("norm", "nonorm")],
]


def is_flag_set(flag: str) -> bool:
    return os.getenv(flag, "FALSE").lower() in ["true", "1", "y", "yes"]


def get_features_args():
    # ABI-stable flags (always on): TORCH_TARGET_VERSION pins the minimum runtime
    # PyTorch version and bans unstable ATen/c10/torch headers at compile time;
    # USE_CUDA exposes aoti_torch_get_current_cuda_stream from the shim.
    # FLASH_MLA_ENABLE_DENSE_BWD registers dense_prefill_bwd (for when we build
    # the extension directly in the fork); vLLM's integrated build will omit it as
    # it is inference-only.
    features_args = [
        "-DTORCH_TARGET_VERSION=0x020a000000000000",  # PyTorch >= 2.10 at runtime
        "-DUSE_CUDA",
        "-DFLASH_MLA_ENABLE_DENSE_BWD",
        # Selects the CUDA branch of csrc/params.h
        "-DFLASH_MLA_IS_BUILD_ON_CUDA",
    ]
    if is_flag_set("FLASH_MLA_DISABLE_FP16"):
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    return features_args


def get_arch_flags(cuda_home):
    # Check NVCC Version
    # NOTE The "CUDA_HOME" here is not necessarily from the `CUDA_HOME` environment variable. For more details, see `torch/utils/cpp_extension.py`
    nvcc_version = subprocess.check_output(
        [os.path.join(cuda_home, "bin", "nvcc"), '--version'], stderr=subprocess.STDOUT
    ).decode('utf-8')
    nvcc_version_number = nvcc_version.split('release ')[1].split(',')[0].strip()
    major, minor = map(int, nvcc_version_number.split('.'))
    print(f'Compiling using NVCC {major}.{minor}')

    DISABLE_SM100 = is_flag_set("FLASH_MLA_DISABLE_SM100")
    DISABLE_SM90 = is_flag_set("FLASH_MLA_DISABLE_SM90")
    if major < 12 or (major == 12 and minor <= 8):
        assert DISABLE_SM100, "sm100 compilation for Flash MLA requires NVCC 12.9 or higher. Please set FLASH_MLA_DISABLE_SM100=1 to disable sm100 compilation, or update your environment."    # TODO Implement this

    arch_flags = []
    if not DISABLE_SM100:
        # We use architecture-specific (sm_100a / sm_103a) targets instead of the family-specific one (sm_100f) for better SASS code generation
        arch_flags.extend(["-gencode", "arch=compute_100a,code=sm_100a"])
        arch_flags.extend(["-gencode", "arch=compute_103a,code=sm_103a"])
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    return arch_flags


def get_nvcc_thread_args():
    nvcc_threads = os.getenv("NVCC_THREADS") or "32"
    return ["--threads", nvcc_threads]


def build_on_cuda_platform():
    from torch.utils.cpp_extension import (
        BuildExtension,
        CUDAExtension,
        IS_WINDOWS,
        CUDA_HOME,
    )

    assert CUDA_HOME is not None, "PyTorch must be compiled with CUDA support"
    subprocess.run(["git", "submodule", "update", "--init", "csrc/3rdparty/cutlass"])

    if IS_WINDOWS:
        cxx_args = ["/O2", "/std:c++20", "/DNDEBUG", "/W0"]
    else:
        cxx_args = ["-O3", "-std=c++20", "-DNDEBUG", "-Wno-deprecated-declarations"]

    ext_modules = [
        CUDAExtension(
            name=EXT_NAME_CUDA,
            sources=[
                *CUDA_API_SOURCES,
                *CUDA_KERNEL_SOURCES,
                *LEGACY_KERNEL_SOURCES,
            ],
            extra_compile_args={
                # kerutils sets KERUTILS_IS_BUILD_ON_CUDA itself when __CUDACC__ is defined, so
                # only the host translation units have to be told which platform they are on.
                "cxx": cxx_args + get_features_args() + ["-DKERUTILS_IS_BUILD_ON_CUDA"],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "-DNDEBUG",
                    "-D_USE_MATH_DEFINES",
                    "-Wno-deprecated-declarations",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills,--warn-on-local-memory-usage,--warn-on-double-precision-use",
                    "-lineinfo",
                    "--source-in-ptx",
                    "--diag-suppress=445",  # "constant "N" is not used in declaring the parameter types of function template"
                ] + get_features_args() + get_arch_flags(CUDA_HOME) + get_nvcc_thread_args(),
            },
            include_dirs=[
                THIS_DIR / "csrc",
                THIS_DIR / "csrc" / "3rdparty" / "kerutils" / "include",
                THIS_DIR / "csrc" / "3rdparty" / "cutlass" / "include",
                THIS_DIR / "csrc" / "3rdparty" / "cutlass" / "tools" / "util" / "include",
                Path(CUDA_HOME) / "targets" / "x86_64-linux" / "include" / "cccl",   # for cuda/std headers in CUDA 13+
                Path(CUDA_HOME) / "targets" / "sbsa-linux" / "include" / "cccl",
            ],
            # The kernels call the CUDA driver API directly (tensor map creation, and
            # cuDriverGetVersion in kerutils). On a machine without the driver installed, e.g. a
            # CPU-only build node, `-lcuda` resolves against the stubs shipped with the toolkit.
            extra_link_args=[
                f"-L{Path(CUDA_HOME) / 'targets' / 'x86_64-linux' / 'lib' / 'stubs'}",
                f"-L{Path(CUDA_HOME) / 'targets' / 'sbsa-linux' / 'lib' / 'stubs'}",
                "-lcuda",
            ],
            # Build against CPython's Limited API (abi3) so one wheel works across
            # multiple CPython versions, which is possible now that pybind11 is gone
            py_limited_api=True,
        )
    ]

    return ext_modules, BuildExtension


def get_version() -> str:
    try:
        cmd = ['git', 'rev-parse', '--short', 'HEAD']
        rev = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
    except Exception as _:
        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        rev = '+' + date_time_str
    return "1.0.0" + rev


def main():
    ext_modules, build_ext = build_on_cuda_platform()

    setup(
        name="flash_mla",
        version=get_version(),
        packages=find_packages(include=['flash_mla']),
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        options={"bdist_wheel": {"py_limited_api": "cp310"}},
    )


if __name__ == "__main__":
    main()
