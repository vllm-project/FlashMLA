import time
import dataclasses
from typing import Tuple, List, Dict, Optional
import copy

import pytest

import rich.console
import rich.table

import torch
import kernelkit as kk

import flash_mla

import lib
import quant
from lib import TestParam
from lib import RawTestParamForDecode as RawTestParam
import ref

"""
Generate testcase for unit test
"""

def gen_testcase() -> List[RawTestParam]:
    # The DeepSeek-V4.1 KV cache formats (V41 / V41_FP4) are only supported on SM100f
    supports_v41 = torch.cuda.get_device_capability()[0] >= 10
    correctness_cases = []
    corner_cases = []
    for d_qk in [576, 512]:
        for have_extra_k in ([False, True] if d_qk == 512 else [False]):
            for have_extra_topk_len in ([False, True] if have_extra_k else [False]):
                for have_topk_len in ([False, True] if d_qk == 512 else [False]):
                    for h_q in [64, 128]:
                        cur_correctness_cases = [
                            RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                        have_topk_length=have_topk_len,
                                        enable_attn_sink=True,
                                        extra_s_k=extra_s_k,
                                        extra_topk=extra_topk,
                                        block_size=block_size,
                                        extra_block_size=extra_block_size,
                                        have_extra_topk_length=have_extra_topk_len,
                                        d_qk=d_qk,
                                        check_correctness=True,
                                        num_runs=0)
                            for (s_k, topk, block_size) in [
                                (512, 64, 2),
                                (512, 64, 64),
                                (512, 64, 69),
                                (1024, 576, 2),
                                (1024, 576, 61),
                                (2046, 2048, 2),
                                (2046, 2048, 64),
                                (2046, 2048, 576)
                            ]
                            for (extra_s_k, extra_topk, extra_block_size) in ([
                                (512, 64, 2),
                                (512, 64, 64),
                                (512, 64, 69),
                                (1024, 576, 2),
                                (1024, 576, 61),
                                (2046, 2048, 2),
                                (2046, 2048, 64),
                                (2046, 2048, 576)
                            ] if have_extra_k else [(None, None, None)])
                            for b in [4, 74, 321]
                            for s_q in [1, 3]
                            for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                        ]
                        correctness_cases.extend(cur_correctness_cases)

                        cur_corner_cases = [
                            RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                        is_all_indices_invalid=is_all_indices_invalid,
                                        have_zero_seqlen_k=have_zero_seqlen_k,
                                        have_topk_length=have_topk_len,
                                        enable_attn_sink=enable_attn_sink,
                                        extra_s_k=extra_s_k,
                                        extra_topk=extra_topk,
                                        block_size=block_size,
                                        extra_block_size=extra_block_size,
                                        have_extra_topk_length=have_extra_topk_len,
                                        d_qk=d_qk,
                                        check_correctness=True,
                                        num_runs=0,
                            )
                            for (s_k, topk, block_size) in [
                                (512, 64, 61),
                                (650, 576, 53),
                            ]
                            for (extra_s_k, extra_topk, extra_block_size) in ([
                                (512, 64, 61),
                                (650, 576, 53),
                            ] if have_extra_k else [(None, None, None)])
                            for b in [4, 74, 321]
                            for s_q in [3]
                            for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                            for is_all_indices_invalid in [True, False]
                            for have_zero_seqlen_k in [True, False]
                            for enable_attn_sink in [True, False]
                            if (is_all_indices_invalid or have_zero_seqlen_k or enable_attn_sink)
                        ]
                        corner_cases.extend(cur_corner_cases)

    # DeepSeek-V4.1: fp8 (V41) KV cache, optionally with an fp4 (V41_FP4) extra KV cache
    if supports_v41:
        for extra_fp4 in [False, True]:
            correctness_cases.extend([
                RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                             have_topk_length=have_topk_len,
                             enable_attn_sink=True,
                             extra_s_k=extra_s_k,
                             extra_topk=extra_topk,
                             block_size=block_size,
                             extra_block_size=extra_block_size,
                             have_extra_topk_length=have_extra_topk_len,
                             d_qk=512,
                             kvcache_layout=quant.KVCacheLayout.V41_FP8Sparse,
                             extra_kvcache_layout=quant.KVCacheLayout.V41_FP4 if extra_fp4 else None,
                             check_correctness=True,
                             num_runs=0)
                for h_q in [64, 128]
                for have_extra_topk_len in [False, True]
                for have_topk_len in [False]
                for (s_k, topk, block_size) in [(512, 64, 64), (1024, 576, 61)]
                for (extra_s_k, extra_topk, extra_block_size) in [(512, 64, 64), (650, 576, 53)]
                for b in [4]
                for s_q in [1, 3]
                for is_varlen in [True]
            ])

        # V3.2 geometry with a 352-byte NVFP4-NoPE/fp8-RoPE KV record.
        for h_q in [64, 128]:
            for have_topk_len in [False, True]:
                correctness_cases.extend([
                    RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                 have_topk_length=have_topk_len,
                                 enable_attn_sink=True,
                                 block_size=block_size,
                                 d_qk=576,
                                 kvcache_layout=quant.KVCacheLayout.V32_NVFP4_FP8ROPE,
                                 check_correctness=True,
                                 num_runs=0)
                    for (s_k, topk, block_size) in [
                        (512, 64, 2),
                        (512, 64, 64),
                        (512, 64, 69),
                        (1024, 576, 61),
                        (2046, 2048, 64),
                    ]
                    for b in [4, 74]
                    for s_q in [1, 3]
                    for is_varlen in ([True, False] if (b == 74 and not have_topk_len) else [True])
                ])
            corner_cases.extend([
                RawTestParam(b, h_q, 3, 1, s_k, True, topk,
                             is_all_indices_invalid=is_all_indices_invalid,
                             have_zero_seqlen_k=have_zero_seqlen_k,
                             enable_attn_sink=enable_attn_sink,
                             block_size=block_size,
                             d_qk=576,
                             kvcache_layout=quant.KVCacheLayout.V32_NVFP4_FP8ROPE,
                             check_correctness=True,
                             num_runs=0)
                for (s_k, topk, block_size) in [(512, 64, 61), (650, 576, 53)]
                for b in [4, 74]
                for is_all_indices_invalid in [True, False]
                for have_zero_seqlen_k in [True, False]
                for enable_attn_sink in [True, False]
                if (is_all_indices_invalid or have_zero_seqlen_k or enable_attn_sink)
            ])

    base_and_bszs = [
        # V3.2
        (RawTestParam(0, 128, 2, 1, 32768, True, topk=2048, d_qk=576), [2, 64, 74, 128]),
        # V3.2 geometry with NVFP4 KV cache
        (RawTestParam(0, 128, 2, 1, 32768, True, topk=2048, d_qk=576,
                      kvcache_layout=quant.KVCacheLayout.V32_NVFP4_FP8ROPE), [64, 128]),
        # DeepSeek-V4 CONFIG1
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=512, block_size=256, extra_block_size=64), [2, 64, 74, 128, 74*2, 256]),
        # DeepSeek-V4 CONFIG2
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=64), [2, 64, 74, 128, 74*2, 256]),
        # DeepSeek-V4 CONFIG3
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=2, have_extra_topk_length=True), [2, 64, 74, 128, 74*2, 256]),
        # DeepSeek-V4 CONFIG4
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=2, have_extra_topk_length=True), [2, 64, 74, 128, 74*2, 256]),
    ]
    if supports_v41:
        base_and_bszs += [
            # DeepSeek-V4.1 CONFIG1 (fp8 V41 KV cache + fp4 V41_FP4 extra KV cache)
            (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=512, block_size=256, extra_block_size=64,
                          kvcache_layout=quant.KVCacheLayout.V41_FP8Sparse, extra_kvcache_layout=quant.KVCacheLayout.V41_FP4), [2, 64, 74, 128, 74*2, 256]),
            (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=512, block_size=256, extra_block_size=64,
                          kvcache_layout=quant.KVCacheLayout.V41_FP8Sparse, extra_kvcache_layout=quant.KVCacheLayout.V41_FP4), [2, 64, 74, 128, 74*2, 256])
        ]
    performance_cases = [
        # Production cases
        dataclasses.replace(base, b=b)
        for base, bszs in base_and_bszs
        for b in bszs
    ] + [
        # Peak perf cases
        RawTestParam(74*2, h_q, 2, 1, 32768, True, topk=16384, d_qk=d_qk)
        for h_q in [64, 128]
        for d_qk in [512, 576]
    ] + [
        # Peak perf cases, NVFP4 KV cache
        RawTestParam(74*2, h_q, 2, 1, 32768, True, topk=16384, d_qk=576,
                     kvcache_layout=quant.KVCacheLayout.V32_NVFP4_FP8ROPE)
        for h_q in [64, 128]
    ]

    return correctness_cases + corner_cases + performance_cases


@dataclasses.dataclass
class Result:
    is_correct: bool
    compute_memory_ratio: float
    time_usage_per_us: float
    splitkv_time_usage_us: float
    combine_time_usage_us: float
    achieved_tflops: float
    achieved_gBps: float

_counter = kk.Counter()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90 small-head decode",
)
@pytest.mark.parametrize("heads", [8, 16, 32])
@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize(
    "case",
    [
        "plain",
        "length",
        "extra",
        "empty",
        "unsplit",
        "invariant",
        "unaligned",
        "unaligned_output",
        "persistent",
        "persistent_extra",
        "persistent_empty",
        "persistent_invalid",
        "persistent_length",
        "persistent_multiquery",
        "persistent_unaligned",
        "persistent_split",
    ],
)
@torch.inference_mode()
def test_small_heads_decode_preserves_sparse_attention(heads, dim, case):
    """Native heads preserve FP8 attention, sink, empty-row and strided-output semantics."""
    with torch.device("cuda"):
        # generate_testcase_for_decode uses the caller's default floating dtype.
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            persistent = case.startswith("persistent")
            extra = case in ("extra", "persistent_extra", "persistent_split")
            lengths = case in ("length", "extra", "invariant", "persistent_length")
            p = RawTestParam(
                b=(
                    4
                    if case == "persistent_split"
                    else (16 if case == "persistent_multiquery" else 128)
                )
                if persistent
                else (64 if case == "unsplit" else 2),
                h_q=heads,
                s_q=3
                if case == "persistent_multiquery"
                else (1 if persistent or case == "unsplit" else 3),
                h_kv=1,
                s_kv=257,
                is_varlen=True,
                topk=127 if case == "extra" else 128,
                d_qk=dim,
                block_size=64 if persistent else 69,
                have_topk_length=lengths,
                enable_attn_sink=case != "plain",
                is_all_indices_invalid=case
                in ("empty", "persistent_empty", "persistent_invalid"),
                extra_s_k=257 if extra else None,
                extra_topk=(512 if persistent else 509) if extra else None,
                extra_block_size=(64 if persistent else 61) if extra else None,
                have_extra_topk_length=case == "extra",
                seed=42,
                num_runs=0,
            ).to_test_param()
            t = lib.generate_testcase_for_decode(p)
        finally:
            torch.set_default_dtype(previous_dtype)
        if case == "persistent_invalid":
            indices = t.kv_scope.indices_in_kvcache
            indices[..., ::2] = 2**31 - 1
            indices[..., 1::2] = -17
        if case in ("unaligned", "persistent_unaligned"):
            cache = t.kv_scope.get_kvcache_for_flash_mla()
            pages = cache.shape[0]
            raw = cache.view(torch.uint8).reshape(pages, -1)
            backing = torch.empty((pages, raw.shape[1] + 1), dtype=torch.uint8)
            backing[:, 1:].copy_(raw)
            t.kv_scope.blocked_k_quantized = backing[:, 1:].view(cache.shape)
            q_storage = torch.empty((*t.q.shape[:-1], dim + 1), dtype=torch.bfloat16)
            t.q = q_storage[..., 1:].copy_(t.q)
        # Match slices of vLLM's padded Q/output buffers; guards catch over-writes.
        storage = torch.full(
            (*t.q.shape[:2], 64, 512), float("nan"), dtype=torch.bfloat16
        )
        out = storage[:, :, 8 : 8 + heads]
        if case == "unaligned_output":
            storage = torch.full(
                (*t.q.shape[:2], 64, 513), float("nan"), dtype=torch.bfloat16
            )
            out = storage[:, :, 8 : 8 + heads, 1:]
        meta, _ = flash_mla.get_mla_metadata()

        def run():
            return flash_mla.flash_mla_with_kvcache(
                t.q,
                t.kv_scope.get_kvcache_for_flash_mla(),
                None,
                None,
                512,
                meta,
                softmax_scale=t.sm_scale,
                is_fp8_kvcache=True,
                indices=t.kv_scope.indices_in_kvcache,
                attn_sink=t.attn_sink,
                topk_length=t.kv_scope.topk_length,
                extra_k_cache=t.extra_kv_scope.get_kvcache_for_flash_mla()
                if extra
                else None,
                extra_indices_in_kvcache=t.extra_kv_scope.indices_in_kvcache
                if extra
                else None,
                extra_topk_length=t.extra_kv_scope.topk_length if extra else None,
                out=out,
                enable_batch_invariant=case in ("invariant", "unsplit"),
            )

        actual = run()
        if persistent and heads == 32:
            uses_persistent = (
                (dim == 512 or not extra)
                and not lengths
                and case != "persistent_unaligned"
            )
            assert (meta.tile_scheduler_metadata is not None) == uses_persistent
        expected = (
            (torch.zeros_like(actual[0]), torch.full_like(actual[1], float("inf")))
            if case == "persistent_invalid"
            else ref.ref_sparse_attn_decode(p, t)
        )
        assert actual[0].data_ptr() == out.data_ptr()
        assert kk.check_is_allclose(
            "out",
            actual[0],
            expected[0],
            abs_tol=1e-3,
            rel_tol=2.01 / 128,
            cos_diff_tol=5e-6,
        )
        torch.testing.assert_close(actual[1], expected[1], atol=1e-5, rtol=8.01 / 65536)
        assert torch.isnan(storage[:, :, :8]).all()
        assert torch.isnan(storage[:, :, 8 + heads :]).all()
        if case == "unaligned_output":
            assert torch.isnan(storage[..., 0]).all()
        saved = tuple(value.clone() for value in actual)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
        torch.testing.assert_close(captured[0], saved[0], atol=0, rtol=0)
        torch.testing.assert_close(captured[1], saved[1], atol=0, rtol=0)
        if case == "invariant":
            single_meta, _ = flash_mla.get_mla_metadata()
            single = flash_mla.flash_mla_with_kvcache(
                t.q[:1],
                t.kv_scope.get_kvcache_for_flash_mla(),
                None,
                None,
                512,
                single_meta,
                softmax_scale=t.sm_scale,
                is_fp8_kvcache=True,
                indices=t.kv_scope.indices_in_kvcache[:1],
                attn_sink=t.attn_sink,
                topk_length=t.kv_scope.topk_length[:1],
                enable_batch_invariant=True,
            )
            torch.testing.assert_close(single[0], saved[0][:1], atol=0, rtol=0)
            torch.testing.assert_close(single[1], saved[1][:1], atol=0, rtol=0)
        if lengths:
            t.kv_scope.topk_length.zero_()
            if extra:
                t.extra_kv_scope.topk_length.zero_()
            graph.replay()
            assert (captured[0] == 0).all()
            assert torch.isposinf(captured[1]).all()


@torch.inference_mode()
def test_flash_mla(p: TestParam) -> Result:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()
    assert p.decode

    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = lib.generate_testcase_for_decode(p)

    tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
    def run_decode():
        return lib.run_flash_mla_decode(p, t, tile_scheduler_metadata, None)
    
    # We first run the kernel once to generate output data for the correctness test
    # We must do this first, otherwise when allocating tensors for storing answers,
    # it may re-use memory that contains the correct answer, leading to false positives
    if p.check_correctness:
        torch.cuda.synchronize()
        out_ans, lse_ans = run_decode()
        torch.cuda.synchronize()
        # torch.set_printoptions(profile='full')
        # print(tile_scheduler_metadata.tile_scheduler_metadata[:, :7])
    
    # We run the performance test before generating the answer for the correctness test to avoid interference
    performance_result = Result(True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if p.num_runs == 0:
        performance_result = Result(True, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        result = kk.bench_kineto(run_decode, p.num_runs)

        splitkv_kernel_name = "flash_fwd_splitkv_mla_fp8_sparse_kernel"
        combine_kernel_name = "flash_fwd_mla_combine_kernel"
        
        # Get individual kernel time usages
        kernel_time_usages_us: Dict[str, Optional[float]] = {}
        def pick_kernel_time_usage(kernel_name: str):
            t = [kernel_name in s for s in result.get_kernel_names()]
            if any(t):
                assert sum(t) == 1
                kernel_time_usages_us[kernel_name] = result.get_kernel_time(kernel_name) * 1e6
            else:
                kernel_time_usages_us[kernel_name] = None
        pick_kernel_time_usage(splitkv_kernel_name)
        pick_kernel_time_usage(combine_kernel_name)

        # Get E2E time usages
        def have_kernel(name: str):
            return kernel_time_usages_us[name] is not None
        
        if kk.is_using_profiling_tools():
            e2e_time_usage_us = 1e6
        else:
            assert have_kernel(splitkv_kernel_name)
            if have_kernel(combine_kernel_name):
                e2e_time_usage_us = result.get_e2e_time(splitkv_kernel_name, combine_kernel_name) * 1e6
            else:
                e2e_time_usage_us = kernel_time_usages_us[splitkv_kernel_name]
        assert e2e_time_usage_us is not None

        flops_and_mem_vol = lib.count_flop_and_mem_vol_for_decode(p, t)

        e2e_time_usage_s = e2e_time_usage_us / 1e6
        theoritical_compute_memory_ratio = flops_and_mem_vol.flop / flops_and_mem_vol.mem_vol
        achieved_tflops = flops_and_mem_vol.flop / e2e_time_usage_s / 1e12
        achieved_gBps = flops_and_mem_vol.mem_vol / e2e_time_usage_s / 1e9
        def print_kernel_time_usage(name: str, short_name: str):
            if kernel_time_usages_us[name] is not None:
                print(f'{short_name} time: {kernel_time_usages_us[name]:.1f} us')
        print(f'Compute/Memory: {theoritical_compute_memory_ratio:.2f}')
        print(f'Time (per): {e2e_time_usage_us:.1f} us')
        print_kernel_time_usage(splitkv_kernel_name, "Splitkv")
        print_kernel_time_usage(combine_kernel_name, "Combine")
        print(f'TFlops: {achieved_tflops:.1f}')
        print(f'GB/s: {achieved_gBps:.0f}')

        performance_result = Result(True, theoritical_compute_memory_ratio, e2e_time_usage_us, kernel_time_usages_us[splitkv_kernel_name] or 0.0, kernel_time_usages_us[combine_kernel_name] or 0.0, achieved_tflops, achieved_gBps)
    
    is_correct = True
    if p.check_correctness:
        torch.cuda.synchronize()
        with torch.profiler.record_function("reference_flash_mla"):
            out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)

        is_out_correct = kk.check_is_allclose("out", out_ans, out_ref, abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=5e-6)
        is_lse_correct = kk.check_is_allclose("lse", lse_ans, lse_ref, abs_tol=1e-6, rel_tol=8.01/65536)
        is_correct &= is_out_correct and is_lse_correct

    performance_result.is_correct = is_correct
    return performance_result


def main():
    dtype = torch.bfloat16
    device = torch.device("cuda:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(32)

    raw_testcases = gen_testcase()
    testcases = [t.to_test_param() for t in raw_testcases]

    print(f"{kk.colors['CYAN_BG']}{len(testcases)} testcases to run{kk.colors['CLEAR']}")

    is_no_cooldown = lib.is_no_cooldown()
    num_testcases_len = len(str(len(testcases)))
    failed_cases = []
    results: List[Tuple[TestParam, Result]] = []
    for testcase_idx, testcase in enumerate(testcases):
        if testcase != testcases[0] and testcase.num_runs > 0 and not is_no_cooldown:
            time.sleep(0.3) # Cooldown
        print(f"[{testcase_idx+1:{num_testcases_len}d}/{len(testcases)}, {testcase_idx/len(testcases)*100:3.0f}%]  ", end='')
        result = test_flash_mla(testcase)
        results.append((testcase, result))
        if not result.is_correct:
            failed_cases.append(testcase)
            import sys
            sys.exit(1)

    console = rich.console.Console(width=120)
    table = rich.table.Table(show_header=True, header_style="bold cyan")
    table.add_column("topk")
    table.add_column("Bsz")
    table.add_column("h_q&k")
    table.add_column("sq")
    table.add_column("sk")
    table.add_column("d_qk")
    table.add_column("Feats")
    table.add_column("C/M")
    table.add_column("TFlops")
    table.add_column("GBps")
    table.add_column("us")
    table.add_column(" ")

    for testcase, result in results:
        assert testcase.decode
        topk_str = f"{testcase.topk}" if testcase.decode.extra_topk is None else f"{testcase.topk}+{testcase.decode.extra_topk}"
        table.add_row(
            topk_str,
            str(testcase.decode.b),
            f"{testcase.h_q:3d} {testcase.h_kv}",
            str(testcase.s_q),
            str(testcase.s_kv),
            str(testcase.d_qk),
            " V"[testcase.decode.is_varlen] + " L"[testcase.have_topk_length] + " E"[testcase.decode.have_extra_topk_length],
            f"{result.compute_memory_ratio:3.0f}",
            f"{result.achieved_tflops:3.0f}",
            f"{result.achieved_gBps:4.0f}",
            f"{result.time_usage_per_us:4.1f}",
            "" if result.is_correct else "X"
        )
    console.print(table)

    def geomean(l) -> float:
        import numpy
        return numpy.exp(numpy.mean(numpy.log(l)))
    
    num_correct_testcases = [result.is_correct for t, result in results if t.check_correctness].count(True)
    num_correctness_cases = sum([1 for t in testcases if t.check_correctness])
    if num_correct_testcases == num_correctness_cases:
        print(f"{kk.colors['GREEN_BG']}{num_correct_testcases}/{num_correctness_cases} correctness cases passed{kk.colors['CLEAR']}")
    else:
        print(f"{kk.colors['RED_BG']}{num_correct_testcases}/{num_correctness_cases} correctness cases passed{kk.colors['CLEAR']}")
        for t in failed_cases:
            print(f"\t{t},")

    valid_achieved_tflops = [result.achieved_tflops for _, result in results if result.achieved_tflops > 0.1]
    if len(valid_achieved_tflops) > 0:
        achieved_tflops_geomean = geomean(valid_achieved_tflops)    # > 0.1 to prune out correctness cases
        print(f"TFlops     geomean: {achieved_tflops_geomean:.1f}")
    

if __name__ == "__main__":
    main()
