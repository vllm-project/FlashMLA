import time
import sys

import pytest
import torch
import kernelkit as kk

from lib import TestParam
import lib
import ref

_counter = kk.Counter()

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("heads", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("topk", [128, 640])
@pytest.mark.parametrize("features", ["plain", "sink", "length", "both"])
@torch.inference_mode()
def test_tp_heads_sparse_prefill(heads, dim, topk, features):
    """TP-local heads preserve sparse masking, sink and auxiliary output semantics."""
    from flash_mla import flash_mla_sparse_fwd

    with torch.device("cuda"):
        p = TestParam(17, 257, topk, h_q=heads, d_qk=dim, seed=42, num_runs=0,
                      have_attn_sink=features in ("sink", "both"),
                      have_topk_length=features in ("length", "both"))
        t = lib.generate_testcase(p)
        # Match vLLM's slice of a padded Q allocation, including head strides.
        q_storage = torch.empty((p.s_q, heads * 2, dim), dtype=torch.bfloat16)
        q_storage[:, ::2].copy_(t.q)
        t.q = q_storage[:, ::2]
        kv_storage = torch.empty((p.s_kv * 2, 1, dim), dtype=torch.bfloat16)
        kv_storage[::2].copy_(t.kv)
        t.kv = kv_storage[::2]
        indices_storage = torch.empty((p.s_q * 2, 1, topk), dtype=torch.int32)
        indices_storage[::2].copy_(t.indices)
        t.indices = indices_storage[::2]
        t.indices[0].fill_(-1)
        t.indices[1, :, :64] = -1
        t.indices[2, :, ::3] = p.s_kv
        if t.topk_length is not None:
            t.topk_length[:8] = torch.tensor([0, 1, 31, 63, 64, 65, 127, 128], dtype=torch.int32)
        if t.attn_sink is not None:
            t.attn_sink[:2] = torch.tensor([float("inf"), -float("inf")])
        out = torch.full((p.s_q, heads, 512), float("nan"), dtype=torch.bfloat16)
        actual = flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale,
                                     attn_sink=t.attn_sink, topk_length=t.topk_length, out=out)
        assert actual[0].data_ptr() == out.data_ptr()
        _, expected, maximum, lse = ref.ref_sparse_attn_fwd(p, t)
        assert kk.check_is_allclose("out", actual[0].float(), expected, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
        torch.testing.assert_close(actual[1], maximum, atol=1e-5, rtol=2.01/65536)
        torch.testing.assert_close(actual[2], lse, atol=1e-5, rtol=2.01/65536)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("head_offset", [0, 8])
@pytest.mark.parametrize("heads", [8, 16, 32])
@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("topk", [128, 640])
@torch.inference_mode()
def test_small_heads_write_into_padded_head_slice(heads, dim, topk, head_offset):
    """The SM90 small-heads kernel writes through the output strides of a head slice."""
    from flash_mla import flash_mla_sparse_fwd

    with torch.device("cuda"):
        p = TestParam(17, 257, topk, h_q=heads, d_qk=dim, seed=42, num_runs=0,
                      have_attn_sink=True, have_topk_length=True)
        t = lib.generate_testcase(p)
        t.topk_length[:8] = torch.tensor([0, 1, 31, 63, 64, 65, 127, 128], dtype=torch.int32)
        padded_storage = torch.full((p.s_q, 64, 512), float("nan"), dtype=torch.bfloat16)
        out = padded_storage[:, head_offset:head_offset + heads]
        assert out.stride(1) == 512 and out.stride(0) == 64 * 512
        actual = flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale,
                                     attn_sink=t.attn_sink, topk_length=t.topk_length, out=out)
        assert actual[0].data_ptr() == out.data_ptr()
        _, expected, maximum, lse = ref.ref_sparse_attn_fwd(p, t)
        assert kk.check_is_allclose("out", actual[0].float(), expected, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
        torch.testing.assert_close(actual[1], maximum, atol=1e-5, rtol=2.01/65536)
        torch.testing.assert_close(actual[2], lse, atol=1e-5, rtol=2.01/65536)
        untouched = torch.cat([padded_storage[:, :head_offset], padded_storage[:, head_offset + heads:]], dim=1)
        assert torch.isnan(untouched).all(), "kernel wrote outside the supplied head slice"


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("heads", [8, 16, 32])
@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("overlap", ["query", "head", "partial_query"])
def test_small_heads_reject_overlapping_output(heads, dim, overlap):
    from flash_mla import flash_mla_sparse_fwd

    q = torch.zeros((3, heads, dim), device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros((17, 1, dim), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros((3, 1, 32), device="cuda", dtype=torch.int32)
    storage = torch.empty((3, heads, 512), device="cuda", dtype=torch.bfloat16)
    if overlap == "query":
        out = storage[:1].expand(3, heads, 512)
    elif overlap == "head":
        out = storage.as_strided((3, heads, 512), (heads * 512, 256, 1))
    else:
        out = storage.as_strided((3, heads, 512), (heads * 512 - 1, 512, 1))
    with pytest.raises(RuntimeError, match="out must not have overlapping rows"):
        flash_mla_sparse_fwd(q, kv, indices, dim ** -0.5, out=out)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("layout", ["head_slice", "head_step"])
def test_large_heads_reject_strided_output(heads, dim, layout):
    from flash_mla import flash_mla_sparse_fwd

    q = torch.zeros((3, heads, dim), device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros((17, 1, dim), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros((3, 1, 128), device="cuda", dtype=torch.int32)
    storage = torch.empty((3, heads * 2, 512), device="cuda", dtype=torch.bfloat16)
    out = storage[:, 8:8 + heads] if layout == "head_slice" else storage[:, ::2]
    with pytest.raises(RuntimeError, match="contiguous"):
        flash_mla_sparse_fwd(q, kv, indices, dim ** -0.5, out=out)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("heads", [8, 16, 32])
@torch.inference_mode()
def test_tp_heads_graph_updates_sparse_lengths(heads):
    """Graph replay reads current device lengths, including fully empty rows."""
    from flash_mla import flash_mla_sparse_fwd

    with torch.device("cuda"):
        p = TestParam(17, 257, 640, h_q=heads, seed=7, num_runs=0,
                      have_topk_length=True, have_attn_sink=True)
        t = lib.generate_testcase(p)
        def run():
            return flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale,
                                       attn_sink=t.attn_sink, topk_length=t.topk_length)
        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        for length in [0, 1, 65, 640]:
            t.topk_length.fill_(length)
            graph.replay()
            expected = run()
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("heads", [20, 24, 48])
@torch.inference_mode()
def test_unsupported_head_counts_are_rejected(heads):
    """Head counts below 64 that are not 8, 16 or 32 must not fall through to the 64-head tile."""
    from flash_mla import flash_mla_sparse_fwd

    with torch.device("cuda"):
        p = TestParam(17, 257, 128, h_q=heads, seed=42, num_runs=0)
        t = lib.generate_testcase(p)
    with pytest.raises(RuntimeError, match="Unsupported h_q"):
        flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale)




@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("heads", [8, 16, 32])
@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("topk", [1, 7, 33, 100, 129])
@torch.inference_mode()
def test_small_heads_unaligned_strides_and_tail(heads, dim, topk):
    """Exercise scalar loads/stores, partial key tiles and clamped lengths."""
    from flash_mla import flash_mla_sparse_fwd

    with torch.device("cuda"):
        p = TestParam(9, 257, topk, h_q=heads, d_qk=dim, seed=19, num_runs=0,
                      have_attn_sink=True, have_topk_length=True)
        t = lib.generate_testcase(p)
        # Odd row strides and an offset pointer force the scalar fallback paths.
        q_storage = torch.empty((p.s_q, heads, dim + 1), dtype=torch.bfloat16)
        t.q = q_storage[..., 1:].copy_(t.q)
        kv_storage = torch.empty((p.s_kv, 1, dim + 1), dtype=torch.bfloat16)
        t.kv = kv_storage[..., 1:].copy_(t.kv)
        t.indices[0].fill_(-1)
        t.indices[1].fill_(p.s_kv)
        t.topk_length[:5] = torch.tensor([-1, 0, 1, topk, topk + 7], dtype=torch.int32)
        storage = torch.full((p.s_q, heads * 2, 513), float("nan"), dtype=torch.bfloat16)
        out = storage[:, ::2, 1:]
        actual = flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale,
                                     attn_sink=t.attn_sink, topk_length=t.topk_length, out=out)
        _, expected, maximum, lse = ref.ref_sparse_attn_fwd(p, t)
        assert actual[0].data_ptr() == out.data_ptr()
        assert kk.check_is_allclose("out", actual[0].float(), expected, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
        torch.testing.assert_close(actual[1], maximum, atol=1e-5, rtol=2.01/65536)
        torch.testing.assert_close(actual[2], lse, atol=1e-5, rtol=2.01/65536)
        assert torch.isnan(storage[:, 1::2]).all()
        assert torch.isnan(storage[:, :, 0]).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("heads", [8, 16, 32])
@pytest.mark.parametrize("shape", [(0, 17, 33), (3, 0, 33), (3, 17, 0)])
@torch.inference_mode()
def test_small_heads_empty_inputs(heads, shape):
    from flash_mla import flash_mla_sparse_fwd

    s_q, s_kv, topk = shape
    q = torch.zeros((s_q, heads, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros((s_kv, 1, 512), device="cuda", dtype=torch.bfloat16)
    indices = torch.full((s_q, 1, topk), -1, device="cuda", dtype=torch.int32)
    out, maximum, lse = flash_mla_sparse_fwd(q, kv, indices, 512 ** -0.5)
    assert out.shape == (s_q, heads, 512)
    assert (out == 0).all()
    assert torch.isneginf(maximum).all()
    assert torch.isposinf(lse).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("heads", [8, 16, 32])
@pytest.mark.parametrize("dim", [512, 576])
@pytest.mark.parametrize("topk", [64, 192, 640, 2048])
@torch.inference_mode()
def test_small_heads_agree_with_padded_heads(heads, dim, topk):
    """Small heads match the 64-head kernel on zero-padded Q, over more queries than SMs."""
    from flash_mla import flash_mla_sparse_fwd

    with torch.device("cuda"):
        # More queries than SMs, so that a CTA of the persistent small-heads kernel
        # handles several queries of different lengths, including empty ones.
        p = TestParam(301, 4096, topk, h_q=heads, d_qk=dim, seed=3, num_runs=0,
                      have_attn_sink=True, have_topk_length=True)
        t = lib.generate_testcase(p)
        t.indices[0].fill_(-1)
        t.indices[1, :, :64] = -1
        t.topk_length[:4] = torch.tensor([0, 1, 64, topk], dtype=torch.int32)
        t.topk_length[100:120] = 0
        t.topk_length[200:220] = torch.arange(20, dtype=torch.int32) % 3 * 64
        actual = flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale,
                                     attn_sink=t.attn_sink, topk_length=t.topk_length)
        # Fold the lengths into the indices and pad topk to a multiple of 128 for the 64-head kernel.
        indices = torch.full((p.s_q, 1, (topk + 127) // 128 * 128), -1, dtype=torch.int32)
        indices[..., :topk] = t.indices
        indices[..., :topk][torch.arange(topk).view(1, 1, topk) >= t.topk_length.view(-1, 1, 1)] = -1
        q_padded = torch.zeros((p.s_q, 64, dim), dtype=torch.bfloat16)
        q_padded[:, :heads] = t.q
        sink_padded = torch.full((64,), float("-inf"))
        sink_padded[:heads] = t.attn_sink
        expected = flash_mla_sparse_fwd(q_padded, t.kv, indices, t.sm_scale, attn_sink=sink_padded)
        torch.testing.assert_close(actual[0], expected[0][:, :heads], atol=0.02, rtol=0.03)
        for a, b in zip(actual[1:], expected[1:]):
            torch.testing.assert_close(a, b[:, :heads], atol=1e-5, rtol=4e-5)
        _, out_ref, maximum, lse = ref.ref_sparse_attn_fwd(p, t)
        assert kk.check_is_allclose("out", actual[0].float(), out_ref, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
        torch.testing.assert_close(actual[1], maximum, atol=1e-5, rtol=2.01/65536)
        torch.testing.assert_close(actual[2], lse, atol=1e-5, rtol=2.01/65536)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("heads", [8, 16, 32])
@pytest.mark.parametrize("dim", [512, 576])
@torch.inference_mode()
def test_small_heads_are_deterministic(heads, dim):
    """Repeated runs are bitwise identical, which catches races between the pipeline stages."""
    from flash_mla import flash_mla_sparse_fwd

    with torch.device("cuda"):
        p = TestParam(1024, 8192, 1024, h_q=heads, d_qk=dim, seed=5, num_runs=0, have_attn_sink=True)
        t = lib.generate_testcase(p)
        expected = flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale, attn_sink=t.attn_sink)
        for _ in range(20):
            actual = flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale, attn_sink=t.attn_sink)
            for a, b in zip(actual, expected):
                assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("heads", [8, 16, 32])
@pytest.mark.parametrize("dim", [512, 576])
@torch.inference_mode()
def test_small_heads_mma_switch_agrees(heads, dim, monkeypatch):
    """USE_MMA_FOR_SMALL_N_HEAD=1 selects the mma.sync kernel, which must agree with the default."""
    from flash_mla import flash_mla_sparse_fwd

    with torch.device("cuda"):
        p = TestParam(65, 1000, 192, h_q=heads, d_qk=dim, seed=23, num_runs=0,
                      have_attn_sink=True, have_topk_length=True)
        t = lib.generate_testcase(p)
        t.topk_length[:4] = torch.tensor([0, 1, 33, 192], dtype=torch.int32)
        monkeypatch.delenv("USE_MMA_FOR_SMALL_N_HEAD", raising=False)
        expected = flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale,
                                       attn_sink=t.attn_sink, topk_length=t.topk_length)
        for value in ("1", "true"):
            monkeypatch.setenv("USE_MMA_FOR_SMALL_N_HEAD", value)
            actual = flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale,
                                         attn_sink=t.attn_sink, topk_length=t.topk_length)
            torch.testing.assert_close(actual[0], expected[0], atol=0.02, rtol=0.03)
            for a, b in zip(actual[1:], expected[1:]):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=4e-5)
        # "0" keeps the default kernel
        monkeypatch.setenv("USE_MMA_FOR_SMALL_N_HEAD", "0")
        actual = flash_mla_sparse_fwd(t.q, t.kv, t.indices, t.sm_scale,
                                     attn_sink=t.attn_sink, topk_length=t.topk_length)
        for a, b in zip(actual, expected):
            assert torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9, reason="SM90 kernel")
@pytest.mark.parametrize("heads", [8, 64])
def test_sparse_prefill_rejects_multiple_kv_heads(heads):
    from flash_mla import flash_mla_sparse_fwd

    q = torch.zeros((3, heads, 512), device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros((17, 2, 512), device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros((3, 2, 32), device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="h_kv == 1"):
        flash_mla_sparse_fwd(q, kv, indices, 512 ** -0.5)


@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()

    print("================")
    print(f"Running on {p}")
    torch.cuda.empty_cache()

    t = lib.generate_testcase(p)
    torch.cuda.synchronize()
    
    def run_prefill():
        return lib.run_flash_mla_sparse_fwd(p, t)
    
    prefill_ans_out, prefill_ans_max_logits, prefill_ans_lse = run_prefill()
    torch.cuda.synchronize()

    if p.num_runs > 0:
        flops_and_mem_vol = lib.count_flop_and_mem_vol(p, t)
        prefill_ans_time = kk.bench_kineto(run_prefill, num_tests=p.num_runs).get_kernel_time("sparse_attn_fwd")
        prefill_flops = flops_and_mem_vol.fwd_flop/prefill_ans_time/1e12
        prefill_mem_bw = flops_and_mem_vol.fwd_mem_vol/prefill_ans_time/1e12
        print(f"Prefill:  {prefill_ans_time*1e6:4.0f} us, {prefill_flops:6.1f} TFlops, {prefill_mem_bw:4.2f} TBps")

    if p.check_correctness:
        torch.cuda.synchronize()
        ref_out, ref_out_fp32, ref_max_logits, ref_lse = ref.ref_sparse_attn_fwd(p, t)
        ref_lse[ref_lse == float("-inf")] = float("+inf")
        torch.cuda.synchronize()

        is_correct = True
        is_correct &= kk.check_is_allclose("out", prefill_ans_out.float(), ref_out_fp32, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
        is_correct &= kk.check_is_allclose("max_logits", prefill_ans_max_logits, ref_max_logits, abs_tol=1e-6, rel_tol=2.01/65536)
        is_correct &= kk.check_is_allclose("lse", prefill_ans_lse, ref_lse, abs_tol=1e-6, rel_tol=2.01/65536)

        return is_correct
    else:
        return True


if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    head_counts = [128, 64]
    if torch.cuda.get_device_capability()[0] == 9:
        head_counts += [8, 16, 32]

    correctness_cases = [
        # Regular shapes
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in head_counts
        for s_kv, topk in [
            # Regular shapes
            (128, 128),
            (256, 256),
            (512, 512),

            # Irregular shapes
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            # Irregular shapes with OOB TopK
            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [
            1, 62, 213
        ]
    ]

    correctness_cases_with_features = [
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, have_attn_sink=have_attn_sink, have_topk_length=have_topk_length, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in head_counts
        for s_kv, topk in [
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [62, 213]
        for have_attn_sink in [False, True]
        for have_topk_length in [False, True]
    ]

    corner_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, is_all_indices_invalid=True, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in head_counts
        for s_q, s_kv, topk in [
            (1, 128, 128),
            (1, 256, 256),
            (1234, 4321, 4096),
            (4096, 2048, 2048)
        ]
    ] + [
        # In these cases, some blocks may not have any valid topk indices
        TestParam(s_q, s_kv, topk, h_q=h_q, is_all_indices_invalid=False, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in head_counts
        for s_kv, topk in [
            (32, 2048),
            (64, 8192)
        ]
        for s_q in [1, 1024]
    ] + [
        # In this testcase, s_q is really large, so we cannot put it on the second dimension of grid shape
        TestParam(70000, 256, 256, h_q=h_q, check_correctness=False, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in head_counts
    ]

    performance_case_templates = [
        # V3.2
        (576, 128, 2048, [8192, 32768, 65536, 98304, 131072]),
        # DeepSeek-V4 CONFIG1
        (512, 64, 512, [8192, 32768, 49152, 65536]),
        # DeepSeek-V4 CONFIG2
        (512, 128, 1024, [8192, 32768, 49152, 65536]),
    ]

    performance_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, d_qk=d_qk, have_attn_sink=True)
        for (d_qk, h_q, topk, s_kv_list) in performance_case_templates
        for s_q in [4096]
        for s_kv in s_kv_list
    ]

    testcases = correctness_cases + correctness_cases_with_features + corner_cases + performance_cases

    is_no_cooldown = lib.is_no_cooldown()
    failed_cases = []
    for test in testcases:
        if test != testcases[0] and test.num_runs > 0 and not is_no_cooldown:
            time.sleep(0.3)
        is_correct = run_test(test)
        if not is_correct:
            failed_cases.append(test)
    
    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(testcases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\033[32m\033[1mAll {len(testcases)} cases passed!\033[0m")

