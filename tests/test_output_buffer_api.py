import pytest
import torch

import flash_mla.flash_mla_interface as interface
from flash_mla import fused_norm_rope_attn_rope_cast as fused


class _FakeOps:
    def __init__(self):
        self.decode_args = None
        self.prefill_args = None

    def sparse_decode_fwd(self, *args):
        self.decode_args = args
        return args[-1], torch.empty(0), None, None

    def dense_decode_fwd(self, *args):
        self.decode_args = args
        return args[-1], torch.empty(0), None, None

    def sparse_prefill_fwd(self, *args):
        self.prefill_args = args
        return [args[-1], torch.empty(0), torch.empty(0)]


def test_sparse_decode_forwards_optional_output(monkeypatch):
    fake_ops = _FakeOps()
    monkeypatch.setattr(interface, "flash_mla_cuda", fake_ops)

    q = torch.empty((1, 1, 64, 576), dtype=torch.bfloat16)
    k_cache = torch.empty((1, 64, 1, 656), dtype=torch.uint8)
    indices = torch.zeros((1, 1, 1), dtype=torch.int32)
    out = torch.empty((1, 1, 64, 512), dtype=torch.bfloat16)

    result, _ = interface.flash_mla_with_kvcache(
        q,
        k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=interface.FlashMLASchedMeta(),
        is_fp8_kvcache=True,
        indices=indices,
        out=out,
    )

    assert fake_ops.decode_args[-1] is out
    assert result is out


def test_dense_decode_forwards_optional_output(monkeypatch):
    fake_ops = _FakeOps()
    monkeypatch.setattr(interface, "flash_mla_cuda", fake_ops)

    q = torch.empty((1, 1, 64, 576), dtype=torch.bfloat16)
    k_cache = torch.empty((1, 64, 1, 576), dtype=torch.bfloat16)
    block_table = torch.zeros((1, 1), dtype=torch.int32)
    cache_seqlens = torch.ones((1,), dtype=torch.int32)
    out = torch.empty((1, 1, 64, 512), dtype=torch.bfloat16)

    result, _ = interface.flash_mla_with_kvcache(
        q,
        k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=512,
        tile_scheduler_metadata=interface.FlashMLASchedMeta(),
        out=out,
    )

    assert fake_ops.decode_args[-1] is out
    assert result is out


def test_sparse_prefill_forwards_optional_output(monkeypatch):
    fake_ops = _FakeOps()
    monkeypatch.setattr(interface, "flash_mla_cuda", fake_ops)

    q = torch.empty((1, 64, 576), dtype=torch.bfloat16)
    kv = torch.empty((1, 1, 576), dtype=torch.bfloat16)
    indices = torch.zeros((1, 1, 1), dtype=torch.int32)
    out = torch.empty((1, 64, 512), dtype=torch.bfloat16)

    result, _, _ = interface.flash_mla_sparse_fwd(q, kv, indices, sm_scale=0.1, out=out)

    assert fake_ops.prefill_args[-1] is out
    assert result is out


# The fused path writes both values and packed scales; compare its allocating
# API against caller-owned, sliced destinations and guard their neighboring rows.


def _fused_case(mode, rows, heads=64):
    torch.manual_seed(19)
    q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    kv = torch.randn(128, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    ids = torch.arange(128, device="cuda", dtype=torch.int32).expand(rows, -1).clone()
    ids[:, -2:] = -1
    positions = torch.arange(rows, device="cuda", dtype=torch.int32)
    angles = (
        torch.arange(rows + 1, device="cuda").float()[:, None]
        * torch.linspace(0.01, 0.1, 32, device="cuda")[None, :]
    )
    rope = torch.cat((angles.cos(), angles.sin()), dim=1)
    kwargs = dict(
        enable_q_norm=True,
        rms_norm_eps=1e-6,
        token_positions=positions,
        is_rope_neox_style=False,
        rope_dim=64,
        cos_sin_cache=rope,
        n_wv_group=heads // 8,
        num_per_channels=32,
        use_tma_aligned_col_major_sf=True,
        round_sf=True,
        use_packed_ue8m0=True,
        q=q,
        sm_scale=512**-0.5,
    )
    if mode == "prefill":
        kwargs.update(kv=kv, indices=ids[:, None])
        fn = fused.prefill
    else:
        # V4.1 cache: all FP8 value rows followed by all per-32 UE8M0 rows.
        payload = kv.to(torch.float8_e4m3fn).view(torch.uint8).flatten()
        scales = torch.full((128 * 16,), 127, dtype=torch.uint8, device="cuda")
        cache = torch.cat((payload, scales)).view(1, 128, 1, 528)
        kwargs.update(k_cache=cache, indices_in_kvcache=ids)
        fn = fused.decode
    return fn, kwargs


def _fused_buffers(rows, groups):
    values = torch.full(
        (rows + 2, groups, 4096), 3, dtype=torch.float8_e4m3fn, device="cuda"
    )
    padded_rows = ((rows + 2 + 3) // 4) * 4
    scale_storage = torch.full(
        (groups, 32, padded_rows), 0x5A5A5A5A, dtype=torch.int32, device="cuda"
    )
    scales = scale_storage.permute(2, 0, 1)
    return values, scales, values[1 : rows + 1], scales[1 : rows + 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM100")
@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize("rows,heads", [(1, 64), (3, 64), (7, 128), (64, 64)])
def test_fused_caller_outputs_preserve_values_scales_and_guards(mode, rows, heads):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("fused kernels require SM100")
    fn, kwargs = _fused_case(mode, rows, heads)
    reference = fn(**kwargs)
    values, scales, out, sf = _fused_buffers(rows, heads // 8)
    actual = fn(**kwargs, out_fp8=out, out_sf=sf)
    assert actual[0].data_ptr() == out.data_ptr()
    assert actual[1].data_ptr() == sf.data_ptr()
    torch.testing.assert_close(
        out.view(torch.uint8), reference[0].view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(sf, reference[1], rtol=0, atol=0)
    for got, want in zip(actual[2:], reference[2:]):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert torch.all(values[[0, rows + 1]].float() == 3)
    assert torch.all(scales[[0, rows + 1]] == 0x5A5A5A5A)
    # An independently omitted destination still gets allocated.
    one = fn(**kwargs, out_fp8=out)
    torch.testing.assert_close(one[1], reference[1], rtol=0, atol=0)
    other = fn(**kwargs, out_sf=sf)
    torch.testing.assert_close(
        other[0].view(torch.uint8), reference[0].view(torch.uint8), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM100")
@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_fused_output_reuse_under_cuda_graph(mode):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("fused kernels require SM100")
    fn, kwargs = _fused_case(mode, 3)
    _, _, out, sf = _fused_buffers(3, 8)
    fn(**kwargs, out_fp8=out, out_sf=sf)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(**kwargs, out_fp8=out, out_sf=sf)
    previous = out.view(torch.uint8).clone()
    for _ in range(2):
        kwargs["q"].add_(0.125)
        graph.replay()
        reference = fn(**kwargs)
        torch.testing.assert_close(
            out.view(torch.uint8), reference[0].view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(sf, reference[1], rtol=0, atol=0)
        assert not torch.equal(out.view(torch.uint8), previous)
        previous.copy_(out.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM100")
@pytest.mark.parametrize("mode", ["prefill", "decode"])
@pytest.mark.parametrize(
    "bad",
    [
        "value_dtype",
        "value_shape",
        "scale_dtype",
        "scale_layout",
        "scale_overlap",
        "device",
    ],
)
def test_fused_rejects_invalid_output_destinations(mode, bad):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("fused kernels require SM100")
    fn, kwargs = _fused_case(mode, 3)
    _, _, out, sf = _fused_buffers(3, 8)
    if bad == "value_dtype":
        out = out.to(torch.bfloat16)
    elif bad == "value_shape":
        out = out[:2]
    elif bad == "scale_dtype":
        sf = sf.to(torch.float32)
    elif bad == "scale_layout":
        sf = sf.contiguous()
    elif bad == "scale_overlap":
        sf = sf.as_strided(sf.shape, (1, 1, sf.stride(2)))
    elif bad == "device":
        out = out.cpu()
    with pytest.raises(RuntimeError):
        fn(**kwargs, out_fp8=out, out_sf=sf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires SM100")
@pytest.mark.parametrize("rows,heads", [(1, 64), (3, 128)])
def test_fused_decode_caller_outputs_with_fp4_extra_cache(rows, heads):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("fused kernels require SM100")
    fn, kwargs = _fused_case("decode", rows, heads)
    # Each FP4 nibble is +0.5, with a per-16 FP8 scale of 1.
    payload = torch.full((128 * 256,), 0x11, dtype=torch.uint8, device="cuda")
    scales = torch.ones((128 * 32,), dtype=torch.float8_e4m3fn, device="cuda")
    kwargs["extra_k_cache"] = torch.cat((payload, scales.view(torch.uint8))).view(
        1, 128, 1, 288
    )
    kwargs["extra_indices_in_kvcache"] = kwargs["indices_in_kvcache"]
    reference = fn(**kwargs)
    _, _, out, sf = _fused_buffers(rows, heads // 8)
    result = fn(**kwargs, out_fp8=out, out_sf=sf)
    assert result[0].data_ptr() == out.data_ptr()
    assert result[1].data_ptr() == sf.data_ptr()
    torch.testing.assert_close(
        out.view(torch.uint8), reference[0].view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(sf, reference[1], rtol=0, atol=0)
