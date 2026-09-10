import torch

import flash_mla.flash_mla_interface as interface


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

    result, _, _ = interface.flash_mla_sparse_fwd(
        q, kv, indices, sm_scale=0.1, out=out
    )

    assert fake_ops.prefill_args[-1] is out
    assert result is out
