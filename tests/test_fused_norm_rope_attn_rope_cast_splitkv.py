# Split-KV mode of the fused norm+RoPE+attn+RoPE+cast decode kernel.
#
# Deliberately depends only on torch + flash_mla so it runs anywhere the
# extension builds: it constructs the V4.1 paged FP8 KV records directly
# instead of going through tests/quant.py (kernelkit) or tile_kernels.
#
# The oracle is the kernel itself with mega_num_splits=1, which takes the same
# fp32-partial path over the whole KV range. That isolates the split logic from
# the FP8 epilogue, so a real regression shows up far below quantisation noise.
import math

import pytest
import torch

import flash_mla
from flash_mla import fused_norm_rope_attn_rope_cast as F

D_QK = D_V = 512
ROPE_DIM = 64
QUANT_TILE = 32
NUM_SCALES = D_QK // QUANT_TILE
BYTES_PER_TOKEN = D_QK + NUM_SCALES  # 528: fp8 data region, then ue8m0 scales
WV_GROUP_SIZE = 8
FP8_MAX = 448.0
H_Q = 64  # split-KV is CLUSTER_SIZE == 1 only


def _cos_sin(max_pos, device):
    inv = 1.0 / (10000 ** (torch.arange(0, ROPE_DIM, 2, device=device).float() / ROPE_DIM))
    f = torch.outer(torch.arange(max_pos, device=device).float(), inv)
    return torch.cat([f.cos(), f.sin()], -1).contiguous()


def _quantize_v41(vals):
    n = vals.shape[0]
    g = vals.reshape(n, NUM_SCALES, QUANT_TILE).float()
    amax = g.abs().amax(-1).clamp(min=1e-30)
    exp = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(-127, 127)
    q = (g / torch.exp2(exp).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(n, D_QK).view(torch.uint8), (exp + 127).to(torch.uint8)


def _paged_cache(num_tokens, page, device, gen, positive):
    n_blocks = (num_tokens + page - 1) // page + 1
    n = n_blocks * page
    vals = torch.randn(n, D_QK, device=device, generator=gen, dtype=torch.float32)
    if positive:
        vals = vals.abs() + 0.5
    data, sf = _quantize_v41(vals)
    blk = torch.empty(n_blocks, page * BYTES_PER_TOKEN, dtype=torch.uint8, device=device)
    blk[:, : page * D_QK] = data.reshape(n_blocks, page * D_QK)
    blk[:, page * D_QK :] = sf.reshape(n_blocks, page * NUM_SCALES)
    view = blk.as_strided(
        (n_blocks, page, 1, BYTES_PER_TOKEN),
        (page * BYTES_PER_TOKEN, BYTES_PER_TOKEN, BYTES_PER_TOKEN, 1),
    )
    return view, n


def _inputs(s_q, topk_swa, topk_extra, device, seed, positive):
    gen = torch.Generator(device=device).manual_seed(seed)
    cos_sin = _cos_sin(65536, device)
    q = torch.randn(s_q, H_Q, D_QK, device=device, generator=gen, dtype=torch.bfloat16)
    # 16-element d-chunks interleaved across heads, i.e. the permute_q_b_proj layout
    q = q.reshape(s_q, H_Q, D_QK // 16, 16).permute(0, 2, 1, 3).reshape(s_q, H_Q, D_QK).contiguous()
    swa, n_swa = _paged_cache(s_q * topk_swa + 256, 32, device, gen, positive)
    ext, n_ext = _paged_cache(65536, 128, device, gen, positive)
    return dict(
        q=q, cos_sin=cos_sin, swa=swa, ext=ext,
        swa_idx=torch.randint(0, n_swa, (s_q, topk_swa), device=device, generator=gen, dtype=torch.int32),
        ex_idx=torch.randint(0, n_ext, (s_q, topk_extra), device=device, generator=gen, dtype=torch.int32),
        swa_len=torch.full((s_q,), topk_swa, device=device, dtype=torch.int32),
        ex_len=torch.full((s_q,), topk_extra, device=device, dtype=torch.int32),
        pos=torch.randint(0, 60000, (s_q,), device=device, generator=gen, dtype=torch.int32),
    )


def _decode(t, sink, num_splits, s_q):
    oa = torch.zeros(num_splits, s_q, H_Q, D_V, device=t["q"].device, dtype=torch.float32)
    la = torch.full((num_splits, s_q, H_Q), float("-inf"), device=t["q"].device, dtype=torch.float32)
    F.decode(
        False, 1e-6, t["pos"], False, ROPE_DIM, t["cos_sin"],
        WV_GROUP_SIZE, QUANT_TILE, True, True, True,
        t["q"], t["swa"], t["swa_idx"], D_QK ** -0.5, D_V,
        attn_sink=sink, topk_length=t["swa_len"],
        extra_k_cache=t["ext"], extra_indices_in_kvcache=t["ex_idx"],
        extra_topk_length=t["ex_len"],
        mega_num_splits=num_splits, mega_o_accum=oa, mega_lse_accum=la,
    )
    return oa, la


def _merge(oa, la, sink):
    """Reference merge, mirroring smxx/decode/combine."""
    m = la.max(dim=0).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    g = m + torch.log2(torch.exp2(la - m).sum(dim=0).clamp(min=1e-38))
    if sink is not None:
        sk = sink.float() * math.log2(math.e)
        g = g + torch.log2(1.0 + torch.exp2(sk - g))
    w = torch.nan_to_num(torch.exp2(la - g), nan=0.0, posinf=0.0, neginf=0.0)
    return (oa * w.unsqueeze(-1)).sum(dim=0), g


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("s_q", [1, 6, 16])
@pytest.mark.parametrize("num_splits", [2, 4, 8])
@pytest.mark.parametrize("sink_mode", ["none", "real"])
def test_splitkv_matches_single_split(s_q, num_splits, sink_mode):
    device = torch.device("cuda")
    topk_swa, topk_extra = 128, 512
    # Positive KV keeps the V sum out of heavy cancellation, so the fp32
    # re-association floor stays well below the tolerances below.
    t = _inputs(s_q, topk_swa, topk_extra, device, seed=1234, positive=True)
    sink = None
    if sink_mode == "real":
        sink = torch.randn(H_Q, device=device, dtype=torch.float32)

    ref_o, ref_g = _merge(*_decode(t, sink, 1, s_q), sink)
    got_o, got_g = _merge(*_decode(t, sink, num_splits, s_q), sink)

    torch.testing.assert_close(got_g, ref_g, atol=1e-4, rtol=1e-5)
    scale = ref_o.abs().mean().clamp(min=1e-6)
    assert (got_o - ref_o).abs().mean() / scale < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_splitkv_more_splits_than_kv_blocks():
    """Degenerate splits re-run the last block but must not double count."""
    device = torch.device("cuda")
    t = _inputs(4, 128, 64, device, seed=7, positive=True)  # 3 KV blocks
    ref_o, ref_g = _merge(*_decode(t, None, 1, 4), None)
    got_o, got_g = _merge(*_decode(t, None, 8, 4), None)
    torch.testing.assert_close(got_g, ref_g, atol=1e-4, rtol=1e-5)
    scale = ref_o.abs().mean().clamp(min=1e-6)
    assert (got_o - ref_o).abs().mean() / scale < 1e-3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_unsplit_path_unchanged():
    """num_splits=1 without partial buffers still takes the fused FP8 epilogue."""
    device = torch.device("cuda")
    t = _inputs(6, 128, 512, device, seed=99, positive=True)
    out_fp8, out_sf, lse = F.decode(
        False, 1e-6, t["pos"], False, ROPE_DIM, t["cos_sin"],
        WV_GROUP_SIZE, QUANT_TILE, True, True, True,
        t["q"], t["swa"], t["swa_idx"], D_QK ** -0.5, D_V,
        attn_sink=None, topk_length=t["swa_len"],
        extra_k_cache=t["ext"], extra_indices_in_kvcache=t["ex_idx"],
        extra_topk_length=t["ex_len"],
    )
    assert out_fp8.shape == (6, H_Q // WV_GROUP_SIZE, WV_GROUP_SIZE * D_V)
    assert torch.isfinite(out_fp8.float()).all()
    assert torch.isfinite(lse).all()
