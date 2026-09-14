"""
Plumbing test for the optional FP8 output buffers (`out=(out_fp8, out_sf)`) of the fused
norm + RoPE + attn + RoPE + cast kernel. Self-contained: needs only torch and flash_mla (no
DeepGEMM / TileKernels / kernelkit), so it does not check numerics against a reference. It checks:

  1. writing into caller-provided buffers is bit-identical to the internally allocated outputs
     (prefill and decode, V4 and V4.1 fp8 caches, h_q 64 and 128, s_q 1 included)
  2. a "prefill" launch and a "decode" launch can write disjoint token ranges of ONE shared buffer
     pair whose scale leading dim is ceil4(N) for the total N, bit-identical to standalone runs and
     without touching each other's range. The per-segment leading dim deliberately differs from
     ceil4(segment), which is the case a single deep_gemm.fp8_einsum over all N tokens needs
  3. invalid buffers are rejected

Run: python tests/test_fused_optional_out.py   (or pytest)
"""
import sys

import pytest
import torch

from flash_mla import fused_norm_rope_attn_rope_cast as fused

D = 512
ROPE_DIM = 64
MAX_POS = 8192
WV_GROUP_SIZE = 8
DEV = "cuda"


def build_cos_sin_cache(max_pos: int, base: float = 100000.0) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=DEV) / ROPE_DIM))
    freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=DEV), inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)  # [max_pos, ROPE_DIM]


COS_SIN = build_cos_sin_cache(MAX_POS)


def ceil4(n: int) -> int:
    return (n + 3) // 4 * 4


def alloc_outputs(n_tokens: int, n_wv_group: int):
    """The layout the API allocates internally: contiguous fp8 activation + MN-major, TMA-aligned packed ue8m0 scales."""
    row = WV_GROUP_SIZE * D
    out_fp8 = torch.empty((n_tokens, n_wv_group, row), dtype=torch.float8_e4m3fn, device=DEV)
    sf_buf = torch.empty((n_wv_group, row // 128, ceil4(n_tokens)), dtype=torch.int32, device=DEV)
    out_sf = sf_buf.permute(2, 0, 1)[:n_tokens]  # [n_tokens, n_wv_group, row // 128], strides (1, 32 * ceil4, ceil4)
    return out_fp8, out_sf


# bytes/token, data row width (TMA_K_STRIDE), scale row width, fp8 bytes in the data row, scale bytes in use
KV_FORMATS = {
    "v4": dict(bpt=584, T=576, S=8, fp8_bytes=448, n_scales=7),
    "v41": dict(bpt=528, T=512, S=16, fp8_bytes=512, n_scales=16),
}


def make_kv_cache(fmt: str, num_blocks: int, block_size: int, gen: torch.Generator) -> torch.Tensor:
    """A paged cache with random but finite contents in the block layout the kernels expect
    ([block_size data rows][block_size scale rows][pad to a multiple of TMA_K_STRIDE])."""
    f = KV_FORMATS[fmt]
    stride0 = -(-block_size * f["bpt"] // f["T"]) * f["T"]
    buf = torch.zeros((num_blocks, stride0), dtype=torch.uint8, device=DEV)
    data = buf[:, : block_size * f["T"]].view(num_blocks, block_size, f["T"])
    # e4m3 codes 0x00..0x7E are finite non-negative values (0x7F is NaN)
    data[..., : f["fp8_bytes"]] = torch.randint(
        0, 0x7F, (num_blocks, block_size, f["fp8_bytes"]), dtype=torch.uint8, device=DEV, generator=gen)
    if f["T"] > f["fp8_bytes"]:  # V4: unquantized bf16 RoPE tail
        tail = torch.randn((num_blocks, block_size, (f["T"] - f["fp8_bytes"]) // 2), device=DEV, generator=gen)
        data[..., f["fp8_bytes"]:] = tail.to(torch.bfloat16).view(torch.uint8)
    scales = buf[:, block_size * f["T"]: block_size * f["bpt"]].view(num_blocks, block_size, f["S"])
    scales[..., : f["n_scales"]] = torch.randint(  # ue8m0 exponents for 2^-3 .. 2^3
        124, 131, (num_blocks, block_size, f["n_scales"]), dtype=torch.uint8, device=DEV, generator=gen)
    return buf[:, : block_size * f["bpt"]].view(num_blocks, block_size, 1, f["bpt"])


def prefill_inputs(s_q: int, h_q: int, gen: torch.Generator, s_kv: int = 4096, topk: int = 256):
    q = torch.randn((s_q, h_q, D), dtype=torch.bfloat16, device=DEV, generator=gen)
    kv = torch.randn((s_kv, 1, D), dtype=torch.bfloat16, device=DEV, generator=gen)
    indices = torch.randint(0, s_kv, (s_q, 1, topk), dtype=torch.int32, device=DEV, generator=gen)
    indices[torch.rand(indices.shape, device=DEV, generator=gen) < 0.1] = -1
    if s_q > 1:
        indices[1] = -1  # a token with no valid KV: lse = +inf, out = 0
    positions = torch.randint(0, MAX_POS, (s_q,), dtype=torch.int32, device=DEV, generator=gen)
    attn_sink = torch.randn((h_q,), dtype=torch.float32, device=DEV, generator=gen)
    return q, kv, indices, positions, attn_sink


def decode_inputs(s_q: int, h_q: int, fmt: str, gen: torch.Generator, num_blocks: int = 64, block_size: int = 64, topk: int = 256):
    q = torch.randn((s_q, h_q, D), dtype=torch.bfloat16, device=DEV, generator=gen)
    k_cache = make_kv_cache(fmt, num_blocks, block_size, gen)
    indices = torch.randint(0, num_blocks * block_size, (s_q, topk), dtype=torch.int32, device=DEV, generator=gen)
    indices[torch.rand(indices.shape, device=DEV, generator=gen) < 0.1] = -1
    positions = torch.randint(0, MAX_POS, (s_q,), dtype=torch.int32, device=DEV, generator=gen)
    attn_sink = torch.randn((h_q,), dtype=torch.float32, device=DEV, generator=gen)
    return q, k_cache, indices, positions, attn_sink


def run_prefill(inp, h_q: int, out=None, enable_q_norm: bool = False):
    q, kv, indices, positions, attn_sink = inp
    return fused.prefill(
        enable_q_norm, 1e-4, positions, False, ROPE_DIM, COS_SIN,
        h_q // WV_GROUP_SIZE, 32, True, True, True,
        q, kv, indices, sm_scale=D ** -0.5, attn_sink=attn_sink, out=out)


def run_decode(inp, h_q: int, out=None, enable_q_norm: bool = False):
    q, k_cache, indices, positions, attn_sink = inp
    return fused.decode(
        enable_q_norm, 1e-4, positions, False, ROPE_DIM, COS_SIN,
        h_q // WV_GROUP_SIZE, 32, True, True, True,
        q, k_cache, indices, sm_scale=D ** -0.5, attn_sink=attn_sink, out=out)


def as_bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def assert_same(a: torch.Tensor, b: torch.Tensor, name: str):
    a, b = as_bits(a), as_bits(b)
    assert a.shape == b.shape, f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    assert torch.equal(a, b), f"{name}: values differ"


PREFILL_NAMES = ("out_fp8", "out_sf", "max_logits", "lse")
DECODE_NAMES = ("out_fp8", "out_sf", "lse")


@pytest.mark.parametrize("enable_q_norm", [False, True])
@pytest.mark.parametrize("s_q", [1, 7, 200])
@pytest.mark.parametrize("h_q", [64, 128])
def test_prefill_provided_matches_allocated(h_q, s_q, enable_q_norm):
    gen = torch.Generator(device=DEV).manual_seed(1000 * s_q + h_q + int(enable_q_norm))
    inp = prefill_inputs(s_q, h_q, gen)
    ref = run_prefill(inp, h_q, enable_q_norm=enable_q_norm)
    again = run_prefill(inp, h_q, enable_q_norm=enable_q_norm)
    for a, b, n in zip(ref, again, PREFILL_NAMES):
        assert_same(a, b, f"determinism/{n}")

    out = alloc_outputs(s_q, h_q // WV_GROUP_SIZE)
    got = run_prefill(inp, h_q, out=out, enable_q_norm=enable_q_norm)
    assert got[0].data_ptr() == out[0].data_ptr() and got[1].data_ptr() == out[1].data_ptr()
    for a, b, n in zip(ref, got, PREFILL_NAMES):
        assert_same(a, b, n)


@pytest.mark.parametrize("fmt", ["v4", "v41"])
@pytest.mark.parametrize("s_q", [1, 5, 128])
@pytest.mark.parametrize("h_q", [64, 128])
def test_decode_provided_matches_allocated(h_q, s_q, fmt):
    gen = torch.Generator(device=DEV).manual_seed(1000 * s_q + h_q + len(fmt))
    inp = decode_inputs(s_q, h_q, fmt, gen)
    ref = run_decode(inp, h_q)
    again = run_decode(inp, h_q)
    for a, b, n in zip(ref, again, DECODE_NAMES):
        assert_same(a, b, f"determinism/{n}")

    out = alloc_outputs(s_q, h_q // WV_GROUP_SIZE)
    got = run_decode(inp, h_q, out=out)
    assert got[0].data_ptr() == out[0].data_ptr() and got[1].data_ptr() == out[1].data_ptr()
    for a, b, n in zip(ref, got, DECODE_NAMES):
        assert_same(a, b, n)


@pytest.mark.parametrize("n_p,n_d", [(37, 91), (1, 6), (64, 3)])
@pytest.mark.parametrize("h_q", [64, 128])
def test_shared_buffer_segments(h_q, n_p, n_d):
    """Prefill tokens [0, n_p) and decode tokens [n_p, N) share one buffer pair sized for N."""
    n_g = h_q // WV_GROUP_SIZE
    N = n_p + n_d
    gen = torch.Generator(device=DEV).manual_seed(7 * N + h_q)
    pin = prefill_inputs(n_p, h_q, gen)
    din = decode_inputs(n_d, h_q, "v41", gen)
    ref_p = run_prefill(pin, h_q)
    ref_d = run_decode(din, h_q)

    out_all, sf_all = alloc_outputs(N, n_g)
    assert sf_all.stride(2) == ceil4(N)
    seg_p = (out_all[:n_p], sf_all[:n_p])
    seg_d = (out_all[n_p:], sf_all[n_p:])
    # The segments' scale leading dim is ceil4(N), not ceil4(segment): the API must accept that
    assert seg_p[1].stride(2) != ceil4(n_p) or seg_d[1].stride(2) != ceil4(n_d)

    # Sentinels, so we can see that each launch only touches its own token range. Fill the scale buffer through a view of
    # its whole storage, since `sf_all` itself does not cover the padding columns [N, ceil4(N)).
    out_all.view(torch.uint8).fill_(0xAA)
    sf_full = sf_all.as_strided((n_g, D * WV_GROUP_SIZE // 128, ceil4(N)), (32 * ceil4(N), ceil4(N), 1))
    sf_full.fill_(-1)

    run_prefill(pin, h_q, out=seg_p)
    assert torch.all(as_bits(out_all[n_p:]) == 0xAA) and torch.all(sf_all[n_p:] == -1), "prefill wrote outside its range"
    assert_same(out_all[:n_p], ref_p[0], "prefill/out_fp8")
    assert_same(sf_all[:n_p], ref_p[1], "prefill/out_sf")

    run_decode(din, h_q, out=seg_d)
    assert_same(out_all[:n_p], ref_p[0], "prefill/out_fp8 after decode")
    assert_same(sf_all[:n_p], ref_p[1], "prefill/out_sf after decode")
    assert_same(out_all[n_p:], ref_d[0], "decode/out_fp8")
    assert_same(sf_all[n_p:], ref_d[1], "decode/out_sf")

    # The padding columns [N, ceil4(N)) of the scale buffer are never written
    assert torch.all(sf_full[..., N:] == -1)


def test_rejects_bad_buffers():
    h_q, s_q = 64, 8
    n_g = h_q // WV_GROUP_SIZE
    gen = torch.Generator(device=DEV).manual_seed(0)
    inp = prefill_inputs(s_q, h_q, gen, s_kv=1024, topk=64)
    good_fp8, good_sf = alloc_outputs(s_q, n_g)
    row = WV_GROUP_SIZE * D

    def expect_error(out):
        with pytest.raises(RuntimeError):
            run_prefill(inp, h_q, out=out)

    expect_error((good_fp8, None))                                                   # only one of the pair
    expect_error((torch.empty_like(good_fp8, dtype=torch.bfloat16), good_sf))        # wrong activation dtype
    expect_error((good_fp8, good_sf.to(torch.float32)))                              # wrong scale dtype
    expect_error((good_fp8, good_sf.contiguous()))                                   # token dim not unit-stride
    expect_error(alloc_outputs(s_q + 1, n_g))                                        # wrong token count
    expect_error((torch.empty((s_q, n_g, 2 * row), dtype=torch.float8_e4m3fn, device=DEV)[..., :row], good_sf))  # non-contiguous
    misaligned = torch.empty((s_q * n_g * row + 16,), dtype=torch.float8_e4m3fn, device=DEV)[16:].view(s_q, n_g, row)
    expect_error((misaligned, good_sf))                                              # 16 B off a 32 B boundary
    # Overlapping head-dim columns: a scale view whose leading dim (4) is smaller than s_q (8)
    small = torch.empty((2 * n_g * (row // 128) * 4,), dtype=torch.int32, device=DEV)
    expect_error((good_fp8, small.as_strided((s_q, n_g, row // 128), (1, 32 * 4, 4))))

    # And the good pair still works
    run_prefill(inp, h_q, out=(good_fp8, good_sf))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
