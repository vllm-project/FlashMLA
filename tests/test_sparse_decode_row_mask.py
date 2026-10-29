import pytest
import torch

import flash_mla

PAGE, ROWS, H_PAD, D = 64, 8, 64, 512
BYTES_PER_TOKEN = 448 + 64 * 2 + 8  # fp8 NoPE + bf16 RoPE + e8m0 scales


def _make_cache(num_pages: int) -> torch.Tensor:
    cache = torch.randint(0, 64, (num_pages, PAGE, BYTES_PER_TOKEN), dtype=torch.uint8)
    flat = cache.view(num_pages, -1)
    flat[:, : PAGE * 576].view(num_pages, PAGE, 576)[..., 448:] = 0
    flat[:, PAGE * 576 :] = 127  # scale 1.0
    return cache.cuda().unsqueeze(2)


def _make_indices(num_tokens, slots, width, num_keys):
    """Overlapping key sets of unequal length, as consecutive spec tokens produce."""
    base = torch.randperm(num_keys)[: width + 16]
    idx = torch.full((num_tokens, 1, width), -1, dtype=torch.int32)
    lens = torch.empty(num_tokens, dtype=torch.int32)
    for t in range(num_tokens):
        n = width - 4 * (t % slots)
        idx[t, 0, :n] = base[t % slots : t % slots + n].int()
        lens[t] = n
    return idx, lens


def _union(idx, lens, slots):
    """Per request: sorted union of its slots' keys and the bitmask of owning slots."""
    num_reqs, width = idx.shape[0] // slots, idx.shape[-1]
    out_w = (slots * width + 63) // 64 * 64
    u_idx = torch.full((num_reqs, 1, out_w), -1, dtype=torch.int32)
    u_mask = torch.zeros((num_reqs, 1, out_w), dtype=torch.uint8)
    u_len = torch.empty(num_reqs, dtype=torch.int32)
    for r in range(num_reqs):
        bits = {}
        for s in range(slots):
            t = r * slots + s
            for k in idx[t, 0, : lens[t]].tolist():
                bits[k] = bits.get(k, 0) | (1 << s)
        keys = sorted(bits)
        u_idx[r, 0, : len(keys)] = torch.tensor(keys, dtype=torch.int32)
        u_mask[r, 0, : len(keys)] = torch.tensor([bits[k] for k in keys], dtype=torch.uint8)
        u_len[r] = len(keys)
    return u_idx, u_mask, u_len


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="row masks are SM90 only",
)
@pytest.mark.parametrize("slots", [2, 4])
@pytest.mark.parametrize("with_extra", [False, True])
def test_row_mask_matches_unpacked(slots, with_extra):
    """Packing `slots` queries into one head tile with per-key slot masks must
    give each query the same output as attending its own keys unpacked."""
    torch.manual_seed(0)
    num_reqs, num_pages, width, extra_width = 5, 16, 128, 192
    num_tokens = num_reqs * slots
    k_cache = _make_cache(num_pages)
    extra_cache = _make_cache(num_pages) if with_extra else None

    q = torch.randn(num_tokens, H_PAD, D, dtype=torch.bfloat16, device="cuda")
    q[:, ROWS:] = 0
    sink = torch.full((H_PAD,), float("-inf"), device="cuda")
    sink[:ROWS] = torch.randn(ROWS, device="cuda")

    idx, lens = _make_indices(num_tokens, slots, width, num_pages * PAGE)
    ex_idx, ex_lens = _make_indices(num_tokens, slots, extra_width, num_pages * PAGE)

    def run(qq, i, ln, sk, ei=None, el=None, mask=None, emask=None, rows_per_slot=0):
        out, _ = flash_mla.flash_mla_with_kvcache(
            qq,
            k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=D,
            tile_scheduler_metadata=flash_mla.get_mla_metadata()[0],
            is_fp8_kvcache=True,
            indices=i.cuda(),
            topk_length=ln.cuda(),
            softmax_scale=D**-0.5,
            attn_sink=sk,
            extra_k_cache=extra_cache,
            extra_indices_in_kvcache=None if ei is None else ei.cuda(),
            extra_topk_length=None if el is None else el.cuda(),
            indices_mask=None if mask is None else mask.cuda(),
            extra_indices_mask=None if emask is None else emask.cuda(),
            rows_per_slot=rows_per_slot,
        )
        return out

    ref = run(q.unsqueeze(1), idx, lens, sink,
              ex_idx if with_extra else None, ex_lens if with_extra else None)[:, 0, :ROWS]

    u_idx, u_mask, u_len = _union(idx, lens, slots)
    if with_extra:
        eu_idx, eu_mask, eu_len = _union(ex_idx, ex_lens, slots)
    else:
        eu_idx = eu_mask = eu_len = None

    packed_q = torch.zeros(num_reqs, 1, H_PAD, D, dtype=torch.bfloat16, device="cuda")
    packed_q.view(num_reqs, H_PAD // ROWS, ROWS, D)[:, :slots] = (
        q.view(num_reqs, slots, H_PAD, D)[:, :, :ROWS]
    )
    packed_sink = torch.full((H_PAD,), float("-inf"), device="cuda")
    packed_sink[: slots * ROWS] = sink[:ROWS].repeat(slots)

    out = run(packed_q, u_idx, u_len, packed_sink, eu_idx, eu_len,
              u_mask, eu_mask, rows_per_slot=ROWS)
    got = out.view(num_reqs, H_PAD // ROWS, ROWS, D)[:, :slots].reshape(num_tokens, ROWS, D)
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


def _call(**kw):
    k_cache = _make_cache(2)
    q = torch.zeros(1, 1, H_PAD, D, dtype=torch.bfloat16, device="cuda")
    idx = torch.zeros(1, 1, 64, dtype=torch.int32, device="cuda")
    return flash_mla.flash_mla_with_kvcache(
        q, k_cache, block_table=None, cache_seqlens=None, head_dim_v=D,
        tile_scheduler_metadata=flash_mla.get_mla_metadata()[0],
        is_fp8_kvcache=True, indices=idx, **kw,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="row masks are SM90 only",
)
@pytest.mark.parametrize(
    "kwargs, match",
    [
        # masks without rows_per_slot
        (dict(indices_mask="mask", rows_per_slot=0), "rows_per_slot"),
        # 64 / 4 = 16 slots, more than the 8 mask bits
        (dict(indices_mask="mask", rows_per_slot=4), "at most 8 slots"),
        # rows_per_slot without masks
        (dict(rows_per_slot=8), "requires row masks"),
        # wrong dtype
        (dict(indices_mask="bad_dtype", rows_per_slot=8), "uint8"),
    ],
)
def test_row_mask_rejects_invalid_args(kwargs, match):
    masks = {
        "mask": torch.ones(1, 1, 64, dtype=torch.uint8, device="cuda"),
        "bad_dtype": torch.ones(1, 1, 64, dtype=torch.int32, device="cuda"),
    }
    kwargs = {k: masks.get(v, v) if isinstance(v, str) else v for k, v in kwargs.items()}
    with pytest.raises(Exception, match=match):
        _call(**kwargs)
