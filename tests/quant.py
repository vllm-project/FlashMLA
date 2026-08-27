import enum
from typing import Tuple

import torch

class FP8KVCacheLayout(enum.Enum):
    V32_FP8Sparse = 1
    MODEL1_FP8Sparse = 2
    NVFP4_FP8Rope = 3   # NVFP4 (e2m1, per-16 e4m3 SF) NoPE + FP8 (e4m3, unscaled) RoPE, 352B/token

    def get_meta(self) -> Tuple[int, int, int, int, int]:
        # Return: (d, d_nope, d_rope, tile_size, num_tiles)
        return {
            FP8KVCacheLayout.V32_FP8Sparse: (576, 512, 64, 128, 4),
            FP8KVCacheLayout.MODEL1_FP8Sparse: (512, 448, 64, 64, 7),
            FP8KVCacheLayout.NVFP4_FP8Rope: (576, 512, 64, 16, 32),
        }[self]

    def is_nvfp4(self) -> bool:
        return self is FP8KVCacheLayout.NVFP4_FP8Rope

    def bytes_per_token(self) -> int:
        return {
            FP8KVCacheLayout.V32_FP8Sparse: 656,
            FP8KVCacheLayout.MODEL1_FP8Sparse: 584,
            FP8KVCacheLayout.NVFP4_FP8Rope: 352,
        }[self]

def _cast_scale_inv_to_ue8m0(scales_inv: torch.Tensor, out_dtype = torch.float32) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(out_dtype)

# The 8 non-negative values representable in fp4 e2m1
_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Midpoints between consecutive e2m1 values, and the round-to-nearest-EVEN winner at each midpoint
_E2M1_MIDPOINTS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
_E2M1_TIE_CODES = [0, 2, 2, 4, 4, 6, 6]

def _cast_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even quantization to e2m1. Returns uint8 nibble codes (sign<<3 | mag)."""
    xf = x.float()
    xa = xf.abs().clamp(max=6.0)
    mids = torch.tensor(_E2M1_MIDPOINTS, device=x.device, dtype=torch.float32)
    codes = torch.bucketize(xa, mids, right=True).to(torch.uint8)  # x == midpoint goes UP here...
    for mid, tie_code in zip(_E2M1_MIDPOINTS, _E2M1_TIE_CODES):    # ...and is fixed to the even value here
        codes = torch.where(xa == mid, torch.tensor(tie_code, dtype=torch.uint8, device=x.device), codes)
    codes = codes | (xf < 0).to(torch.uint8) * 8
    return codes

def _e2m1_codes_to_float(codes: torch.Tensor) -> torch.Tensor:
    """Decode uint8 nibble codes (low 4 bits used) to float32 values."""
    table = torch.tensor(_E2M1_VALUES + [-v for v in _E2M1_VALUES], device=codes.device, dtype=torch.float32)
    return table[codes.long() & 0xF]

def _pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """Pack e2m1 nibble codes pairwise into bytes: low nibble = even element, high nibble = odd."""
    assert codes.shape[-1] % 2 == 0
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)

def _unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_e2m1. Returns nibble codes with last dim doubled."""
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    return torch.stack([lo, hi], dim=-1).flatten(start_dim=-2)

# --- NVFP4 scale-factor permutation -------------------------------------------------
# The kernel's dequant warpgroup gives each thread 8 of a token's 32 scale factors: thread
# q (= idx_in_group/2, in [0,4)) owns element blocks {4c + q : c = 0..7}, which in element
# order are 8 bytes with stride 4. The on-wire tail stores them permuted so those 8 are
# contiguous and can be fetched with a single 8-byte load:
#     scale for element block s  ->  byte  8*(s & 3) + (s >> 2)
# Keep this in lockstep with NVFP4_SF_NOPE_OFFSET/nvfp4_sf_byte in
# csrc/sm100/decode/head64/config.h and with the production writer in vLLM.
_NVFP4_SF_COLS = 8    # scale factors owned by one dequant thread (kernel COLS_PER_GROUP)
_NVFP4_SF_QUADS = 4   # distinct thread-quarter indices q (kernel GROUP_SIZE/2)

def _nvfp4_permute_sf(sf: torch.Tensor) -> torch.Tensor:
    """[..., 32] in element-block order -> [..., 32] in on-wire (kernel) order."""
    return sf.unflatten(-1, (_NVFP4_SF_COLS, _NVFP4_SF_QUADS)).transpose(-1, -2).flatten(-2)

def _nvfp4_unpermute_sf(sf: torch.Tensor) -> torch.Tensor:
    """Inverse of _nvfp4_permute_sf."""
    return sf.unflatten(-1, (_NVFP4_SF_QUADS, _NVFP4_SF_COLS)).transpose(-1, -2).flatten(-2)

def _quant_tiles_e4m3_sf(x: torch.Tensor, tile_size: int, max_val: float):
    """
    Per-`tile_size` quantization with e4m3 scale factors: sf = e4m3(amax/max_val)
    rounded UP to the next representable e4m3 value (so that amax/float(sf) never
    exceeds max_val — round-to-nearest can round down by up to 12.5% in e4m3's
    subnormal range, saturating the largest values of a tile), q = x / float(sf).
    Returns (x_scaled, sf) where x_scaled is float32 (not yet cast to the target
    dtype) of the same shape as x, and sf is float8_e4m3fn of shape
    (*x.shape[:-1], x.shape[-1]//tile_size).
    """
    tiles = x.float().unflatten(-1, (-1, tile_size))    # [..., num_tiles, tile_size]
    amax = tiles.abs().amax(dim=-1)
    sf_target = torch.clamp_min(amax / max_val, 2.0**-9)
    sf = sf_target.to(torch.float8_e4m3fn)
    # Round up: positive e4m3 bit patterns are monotonic (0x7E = 448 is max finite)
    sf_bits = sf.view(torch.uint8)
    bump = (sf.float() < sf_target) & (sf_bits < 0x7E)
    sf = torch.where(bump, (sf_bits + 1).view(torch.float8_e4m3fn), sf)
    x_scaled = tiles / sf.float().unsqueeze(-1)
    return x_scaled.flatten(-2), sf

def quantize_k_cache(
    input_k_cache: torch.Tensor,    # (num_blocks, block_size, h_k, d)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    Quantize the k-cache
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)    # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        bytes_per_token = d_nope + num_tiles*4 + input_elem_size*d_rope
        result = torch.empty((num_blocks, block_size+1, bytes_per_token), dtype=torch.float8_e4m3fn, device=input_k_cache.device)[:, :block_size, :]
        result_k_nope_part = result[..., :d_nope]
        result_k_scale_factor = result[..., d_nope: d_nope + num_tiles*4].view(torch.float32)
        result_k_rope_part = result[..., d_nope + num_tiles*4:].view(input_k_cache.dtype)
        result_k_rope_part[:] = input_k_cache[..., d_nope:]

        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(dim=-1).values.float() / 448.0 # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

            cur_scale_factors_inv.unsqueeze_(-1)    # [num_blocks, block_size, 1]
            cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
            result_k_nope_part[..., tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_quantized_nope
        
        result = result.view(num_blocks, block_size, 1, -1)
        return result
    
    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        bytes_per_token = d_nope + 2*d_rope + num_tiles + 1
        size_per_block_padded = (block_size*bytes_per_token + 576-1) // 576 * 576
        result = torch.empty((num_blocks, size_per_block_padded), dtype=torch.float8_e4m3fn, device=input_k_cache.device)[:, :block_size*bytes_per_token]
        result_k_nope_rope_part = result[:, :block_size*(d_nope+2*d_rope)].view(num_blocks, block_size, d_nope + 2*d_rope)
        result_k_nope = result_k_nope_rope_part[:, :, :d_nope]  # [num_blocks, block_size, d_nope]
        result_k_rope = result_k_nope_rope_part[:, :, d_nope:].view(input_k_cache.dtype)  # [num_blocks, block_size, d_rope]
        result_k_scale_factor = result[:, block_size*(d_nope+2*d_rope):].view(num_blocks, block_size, 8)[:, :, :7].view(torch.float8_e8m0fnu)   # [num_blocks, block_size, num_tiles]

        result_k_rope[:] = input_k_cache[..., d_nope:]
        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = torch.abs(input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size]).max(dim=-1).values.float() / 448.0 # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv.to(torch.float8_e8m0fnu)

            cur_scale_factors_inv = cur_scale_factors_inv.view(num_blocks, block_size, 1)
            cur_quantized_nope = (input_k_cache[..., tile_idx*tile_size:(tile_idx+1)*tile_size].float() / cur_scale_factors_inv.float()).to(torch.float8_e4m3fn)
            result_k_nope[:, :, tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_quantized_nope
        
        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout.is_nvfp4():
        bytes_per_token = kvcache_layout.bytes_per_token()
        num_nope_sf = d_nope // tile_size    # 32
        sf_nope_off = d_nope // 2 + d_rope   # NoPE is e2m1 (2 values/byte), RoPE is e4m3

        # Over-allocate one extra token row per block (mirroring the V32 layout above) so that
        # any trailing TMA reads stay within valid memory.
        result = torch.zeros((num_blocks, block_size+1, bytes_per_token), dtype=torch.uint8, device=input_k_cache.device)[:, :block_size, :]

        # NoPE: e2m1 with per-16 e4m3 scale factors
        nope_scaled, nope_sf = _quant_tiles_e4m3_sf(input_k_cache[..., :d_nope], tile_size, 6.0)
        result[..., :d_nope//2] = _pack_e2m1(_cast_to_e2m1_codes(nope_scaled))
        result[..., sf_nope_off:sf_nope_off+num_nope_sf] = _nvfp4_permute_sf(nope_sf.view(torch.uint8))

        # RoPE: plain e4m3, no scale factor
        result[..., d_nope//2:sf_nope_off] = input_k_cache[..., d_nope:].to(torch.float8_e4m3fn).view(torch.uint8)

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,    # (num_blocks, block_size, 1, bytes_per_token)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty((num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device)

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

        input_nope = quant_k_cache[..., :d_nope]
        input_scale = quant_k_cache[..., d_nope:d_nope + num_tiles*4].view(torch.float32)
        input_rope = quant_k_cache[..., d_nope + num_tiles*4:].view(torch.bfloat16)
        result[..., d_nope:] = input_rope

        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[..., tile_idx*tile_size:(tile_idx+1)*tile_size].to(torch.float32)
            cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
            result[..., tile_idx*tile_size:(tile_idx+1)*tile_size] = cur_nope * cur_scales

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, -1)  # [num_blocks, ...]  
        input_nope_rope = quant_k_cache[:, :block_size*(d_nope+2*d_rope)].view(num_blocks, block_size, d_nope + 2*d_rope)
        input_nope = input_nope_rope[:, :, :d_nope]
        input_rope = input_nope_rope[:, :, d_nope:].view(torch.bfloat16)
        input_scale = quant_k_cache[:, block_size*(d_nope+2*d_rope):].view(num_blocks, block_size, 8)[:, :, :7].view(torch.float8_e8m0fnu)   # [num_blocks, block_size, num_tiles]

        result[..., d_nope:] = input_rope
        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[..., tile_idx*tile_size:(tile_idx+1)*tile_size].to(torch.bfloat16)
            cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
            result[..., tile_idx*tile_size: (tile_idx+1)*tile_size] = cur_nope * cur_scales

    elif kvcache_layout.is_nvfp4():
        # NOTE This must match the kernel's dequantization bit-for-bit. The kernel multiplies
        # the (exactly-representable) data value with the bf16-converted e4m3 scale factor in
        # bf16; since data*sf has at most 8 mantissa bits, a float32 multiply followed by a
        # bf16 round-trip produces identical bits.
        num_nope_sf = d_nope // tile_size
        sf_nope_off = d_nope // 2 + d_rope

        quant_k_cache = quant_k_cache.view(torch.uint8).view(num_blocks, block_size, -1)
        nope_vals = _e2m1_codes_to_float(_unpack_e2m1(quant_k_cache[..., :d_nope//2]))   # [nb, bs, d_nope] fp32
        nope_sf = _nvfp4_unpermute_sf(
            quant_k_cache[..., sf_nope_off:sf_nope_off+num_nope_sf]).view(torch.float8_e4m3fn).float()
        result[..., :d_nope] = (nope_vals.unflatten(-1, (-1, tile_size)) * nope_sf.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)

        # RoPE: plain e4m3, no scale factor
        result[..., d_nope:] = quant_k_cache[..., d_nope//2:sf_nope_off].view(torch.float8_e4m3fn).to(torch.bfloat16)

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")
    
    result = result.view(num_blocks, block_size, 1, d)
    return result


def abs_indices2indices_in_kvcache(
    abs_indices: torch.Tensor,  # [b, s_q, topk]
    block_table: torch.Tensor,  # [b, /]
    block_size: int,
) -> torch.Tensor:
    """
    Convert abs_indices (logical index, ranging from 0 to s_k-1) to index expected by the sparse attn kernel
    Equivalent to:
    
    b, s_q, topk = abs_indices.shape
    indices_in_kvcache = torch.empty_like(abs_indices)
    for i in range(b):
        cur_abs_indices = abs_indices[i, :, :].clone()  # [s_q, topk]
        invalid_mask = cur_abs_indices == -1
        cur_abs_indices[invalid_mask] = 0
        cur_indices_in_kvcache = block_table[i].index_select(0, cur_abs_indices.flatten()//block_size).view(s_q, topk)*block_size + cur_abs_indices%block_size
        cur_indices_in_kvcache[invalid_mask] = -1
        indices_in_kvcache[i] = cur_indices_in_kvcache
    return indices_in_kvcache

    """
    b, s_q, topk = abs_indices.shape
    _, max_blocks_per_seq = block_table.shape

    abs_indices = abs_indices.clone()
    invalid_mask = abs_indices == -1
    abs_indices[invalid_mask] = 0

    real_block_idxs = block_table.view(-1).index_select(0, (abs_indices//block_size + torch.arange(0, b).view(b, 1, 1)*max_blocks_per_seq).view(-1))
    indices_in_kvcache = real_block_idxs.view(b, s_q, topk)*block_size + abs_indices%block_size
    indices_in_kvcache[invalid_mask] = -1

    return indices_in_kvcache