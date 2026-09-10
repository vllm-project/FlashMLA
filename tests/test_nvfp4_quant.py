import torch

import quant


def test_nvfp4_scale_permutation_matches_wire_contract():
    scales = torch.arange(32, dtype=torch.uint8)
    wire_scales = quant._nvfp4_permute_scales(scales)

    for scale_index in range(32):
        wire_index = 8 * (scale_index & 3) + (scale_index >> 2)
        assert wire_scales[wire_index].item() == scale_index

    assert torch.equal(quant._nvfp4_unpermute_scales(wire_scales), scales)


def test_nvfp4_record_is_352_bytes_and_preserves_fp8_rope():
    layout = quant.KVCacheLayout.V32_NVFP4_FP8ROPE
    source = torch.linspace(-1.0, 1.0, 2 * 576, dtype=torch.float32)
    source = source.view(1, 2, 1, 576).to(torch.bfloat16)

    encoded = quant.quantize_k_cache(source, layout)
    decoded = quant.dequantize_k_cache(encoded, layout)

    assert encoded.dtype == torch.uint8
    assert encoded.shape == (1, 2, 1, 352)
    expected_rope = source[..., 512:].to(torch.float8_e4m3fn).to(torch.bfloat16)
    assert torch.equal(decoded[..., 512:], expected_rope)
