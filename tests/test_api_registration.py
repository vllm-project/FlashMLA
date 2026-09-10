import torch

import flash_mla


def test_stable_extension_registers_all_public_operators():
    expected_ops = {
        "sparse_decode_fwd",
        "dense_decode_fwd",
        "sparse_prefill_fwd",
        "dense_prefill_fwd",
        "dense_prefill_bwd",
        "fused_norm_rope_attn_rope_cast_fwd",
        "fused_norm_rope_attn_rope_cast_decode",
        "permute_q_b_proj",
        "permute_wv_proj",
    }

    missing_ops = sorted(
        name for name in expected_ops if not hasattr(torch.ops._flashmla_C, name)
    )

    assert missing_ops == []
    assert hasattr(flash_mla, "fused_norm_rope_attn_rope_cast")
