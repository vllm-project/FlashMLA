# Reproduce the dense decoding case that sits on the tolerance tail (b=64, s_k=4096, h_q=126, h_kv=3)
# and dump the eight worst lse positions. Run from the repository root after building:
#   python tests/repro_dense_h_kv3_lse.py [--device cuda:0]
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to run on")
args = parser.parse_args()

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(args.device)
torch.cuda.set_device(args.device)

from test_flash_mla_dense_decoding import TestParam, generate_test_data, reference_torch

import flash_mla
import kernelkit as kk

t = TestParam(64, 1, 4096, False, True, test_performance=False, have_zero_seqlen_k=False,
              block_size=64, h_q=126, h_kv=3, d=576, dv=512, seed=0)
meta, nsplits = flash_mla.get_mla_metadata()
cache_seqlens, q, block_table, blocked_k = generate_test_data(t)
out_ans, lse_ans = flash_mla.flash_mla_with_kvcache(q, blocked_k, block_table, cache_seqlens, t.dv, meta, nsplits, causal=t.is_causal)
out_ref, lse_ref = reference_torch(cache_seqlens, block_table, q, blocked_k, t.dv, t.is_causal)
ok = True
ok &= kk.check_is_allclose("out", out_ans, out_ref, abs_tol=8e-4, rel_tol=2.01/128, cos_diff_tol=5e-6)
ok &= kk.check_is_allclose("lse", lse_ans, lse_ref, abs_tol=1e-6, rel_tol=8.01/65536)
res = "PASS" if ok else "FAIL"
print("SINGLE_CASE %s" % res, flush=True)

# dig into lse errors
d = (lse_ans - lse_ref).abs()
rel = d / (lse_ref.abs() + 1e-30)
idx = (d > 1e-6) | (rel > 8.01/65536)
print("num bad lse:", idx.sum().item())
bad = idx.nonzero()[:8]
for b_ in bad:
    i, j = b_[0].item(), b_[1].item()
    print(f"  pos [{i},{j}] ans={lse_ans[i,j].item():.8f} ref={lse_ref[i,j].item():.8f} diff={d[i,j].item():.3e} h_q={j}")
