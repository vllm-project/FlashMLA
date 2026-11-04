# Feasibility probe for an E4M3 KV plane on the SM120 dense decode kernel
# (docs/20261027-sm120-perf-plan.md row 7). The kernel change is only viable if
# requantizing K to E4M3 at tile granularity keeps the existing accuracy gate,
# so this pushes the *data* through the same quantize -> dequantize the kernel
# would do and runs today's kernel on the result.
#
#   CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python tests/quant_feasibility_sm120.py \
#       [--gran none|token|page|tile] [--part all|nope|rope] [--had]
#
# `none` is the control: the same path with no precision loss, so the E4M3 rows
# can be read as the error the plane would *add* rather than the total.
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import flash_mla
import kernelkit as kk
import test_flash_mla_dense_decoding as dense

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--gran", choices=["none", "token", "page", "tile"], default="tile")
# Which half of the 576-dim K to quantize: the 512 nope dims, the 64 rope dims,
# or both. The rope dims carry the larger magnitudes, so splitting them out says
# whether they are what eats the mantissa.
parser.add_argument("--part", choices=["all", "nope", "rope"], default="all")
# Apply a 64-point Hadamard rotation inside each tile before quantizing. The
# scores are invariant to it (it can be folded into Q), and it evens out the
# per-tile magnitudes so the 3-bit mantissa is spent on signal, not outliers.
parser.add_argument("--had", action="store_true")
# Gate override, to show what an opt-in fp8 mode would have to document.
parser.add_argument("--abs-tol", type=float, default=5e-3)
parser.add_argument("--cos-tol", type=float, default=5e-6)
parser.add_argument("--cases", type=int, default=6)
args = parser.parse_args()

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(args.device)
torch.cuda.set_device(args.device)

E4M3_MAX = 448.0

_h = torch.ones(1, 1)
while _h.shape[0] < 64:
    _h = torch.cat([torch.cat([_h, _h], 1), torch.cat([_h, -_h], 1)], 0)
HAD = (_h / 8.0).to(torch.bfloat16)


def requant(x, gran, part, had):
    """Amax-scale x to E4M3 and back, in place. x: [pages, tokens, h_kv, d].

    The generated K carries NaN padding tokens; they are zeroed in place so the
    scales stay finite."""
    d = x.shape[-1]
    x.nan_to_num_()
    if gran == "none":
        return x
    assert not had or gran == "tile", "the Hadamard probe is tile only"
    lo, hi = {"all": (0, d), "nope": (0, d - 64), "rope": (d - 64, d)}[part]
    v = x[..., lo:hi]
    shape = v.shape
    v = v.reshape(*shape[:-1], shape[-1] // 64, 64)
    if had:
        v = v @ HAD
    if gran == "token":
        amax = v.abs().amax(dim=(-2, -1), keepdim=True)
    elif gran == "page":
        amax = v.abs().amax(dim=(1, 2, 3, 4), keepdim=True)
    else:  # tile: 64 dims, the kernel's QUANT tile
        amax = v.abs().amax(-1, keepdim=True)
    scale = (amax / E4M3_MAX).clamp_min(1e-12)
    q = (v / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    dq = q.to(x.dtype) * scale
    if had:  # HAD is symmetric and orthonormal, so it is its own inverse
        dq = dq @ HAD
    x[..., lo:hi] = dq.reshape(shape)
    return x


cases = [(128, 1, 16384, False), (128, 2, 4096, False), (64, 1, 8192, True),
         (128, 1, 4096, True), (6, 4, 4096, False), (128, 2, 32768, False)][: args.cases]

all_ok = True
for i, (b, s_q, s_k, causal) in enumerate(cases):
    t = dense.TestParam(b, s_q, s_k, True, causal, test_performance=False, h_q=64, h_kv=1)
    cache_seqlens, q, block_table, blocked_k = dense.generate_test_data(t)
    # The reference must see the *unquantized* K, so take the copy first.
    blocked_k.nan_to_num_()
    k_ref = blocked_k.clone()
    kdq = requant(blocked_k, args.gran, args.part, args.had)
    sched = flash_mla.get_mla_metadata()
    out, lse = flash_mla.flash_mla_with_kvcache(
        q, kdq, block_table, cache_seqlens, t.dv, sched[0], sched[1], causal=causal)
    out_ref, lse_ref = dense.reference_torch(cache_seqlens, block_table, q, k_ref, t.dv, causal)
    # tests/test_flash_mla_dense_decoding.py's sm_120a gates unless overridden.
    ok = kk.check_is_allclose("out", out, out_ref, abs_tol=args.abs_tol, rel_tol=2.01 / 128,
                              cos_diff_tol=args.cos_tol)
    ok &= kk.check_is_allclose("lse", lse, lse_ref, abs_tol=5e-2, rel_tol=8.01 / 65536)
    all_ok &= ok
    out_err = (out.float() - out_ref.float()).abs().max().item()
    a, r = out.double(), out_ref.double()
    cos = 1 - 2 * (a * r).sum().item() / (a * a + r * r).sum().item()
    print(f"case {i} b{b} s_q{s_q} s_k{s_k} causal={causal}: {'PASS' if ok else 'FAIL'}, "
          f"out max abs {out_err:.2e}, cos {cos:.2e} "
          f"(gate abs {args.abs_tol:.0e} / cos {args.cos_tol:.0e})")
    del t, cache_seqlens, q, block_table, blocked_k, k_ref, kdq, out, lse, out_ref, lse_ref
    torch.cuda.empty_cache()

print(f"{'PASS' if all_ok else 'FAIL'} gran={args.gran} part={args.part} had={int(args.had)}")
sys.exit(0 if all_ok else 1)
