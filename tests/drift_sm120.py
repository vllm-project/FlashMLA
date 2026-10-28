# Bit-exact drift gate for the SM120 perf rows (docs/20261027-sm120-perf-plan.md).
# Runs one launch shape N times on identical inputs and compares every output
# bit against the first run. Any drift is a race, not tolerance: the gate is
# DRIFT=0.
#
#   CUDA_VISIBLE_DEVICES=1 python tests/drift_sm120.py dense   --iters 6000
#   CUDA_VISIBLE_DEVICES=1 python tests/drift_sm120.py sparse  --iters 6000
#   CUDA_VISIBLE_DEVICES=1 python tests/drift_sm120.py prefill --iters 6000
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["dense", "sparse", "prefill"])
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--iters", type=int, default=6000)
args = parser.parse_args()

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(args.device)
torch.cuda.set_device(args.device)

import flash_mla

if args.mode == "dense":
    b, s_q, s_k, h_q, h_kv, d, dv = 128, 1, 16384, 64, 1, 576, 512
    block_size = 64
    q = (torch.randn((b, s_q, h_q, d)) / 10).clamp_(-1, 1)
    cache_seqlens = torch.full((b,), s_k, dtype=torch.int32)
    n_blocks = s_k // block_size
    block_table = torch.arange(b * n_blocks, dtype=torch.int32).view(b, n_blocks)
    blocked_k = (torch.randn((b * n_blocks, block_size, h_kv, d)) / 10).clamp_(-1, 1)
    sched = flash_mla.get_mla_metadata()

    def run():
        return flash_mla.flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, dv, sched[0], sched[1], causal=False)[0]
elif args.mode == "prefill":
    from flash_mla import flash_mla_sparse_fwd

    s_q, s_kv, topk, h_q, d_qk, d_v = 4096, 65536, 1024, 128, 576, 512
    q = torch.randn((s_q, h_q, d_qk))
    kv = torch.randn((s_kv, 1, d_qk))
    indices = torch.randint(s_kv, (s_q, 1, topk), dtype=torch.int32)
    out = torch.empty((s_q, h_q, d_v))

    def run():
        flash_mla_sparse_fwd(q, kv, indices, d_qk ** -0.5, d_v, out=out)
        return out
else:
    import lib
    from lib import RawTestParamForDecode

    t = RawTestParamForDecode(148, 128, 2, 1, 32768, True, 16384,
                              d_qk=576, d_v=512, check_correctness=False,
                              num_runs=0, seed=1234).to_test_param()
    tc = lib.generate_testcase_for_decode(t)
    meta, _ = flash_mla.get_mla_metadata()

    def run():
        return lib.run_flash_mla_decode(t, tc, meta, None)[0]

ref = run().view(torch.int16).clone()
channel_drift = 0
max_delta = 0.0
first = None
for i in range(args.iters - 1):
    out = run()
    bad = out.view(torch.int16) != ref
    n = int(bad.sum())
    if n:
        channel_drift += 1
        delta = (out.float() - ref.float()).abs()
        max_delta = max(max_delta, float(delta.max()))
        if first is None:
            first = i + 1
            print(f"first drift at iteration {first}: {n} elements")
print(f"DRIFT={channel_drift}/{args.iters} iterations, max delta {max_delta:.3e}")
sys.exit(1 if channel_drift else 0)
