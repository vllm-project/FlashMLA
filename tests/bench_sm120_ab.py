# A/B bench for the SM120 rows in docs/20261027-sm120-perf-plan.md.
# Decode modes reuse the official test functions so the numbers come from the
# same path. Prefill has no perf harness upstream, so it times the public API
# with CUDA events on the plan's fixed shape.
#
#   CUDA_VISIBLE_DEVICES=1 python tests/bench_sm120_ab.py dense
#   CUDA_VISIBLE_DEVICES=1 python tests/bench_sm120_ab.py sparse
#   CUDA_VISIBLE_DEVICES=1 python tests/bench_sm120_ab.py prefill
#
# Prints one result line per run; take the median of 5 runs.
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["dense", "sparse", "prefill"])
parser.add_argument("--device", type=str, default="cuda:0",
                    help="index inside the visible set; pin the card with CUDA_VISIBLE_DEVICES")
parser.add_argument("--h_q", type=int, default=0, help="0 = mode default")
parser.add_argument("--reps", type=int, default=5, help="timed repeats inside one prefill run")
args = parser.parse_args()

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device(args.device)
torch.cuda.set_device(args.device)

if args.mode == "prefill":
    # Fixed plan shape: s_q 4096, s_k 65536, topk 1024, h_q 128, d_qk 576 / d_v 512.
    from flash_mla import flash_mla_sparse_fwd

    s_q, s_kv, topk, h_q, d_qk, d_v = 4096, 65536, 1024, args.h_q or 128, 576, 512
    q = torch.randn((s_q, h_q, d_qk))
    kv = torch.randn((s_kv, 1, d_qk))
    indices = torch.randint(s_kv, (s_q, 1, topk), dtype=torch.int32)
    out = torch.empty((s_q, h_q, d_v))
    sm_scale = d_qk ** -0.5

    def run():
        return flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v, out=out)

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.reps):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    flops = 4.0 * s_q * h_q * topk * (d_qk + d_v) / 2.0
    ms = statistics.median(samples)
    print(f"BENCH prefill {sorted(samples)}")
    print(f"{ms:.3f} ms, {flops / ms / 1e9:.0f} TFLOPS, "
          f"{s_q * topk * d_qk * 2 / ms / 1e6:.0f} GB/s")
    sys.exit(0)

if args.mode == "dense":
    import test_flash_mla_dense_decoding as m

    t = m.TestParam(128, 1, 16384, True, False, test_performance=True,
                    h_q=args.h_q or 64, h_kv=1)
else:
    import test_flash_mla_sparse_decoding as m
    from lib import RawTestParamForDecode

    t = RawTestParamForDecode(74 * 2, args.h_q or 128, 2, 1, 32768, True, 16384,
                              d_qk=576, d_v=512, num_runs=10).to_test_param()

print("BENCH", args.mode, t)
m.test_flash_mla(t)
