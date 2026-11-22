"""Compare native SM90 small heads with padding to 64, using CUDA graphs.

Run from the repository root:
    PYTHONPATH=. python tests/bench_small_heads_sparse_prefill.py --tokens 1024 8192
Set USE_MMA_FOR_SMALL_N_HEAD=1 to measure the previous mma.sync small-heads
kernel instead of the transposed kernel.
Timings use warm caches; compare ratios within the same run.
"""
import argparse
import statistics

import torch

from flash_mla import flash_mla_sparse_fwd


@torch.inference_mode()
def benchmark(tokens, heads, dim, topk):
    q = torch.randn((tokens, 64, dim), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((8192, 1, dim), device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(8192, (tokens, 1, topk), device="cuda", dtype=torch.int32)
    sink = torch.randn(64, device="cuda", dtype=torch.float32)
    padded_out = torch.empty((tokens, 64, 512), device="cuda", dtype=torch.bfloat16)
    native_out = torch.empty_like(padded_out)[:, :heads]

    def native():
        return flash_mla_sparse_fwd(q[:, :heads], kv, indices, dim ** -0.5,
                                   attn_sink=sink[:heads], out=native_out)

    def padded():
        return flash_mla_sparse_fwd(q, kv, indices, dim ** -0.5,
                                   attn_sink=sink, out=padded_out)

    actual, expected = native(), padded()
    torch.testing.assert_close(actual[0], expected[0][:, :heads], atol=0.02, rtol=0.03)
    for a, b in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(a, b[:, :heads], atol=1e-5, rtol=4e-5)
    graphs = []
    for fn in (native, padded):
        for _ in range(3):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10):
                fn()
        graphs.append(graph)
    samples = [[], []]
    for round_idx in range(6):
        for idx in (0, 1) if round_idx % 2 == 0 else (1, 0):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[idx].replay()
            end.record()
            end.synchronize()
            samples[idx].append(start.elapsed_time(end) / 10)
    native_ms, padded_ms = map(statistics.median, samples)
    print(f"{tokens},{heads},{dim},{topk},{native_ms:.4f},{padded_ms:.4f},{padded_ms/native_ms:.3f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024])
    parser.add_argument("--dims", type=int, nargs="+", default=[512, 576])
    args = parser.parse_args()
    if torch.cuda.get_device_capability()[0] != 9:
        parser.error("requires an SM90 GPU")
    torch.manual_seed(42)
    print("tokens,heads,dim,topk,native_ms,padded_ms,speedup")
    for tokens in args.tokens:
        for dim in args.dims:
            for topk in (128, 640):
                for heads in (8, 16, 32):
                    benchmark(tokens, heads, dim, topk)
