"""Time BF16 ETAP attention plus split reduction with CUPTI and CUDA graphs."""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from flash_mla import flash_mla_sparse_fwd


def select_splits(splits):
    if splits == 0:
        os.environ.pop("FLASH_MLA_ETAP_SPLITS", None)
    else:
        os.environ["FLASH_MLA_ETAP_SPLITS"] = str(splits)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--heads", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--dims", type=int, nargs="+", default=[512, 576])
    parser.add_argument("--topk", type=int, nargs="+", default=[128, 640, 2176, 8192])
    parser.add_argument(
        "--split-limits",
        "--splits",
        dest="splits",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4, 8, 16],
        help="Partition upper limits: 0 selects automatic dispatch, 1 disables splitting",
    )
    parser.add_argument(
        "--lengths",
        nargs="+",
        choices=["full", "full_tensor", "ramp", "one", "empty"],
        default=["full"],
    )
    parser.add_argument("--kv-layout", choices=["shared", "disjoint"], default="shared")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if 1 not in args.splits or any(s < 0 or s > 32 for s in args.splits):
        parser.error("--splits must include 1 and contain integers in [0, 32]")
    if any(t <= 0 for t in args.tokens) or any(k <= 0 for k in args.topk):
        parser.error("query counts and top-k must be positive")
    props = torch.cuda.get_device_properties(0)
    if props.major != 9:
        parser.error("requires SM90")
    os.environ["USE_MMA_FOR_SMALL_N_HEAD"] = "0"
    torch.manual_seed(42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import flash_mla._flashmla_C as library

    metadata = {
        "gpu": props.name,
        "sms": props.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "arguments": vars(args) | {"output": str(args.output)},
        "library_sha256": hashlib.sha256(
            Path(library.__file__).read_bytes()
        ).hexdigest(),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--ignore-submodules=all", "HEAD"])
        ).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    fields = [
        "tokens",
        "heads",
        "dim",
        "topk",
        "lengths",
        "kv_layout",
        "split_limit",
        "median_us",
        "min_us",
        "max_us",
        "speedup",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for tokens, heads, dim, topk, lengths in itertools.product(
            args.tokens, args.heads, args.dims, args.topk, args.lengths
        ):
            q = torch.randn((tokens, heads, dim), device="cuda", dtype=torch.bfloat16)
            kv_per_query = max(8192, topk)
            kv_tokens = kv_per_query * (tokens if args.kv_layout == "disjoint" else 1)
            kv = torch.randn((kv_tokens, 1, dim), device="cuda", dtype=torch.bfloat16)
            indices = torch.stack(
                [
                    torch.randperm(kv_per_query, device="cuda", dtype=torch.int32)[
                        :topk
                    ]
                    for _ in range(tokens)
                ]
            ).unsqueeze(1)
            if args.kv_layout == "disjoint":
                indices += (
                    torch.arange(tokens, device="cuda", dtype=torch.int32)[
                        :, None, None
                    ]
                    * kv_per_query
                )
            valid = None
            if lengths != "full":
                count = (
                    [min(topk, 1 + t * 64) for t in range(tokens)]
                    if lengths == "ramp"
                    else [topk if lengths == "full_tensor" else int(lengths == "one")]
                    * tokens
                )
                valid = torch.tensor(count, device="cuda", dtype=torch.int32)
                indices.masked_fill_(
                    torch.arange(topk, device="cuda")[None, None]
                    >= valid[:, None, None],
                    -1,
                )
            out = torch.empty((tokens, heads, 512), device="cuda", dtype=torch.bfloat16)

            def run(q=q, kv=kv, indices=indices, dim=dim, valid=valid, out=out):
                return flash_mla_sparse_fwd(
                    q, kv, indices, dim**-0.5, topk_length=valid, out=out
                )

            os.environ["FLASH_MLA_ETAP_SPLITS"] = "1"
            base = tuple(t.clone() for t in run())
            for splits in args.splits:
                select_splits(splits)
                actual = run()
                torch.testing.assert_close(actual[0], base[0], atol=0.02, rtol=0.03)
                for a, b in zip(actual[1:], base[1:]):
                    torch.testing.assert_close(a, b, atol=1e-5, rtol=4e-5)
            timings = {splits: [] for splits in args.splits}
            for round_idx in range(args.rounds):
                order = args.splits if round_idx % 2 == 0 else args.splits[::-1]
                for splits in order:
                    select_splits(splits)
                    samples = bench_gpu_time_with_cupti(
                        run,
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                        dry_run_iters=10,
                        repeat_iters=args.repeats,
                    )
                    timings[splits].extend(ms * 1000 for ms in samples)
            medians = {s: statistics.median(values) for s, values in timings.items()}
            for splits, values in timings.items():
                writer.writerow(
                    {
                        "tokens": tokens,
                        "heads": heads,
                        "dim": dim,
                        "topk": topk,
                        "lengths": lengths,
                        "kv_layout": args.kv_layout,
                        "split_limit": splits,
                        "median_us": medians[splits],
                        "min_us": min(values),
                        "max_us": max(values),
                        "speedup": medians[1] / medians[splits],
                    }
                )
            handle.flush()
            print(
                f"T={tokens} H={heads} D={dim} K={topk} {lengths}: {medians}",
                flush=True,
            )


if __name__ == "__main__":
    main()
