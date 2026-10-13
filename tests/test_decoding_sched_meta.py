"""
Check the split-KV schedule plan of sparse decoding attention against a reference.

The reference is a direct Python port of the serial greedy planner that
get_mla_metadata_kernel implements. The plan is read back from the
FlashMLASchedMeta that one decoding call fills in, and must match the
reference in every field.
"""

import random
from typing import List, Optional, Tuple

import torch

import flash_mla

BLOCK_SIZE_N = 64
FIXED_OVERHEAD_NUM_BLOCKS = 5


def ref_plan(cur_s_ks: List[int], num_sm_parts: int) -> Tuple[List[List[int]], List[int]]:
    """Returns (per-part [begin_req, end_req, begin_block, end_block, begin_split, first_splitted, last_splitted], num_splits)."""
    b = len(cur_s_ks)
    num_blocks = [(max(s_k, 1) - 1) // BLOCK_SIZE_N + 1 for s_k in cur_s_ks]
    total = sum(n + FIXED_OVERHEAD_NUM_BLOCKS for n in num_blocks)
    payload = -(-total // num_sm_parts) + FIXED_OVERHEAD_NUM_BLOCKS

    meta = []
    num_splits = [0] * (b + 1)
    now_req, now_block, now_split, cum = 0, 0, 0, 0
    for _ in range(num_sm_parts):
        begin_req, begin_block, begin_split, first_splitted = now_req, now_block, now_split, int(now_block != 0)
        remain = payload
        while now_req < b:
            remain_blocks = num_blocks[now_req] - now_block
            if remain >= remain_blocks + FIXED_OVERHEAD_NUM_BLOCKS:
                cum += now_split + 1
                num_splits[now_req + 1] = cum
                remain -= remain_blocks + FIXED_OVERHEAD_NUM_BLOCKS
                now_req += 1
                now_block = 0
                now_split = 0
            else:
                if remain - FIXED_OVERHEAD_NUM_BLOCKS > 0:
                    now_block += remain - FIXED_OVERHEAD_NUM_BLOCKS
                    now_split += 1
                break
        end_req = now_req if now_block > 0 else now_req - 1
        end_block = now_block if now_block > 0 else num_blocks[now_req - 1]
        last_splitted = int(end_block != num_blocks[end_req])
        if begin_req == end_req:
            first_splitted = last_splitted = int(first_splitted or last_splitted)
        meta.append([begin_req, end_req, begin_block, end_block, begin_split, first_splitted, last_splitted])
    assert now_req == b and now_block == 0 and now_split == 0
    return meta, num_splits


def sparse_plan(b: int, s_q: int, h_q: int, topk: int, topk_length: Optional[List[int]],
                extra_topk: Optional[int], extra_topk_length: Optional[List[int]]):
    # DeepSeek-V4 layout (d_qk 512, 584 bytes per token); all indices are invalid, only the plan matters
    d_qk = 512
    cache = torch.zeros(4, 64, 1, 584, dtype=torch.uint8, device="cuda")
    q = torch.zeros(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device="cuda")
    indices = torch.full((b, s_q, topk), -1, dtype=torch.int32, device="cuda")
    to_tensor = lambda x: None if x is None else torch.tensor(x, dtype=torch.int32, device="cuda")
    sched_meta = flash_mla.get_mla_metadata()[0]
    flash_mla.flash_mla_with_kvcache(
        q, cache, None, None, 512, sched_meta, None, d_qk ** -0.5, False, True, indices,
        extra_k_cache=cache if extra_topk is not None else None,
        extra_indices_in_kvcache=(torch.full((b, s_q, extra_topk), -1, dtype=torch.int32, device="cuda")
                                  if extra_topk is not None else None),
        topk_length=to_tensor(topk_length), extra_topk_length=to_tensor(extra_topk_length))
    return sched_meta


def check(sched_meta, cur_s_ks: List[int]) -> bool:
    meta = sched_meta.tile_scheduler_metadata.cpu().tolist()
    num_splits = sched_meta.num_splits.cpu().tolist()
    ref_meta, ref_num_splits = ref_plan(cur_s_ks, len(meta))
    return num_splits == ref_num_splits and [m[:7] for m in meta] == ref_meta


def main():
    random.seed(0)
    batch_sizes = [1, 2, 3, 8, 16, 31, 32, 33, 64, 74, 100, 128, 257, 1000, 3000]
    num_cases, failed = 0, []
    for _ in range(450):
        b = random.choice(batch_sizes)
        s_q = random.choice([1, 2])
        h_q = random.choice([64, 128])
        topk = random.choice([64, 128, 576, 2048])
        mode = random.choice(["none", "short", "zeros", "mixed"])
        topk_length = None if mode == "none" else [
            random.randint(1, min(80, topk)) if mode == "short" else
            0 if mode == "zeros" else
            min(topk, random.choice([0, 1, 63, 64, 65, random.randint(0, topk), topk]))
            for _ in range(b)
        ]
        extra_topk = random.choice([None, 64, 512])
        extra_topk_length = None
        if extra_topk is not None and random.random() < 0.5:
            extra_topk_length = [random.randint(0, extra_topk) for _ in range(b)]
        sched_meta = sparse_plan(b, s_q, h_q, topk, topk_length, extra_topk, extra_topk_length)
        cur_s_ks = []
        for i in range(b):
            s_k = topk_length[i] if topk_length is not None else topk
            if extra_topk is not None:
                s_k = -(-max(s_k, 1) // BLOCK_SIZE_N) * BLOCK_SIZE_N
                s_k += extra_topk_length[i] if extra_topk_length is not None else extra_topk
            cur_s_ks.append(s_k)
        num_cases += 1
        if not check(sched_meta, cur_s_ks):
            failed.append(("sparse", b, s_q, h_q, topk, mode, extra_topk))
    print(f"{num_cases - len(failed)}/{num_cases} schedule plans match the reference")
    for f in failed:
        print("\tFAILED", f)
    assert not failed


if __name__ == "__main__":
    main()
