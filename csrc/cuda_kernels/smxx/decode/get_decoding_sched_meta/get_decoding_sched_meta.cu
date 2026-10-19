/*
Sparse Attention Decoding — Tile Scheduler Metadata Generation (GPU Architecture-Independent)

Computes the work partitioning metadata for split-KV decoding. KV blocks from all
batch requests are assigned greedily, in request order, to num_sm_parts SM
partitions, producing a DecodingSchedMeta descriptor per partition and a
cumulative split count array (num_splits_ptr). Supports optional per-request
topk_length and extra KV cache (extra_topk or extra_topk_length).

Template parameters:
  HAVE_TOPK_LEN      — Whether per-request topk_length is provided
  HAVE_EXTRA_TOPK    — Whether extra KV cache is enabled
  HAVE_EXTRA_TOPK_LEN — Whether per-request extra_topk_length is provided

Grid: [1, 1, 1]
Block: [32, 1, 1]

I/O: See GetDecodeSchedMetaParams in csrc/params.h
*/
#include "get_decoding_sched_meta.h"

#include <kerutils/kerutils.cuh>

namespace smxx::decode {

// Warp-wide inclusive prefix sum
static __device__ __forceinline__ int warp_inclusive_sum(int value) {
    const int lane_idx = threadIdx.x % 32;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        int other = __shfl_up_sync(uint32_t(-1), value, offset);
        if (lane_idx >= offset) value += other;
    }
    return value;
}

// The sm parts are filled greedily in request order: each part has a budget of
// `payload`, and each piece of a request it takes costs its blocks plus
// fixed_overhead_num_blocks. Place request i at cost_start[i] (the sum of
// num_blocks + fixed_overhead_num_blocks over earlier requests) and its block t
// at cost_start[i] + t. A piece then costs exactly the distance it spans, so a
// part's start follows from the previous one: with end = start + payload in
// request j, the next part resumes j at end - fixed_overhead_num_blocks if end
// lies more than fixed_overhead_num_blocks into j, and starts j afresh
// otherwise. Only this recurrence is serial; the rest runs warp-parallel.
template<bool HAVE_TOPK_LEN, bool HAVE_EXTRA_TOPK, bool HAVE_EXTRA_TOPK_LEN>
__global__ void __launch_bounds__(32, 1, 1)
get_mla_metadata_kernel(__grid_constant__ const GetDecodeSchedMetaParams params) {
    DecodingSchedMeta *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr;
    int *num_splits_ptr = params.num_splits_ptr;
    const int lane_idx = threadIdx.x;
    const int batch_size = params.b;
    const int block_size_topk = params.block_size_topk;
    const int fixed_overhead_num_blocks = params.fixed_overhead_num_blocks;
    const int num_sm_parts = params.num_sm_parts;

    extern __shared__ int shared_mem[];
    int* num_blocks_shared = shared_mem; // [batch_size]
    int* cost_start_shared = num_blocks_shared + batch_size; // [batch_size+1]
    int* num_pieces_shared = cost_start_shared + batch_size + 1; // [batch_size]
    int* part_start_shared = num_pieces_shared + batch_size; // [num_sm_parts+1]
    int* part_req_shared = part_start_shared + num_sm_parts + 1; // [num_sm_parts+1], the request holding the part's start
    int* part_split_shared = part_req_shared + num_sm_parts + 1; // [num_sm_parts], the split index of that request

    // Iterations are independent, so the global loads of different requests overlap
    #pragma unroll 4
    for (int i = lane_idx; i < batch_size; i += 32) {
        int cur_s_k = HAVE_TOPK_LEN ? __ldg(params.topk_length + i) : params.topk;
        if (cur_s_k == 0) cur_s_k = 1;  // Ensure the main loop will never be empty
        if constexpr (HAVE_EXTRA_TOPK) {
            cur_s_k = ku::ceil(cur_s_k, block_size_topk);
            cur_s_k += HAVE_EXTRA_TOPK_LEN ? __ldg(params.extra_topk_length + i) : params.extra_topk;
        }
        num_blocks_shared[i] = (int)((uint32_t)(cur_s_k-1) / (uint32_t)block_size_topk) + 1;
        num_pieces_shared[i] = 1;
    }
    __syncwarp();

    int total_cost = 0;
    for (int base = 0; base < batch_size; base += 32) {
        int i = base + lane_idx;
        int cost = i < batch_size ? num_blocks_shared[i] + fixed_overhead_num_blocks : 0;
        int inclusive = warp_inclusive_sum(cost);
        if (i < batch_size) cost_start_shared[i] = total_cost + inclusive - cost;
        total_cost += __shfl_sync(uint32_t(-1), inclusive, 31);
    }
    if (lane_idx == 0) cost_start_shared[batch_size] = total_cost;
    __syncwarp();

    int num_busy_parts = 0;
    if (lane_idx == 0) {
        const int payload = (int)ku::ceil_div((uint32_t)total_cost, (uint32_t)num_sm_parts) + fixed_overhead_num_blocks;
        // Distance between the starts of consecutive parts inside one request
        const int payload_within_req = payload - fixed_overhead_num_blocks;
        int start = 0, req_idx = 0, split_idx = 0;
        // The line positions where request req_idx and the one after it start
        int req_cost_start = 0, next_req_cost_start = cost_start_shared[1];
        int part_idx = 0;
        while (part_idx < num_sm_parts) {
            int end = start + payload;
            if (end < next_req_cost_start) {
                // This part and possibly the next ones end inside request req_idx
                do {
                    part_start_shared[part_idx] = start;
                    part_req_shared[part_idx] = req_idx;
                    part_split_shared[part_idx] = split_idx;
                    ++part_idx;
                    start += payload_within_req;
                    end += payload_within_req;
                    ++split_idx;
                } while (end < next_req_cost_start && part_idx < num_sm_parts);
                continue;
            }
            part_start_shared[part_idx] = start;
            part_req_shared[part_idx] = req_idx;
            part_split_shared[part_idx] = split_idx;
            ++part_idx;
            if (end >= total_cost) {
                // Everything left fits into this part
                start = total_cost;
                req_idx = batch_size;
                break;
            }
            // `end` lies in a later request: step to the last one starting at or before it
            do {
                ++req_idx;
                req_cost_start = next_req_cost_start;
                next_req_cost_start = cost_start_shared[req_idx + 1];
            } while (next_req_cost_start <= end);
            if (end - req_cost_start > fixed_overhead_num_blocks) {
                split_idx = 1;
                start = end - fixed_overhead_num_blocks;
            } else {
                split_idx = 0;
                start = req_cost_start;
            }
        }
        KU_TRAP_ONLY_DEVICE_ASSERT(req_idx == batch_size);
        num_busy_parts = part_idx;
    }
    num_busy_parts = __shfl_sync(uint32_t(-1), num_busy_parts, 0);
    // The parts after the one that took the rest have nothing to do
    for (int part_idx = num_busy_parts + lane_idx; part_idx <= num_sm_parts; part_idx += 32) {
        part_start_shared[part_idx] = total_cost;
        part_req_shared[part_idx] = batch_size;
        if (part_idx < num_sm_parts) part_split_shared[part_idx] = 0;
    }
    __syncwarp();

    for (int part_idx = lane_idx; part_idx < num_sm_parts; part_idx += 32) {
        int begin_req_idx = part_req_shared[part_idx];
        int begin_block = begin_req_idx < batch_size ? part_start_shared[part_idx] - cost_start_shared[begin_req_idx] : 0;
        int next_req_idx = part_req_shared[part_idx+1];
        int next_block = next_req_idx < batch_size ? part_start_shared[part_idx+1] - cost_start_shared[next_req_idx] : 0;
        DecodingSchedMeta cur_meta;
        cur_meta.begin_req_idx = begin_req_idx;
        cur_meta.begin_block_idx = begin_block;
        cur_meta.begin_split_idx = part_split_shared[part_idx];
        cur_meta.is_first_req_splitted = begin_block != 0;
        cur_meta.end_req_idx = next_block > 0 ? next_req_idx : next_req_idx - 1;
        cur_meta.end_block_idx = next_block > 0 ? next_block : num_blocks_shared[next_req_idx-1];
        cur_meta.is_last_req_splitted = cur_meta.end_block_idx != num_blocks_shared[cur_meta.end_req_idx];
        if (cur_meta.begin_req_idx == cur_meta.end_req_idx) {
            cur_meta.is_first_req_splitted = cur_meta.is_last_req_splitted = cur_meta.is_first_req_splitted || cur_meta.is_last_req_splitted;
        }
        #ifdef KERUTILS_ENABLE_SM100A
            KU_STG_256(tile_scheduler_metadata_ptr+part_idx, &cur_meta, "evict_normal", "evict_normal");
        #else
            tile_scheduler_metadata_ptr[part_idx] = cur_meta;
        #endif
        if (begin_block != 0) {
            // This part holds one more piece of the request it resumes
            atomicAdd(num_pieces_shared + begin_req_idx, 1);
        }
    }
    __syncwarp();

    int num_splits_so_far = 0;
    for (int base = 0; base <= batch_size; base += 32) {
        int i = base + lane_idx;
        int num_pieces = i < batch_size ? num_pieces_shared[i] : 0;
        int inclusive = warp_inclusive_sum(num_pieces);
        if (i <= batch_size) num_splits_ptr[i] = num_splits_so_far + inclusive - num_pieces;
        num_splits_so_far += __shfl_sync(uint32_t(-1), inclusive, 31);
    }
}


void run_get_decoding_sched_meta_kernel(const GetDecodeSchedMetaParams& params) {
    int smem_size = (params.b * 3 + 1 + (params.num_sm_parts + 1) * 3) * sizeof(int);
    auto kernel = 
        params.topk_length != nullptr ?
            (params.extra_topk != 0 ?
                (params.extra_topk_length != nullptr ?
                    get_mla_metadata_kernel<true, true, true> :
                    get_mla_metadata_kernel<true, true, false>
                ) :
                get_mla_metadata_kernel<true, false, false>
            ) :
            (params.extra_topk != 0 ?
                (params.extra_topk_length != nullptr ?
                    get_mla_metadata_kernel<false, true, true> :
                    get_mla_metadata_kernel<false, true, false>
                ) :
                get_mla_metadata_kernel<false, false, false>
            );
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    kernel<<<1, 32, smem_size, params.stream>>>(params);
    KU_CHECK_KERNEL_LAUNCH();
}

}
