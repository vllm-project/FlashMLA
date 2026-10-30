// Row 16: the producer that fills row 10's XOR plane with `cp.async` instead of
// `tile::gather4`. Measured 1.00x and reverted, but it is the only way the row's
// question was answered (QK keeps 4.1% of row 10's 16%, the rest was the gather
// keeping the copy off the LSU/L1), so the exact code is kept for a re-opening.
//
// How to apply:
//   1. `git apply docs/20261027-sm120-perf-probes/row10_gather4_loader.patch`
//   2. In `csrc/cuda_kernels/sm120/decode/sparse/splitkv_mla.cuh`, replace the
//      block from `// ----- the page: one `tile::gather4` ...` through the
//      zero-fill `make_uint4` loop with the body below.
//   3. Drop the `uint32_t &load_phase` parameter and its declaration, and the
//      three `, load_phase` arguments.
//   4. In the FLASH_MLA_SM120_DBG_K block, drop `slice`/`group`/`rows[]` and
//      read the thread's own token: gmem `gK_base + ch*16`, smem `kplane +
//      token*K_TMA_W + ((ch ^ (token & 7)) << 4)`.
//   5. Drops `row_unit`, `mbar_arrive_expect` and `mbar_wait` from this path;
//      the plane's publication is cp_commit + cp_wait<0> + the block barrier,
//      as in HEAD. The host still encodes `tensor_map_K`; it is unused here.

    // ----- the page: cp.async of 16 B chunks into the XOR-swizzled slices -----
    // The plane is NUM_K_SLICES slices of 64 tokens x 128 B, and the 16 B chunk
    // a write lands in is rotated by (token & 7) - the layout is the one row 10
    // produced with `tile::gather4`, kept here while the producer goes back to
    // cp.async: the gather paid 54% on the copy, the layout is what made QK 16%
    // faster (b8 fragment reads spread over the banks instead of 32 lanes on a
    // 768 B stride). A 16 B chunk never straddles a slice: HEAD_DIM_NOPE is a
    // multiple of K_TMA_W and the rope gets a slice of its own.
    // Invalid tokens take cp.async's zero-fill form (source size 0).
    uint8_t* kplane = plan.u.k.data();
    fp8* gK_nope = gK_base;
    CUTE_UNROLL
    for (int i = 0; i < 8; ++i) {
        const int dim16 = (tg*8 + i)*16;
        if (dim16 >= HEAD_DIM_NOPE) break;
        const int sl = dim16 / K_TMA_W;
        cp_async16((void*)Kernel::k_row_ptr(kplane + sl*K_SLICE_BYTES, token, dim16 % K_TMA_W),
                   gK_nope + dim16, is_valid ? 16 : 0);
    }

    {
        bf16* gK_rope;
        if constexpr (MODEL_TYPE == ModelType::V32) {
            // layout: [512 nope fp8][16B fp32 scales][64 rope bf16]
            gK_rope = (bf16*)(gK_base + HEAD_DIM_NOPE + NUM_SCALES*sizeof(float));
        } else {
            // layout: [448 nope fp8][64 rope bf16], scales are NOT in the row
            gK_rope = (bf16*)(gK_base + HEAD_DIM_NOPE);
        }
        const int dim16 = tg*16;   // 0..48 (64 rope dims, 4 threads x 16 B)
        CUTE_UNROLL
        for (int j = 0; j < 2; ++j)
            cp_async16((void*)Kernel::k_row_ptr(kplane + Kernel::K_ROPE_SLICE*K_SLICE_BYTES,
                                                token, 2*(dim16 + j*8)),
                       gK_rope + dim16 + j*8, is_valid ? 16 : 0);
    }
    // One commit group for the whole block; the block barrier publishes it.
    cp_commit();
    cp_wait<0>();
