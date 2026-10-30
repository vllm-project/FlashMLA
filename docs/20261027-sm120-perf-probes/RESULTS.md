# Probe results (2026-10-27, IOTA GPU 1, RTX PRO 6000, sm_120a)

All 14 cells behaved as expected on CUDA 13.3. The run also fixed the probe
semantics; the constraints below are the actual output.

| Probe | Result | Constraint / note |
|---|---|---|
| 01 default | PASS | plain 2D TMA end to end, mbarrier gated |
| 01 subbyte | PASS | 16U4_ALIGN8B, box in elements (256 = 128 B); **smem destination must be 128 B aligned** - a 64 B offset faults with "misaligned address" |
| 01 subbyte + cluster | PASS | `.shared::cluster` sub-byte works with the aligned destination; ALIGN16B, by contrast, only encodes boxDim=128 (64 B) |
| 02 | PASS | the 4th `cp.async` operand is the **source size in bytes**, not a predicate: passing a predicate copies 1 byte |
| 03 | PASS | `ldmatrix.m16n16.x2.trans.shared.b8` legal and deterministic; decode table printed |
| 04 | PASS | 6000 iterations each with and without `fence.proxy.async.shared::cta`; both arms matched - the fence is a correctness contract the schedule did not expose here, keep it anyway |
| 05 | PASS | `mma.sync.kind::f8f6f4` -> `QMMA.16832.F32.E4M3.E4M3`, also under CUDA 12.9 |
| 06 | PASS | `bar.sync` -> `BAR.SYNC.DEFER_BLOCKING`; `mbarrier.try_wait.parity` -> `SYNCS.PHASECHK.TRANS64.TRYWAIT` under both 12.9 and 13.3 |
| 07 | PASS | 256-bit load needs 32 B alignment (+16 faults, +32 works); `st.shared.v8.b32` does not exist |
| 08 | PASS | 64 B-aligned descriptor + prefetch fine; a descriptor at +8 faults; the XOR un-swizzle matches and a linear read does not |
| 09 | PASS | capture, instantiate and two replays correct |
| 10 | PASS | plain gather4 (no `.cta_group::1`; the token does not assemble at sm_120a). **The x coordinate must be 16 B-granular** - col 4 works, col 2/3 faults. Columns past the row are zero-filled |
| 11 | PASS | the SWIZZLE_128B phase follows the **absolute destination smem address**: box-relative addressing mismatches 512 of 8192 cells, absolute 0 |
| 12 | PASS | a 656 B row stride (not a 128 multiple, scales inside the row) and the real 5-slice destination: 0 mismatches; the swizzled image lands where `k_row_ptr()` reads it |
| 13 | PASS | cache 0x7f (e4m3 NaN) dequantizes to NaN, so a cache-only reference must `nan_to_num` - 0 of 64 attended rows are NaN, ~88% of all rows are |
| 14 | PASS | `.3d` is **rejected** by ptxas on `tile::gather4` (2D only); a rank-3 map encodes and the `{x, y0..y3, z}` order is the one ptxas takes |
| 15 | FAIL | a byte-offset y axis (dim1 stride 1) is refused by `cuTensorMapEncodeTiled` with `CUDA_ERROR_INVALID_VALUE`; the y stride must be a multiple of 16, so the row axis is a token index |

Row-10 consequences (the row itself was measured and reverted, see the plan's
"Row 10 result"): a V4 map row is the 576 B data region, because the scales
live in a per-block tail and the block padding to a 576 multiple *is* that
tail; a flat token axis is off by one row per block, so the coordinate is
`block*(block_bytes/row_bytes) + rel`; and each of a group's four gathered
tokens needs its own block and rel. The engine's per-op price, not its issue
count, is what made the row 8% slower.

Row 16 then kept the plane and put `cp.async` back (1.00x, also reverted) and
answered the attribution question: only 4.1% of row 10's 16% QK gain is the
layout, the rest is the gather keeping the copy off the LSU/L1 - the QK phase
re-reads Q from gmem and feels it. The producer is archived verbatim in
`row16_cpasync_into_kslice.snippet.cuh`.

Consequences for the implementation, to carry into the design step:

- A sub-byte KV page cannot be tightly packed at 576 B per row for TMA: every
  TMA destination must be 128 B aligned, so rows need a 640 B stride (or the
  copy falls back to `cp.async`).
- gather4 boxes must start at 16 B multiples; a 128 B box over a 576 B row is
  fine at offsets 0, 128, ... .
- The proxy fence matched on all 6000 iterations here; that is one schedule on
  one driver, so keep it in the code and keep the 6000-run gate.

Row 18 (PV on the fp8 atom, S re-quantized per 128-dim tile) closed negative
at 0.995x / 0.984x, four pairs of five lost in each order, and it re-prices a
phase reading rather than adding a producer: PV's inner loop halves its
instructions, spills disappear (187 registers, `LOCAL:0`, from 255 with a
spill) and PV's cycles do not move - 15.8-16.9k/page against base's
15.8-16.2k, with the softmax's 4x S stores and a 16 KB/page smem round-trip
for the A operand as the visible new costs. The loop is `LDS`/`LDSM`- and
mma-bound, not issue-bound. The exact fp8 B-fragment lane map it produced
(`v0 + 4*v1 + 8*(dwin/8 + q%2) + 64*(twin/32)`, k extent = the 64-token block)
is reusable for any later PV work.

The patch is `row18_pv_fp8.patch`, the accuracy probe (real cache bytes, real
S, unnormalized kernel-shaped values with a fixed power-of-two P scale) is
`row18_p_quant_probe.py`. It measures abs 1.4e-3 / cos 5e-7 against the fp8
gates of 3e-2 / 1e-2, so accuracy was never the blocker here.
