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

Consequences for the implementation, to carry into the design step:

- A sub-byte KV page cannot be tightly packed at 576 B per row for TMA: every
  TMA destination must be 128 B aligned, so rows need a 640 B stride (or the
  copy falls back to `cp.async`).
- gather4 boxes must start at 16 B multiples; a 128 B box over a 576 B row is
  fine at offsets 0, 128, ... .
- The proxy fence matched on all 6000 iterations here; that is one schedule on
  one driver, so keep it in the code and keep the 6000-run gate.
