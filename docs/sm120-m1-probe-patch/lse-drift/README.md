# lse ULP drift investigation (M6 blocker, evening of 2026-09-22)

**Status**: closed; the conclusion is in §12.31 of the plan. The transpose's cross-warpgroup smem hand-off is the sole nondeterminism source, lse is float metadata, and the delivered form keeps both fences.

## Symptom
The full dense decode regression intermittently fails: a single `lse` point exceeds tolerance (abs_tol=1e-6, rel_tol=1.22e-4),
max abs err 6.1e-4 ~ 7.9e-4 at arbitrary (batch, head), while out always passes (cos <= 2.5e-11; out also rises to
~1.2e-4 but stays inside the 8e-4 tolerance). Two full 2352 runs each failed 1 case; for a single case (b64/s_q1/s_k140/
h_q126/h_kv3/d576) the reproduction rate depends on the allocator state.

## Key experiment matrix (all on the 21:22 X build, 186-SM parallel state, num_sm_parts=62)

| Experiment | Data | sched | Allocation before each flash | lse drift |
|---|---|---|---|---|
| lse_repro2 (30 rounds) | fixed objects | reused | none | 0/30 (fixed 9.5e-7 ULP) |
| lse_repro (8 rounds) | generate(seed=0) | new each round | generate allocates | 4/8 |
| lse_repro3 (30 rounds) | generate(seed=0) | reused | generate allocates | 29/30 |
| lse_repro6 phaseB (8 rounds) | fixed + clone | reused | clone's large allocation | 8/8, larger err (1.1e-3) |

Conclusion: the inputs are byte-identical (seed=0 reset) -> the difference is purely inside `flash_mla_with_kvcache`: the intermediate
buffers from new_empty land in **different physical memory** -> stale content is read -> the main hypothesis is
**splitkv fails to write some accumulator slots, and combine / later reads read unwritten slots -> the content drifts with the allocator**.

## Excluded
- PDL/GridDep: a standalone CUDA probe (pdl_probe.cu) ran 50 rounds with 0 errors -> programmatic serialization is reliable on sm120.
- smem races: compute-sanitizer --tool racecheck reports 0 warnings.
- sched meta / num_splits: dumping verified 62 partitions covering batches 0..63 and a self-consistent split structure;
  lse_repro3 (drifts 29/30) and lse_repro2 (0/30) have identical sched.
- combine smem_buf exchange: per-warp private (self-write/self-read + __syncwarp).
- SharedMemoryPlan sM/sL layout: real members, all 64 rows written, and the read (threadIdx) and write (%4==0,
  get_AorC_row_idx 0..63 bijective) mappings are self-consistent.
- Bad-point decoding (translation): single split (batches 32,56,12,44,4,8) and double split (batches 53,45,...) both go bad ->
  not a combine-only path issue; the shared part is the splitkv epilogue's sM.

## In progress (at the time)
- Diagnostic build: `dense_decode.cpp` memsets lse_accum/out_accum before launch (TEMP DIAG).
  - drift disappears -> uninitialized reads confirmed -> locate the unwritten slots by the "fixed error pattern"
  - drift persists -> not uninitialized -> a conservative real race; next step is computing lse directly from register rM in the epilogue
