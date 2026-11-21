# SM120 (sm_120a) performance probes (throwaway)

Probe round that gates the ranked opportunities in
`docs/20261027-sm120-performance-research.md` §6. None of these touch the
kernels; each is a standalone `main()`.

Run on **IOTA, GPU 1** (both cards are sm_120a; GPU 1 is the house default):

```bash
cd docs/20261027-sm120-perf-probes
CUDA_HOME=~/cuda-13.3 ./run_all.sh            # writes results/
CUDA_HOME=~/cuda-13.3 ./run_all.sh probe03    # one probe
```

`run_all.sh` builds with **`-gencode arch=compute_120a,code=sm_120a`**, not
`-arch=sm_120a`: the latter leaves a `compute_120` PTX in the fatbin and ptxas
then rejects every 120a-only instruction. Our library build (`setup.py:187`)
already uses the gencode form.

## Build cells

Some probes carry a second cell that is *expected to fail at ptxas*. The runner
records those too - a failure there is a result, not a broken probe.

| Probe | Cell | Expectation |
|---|---|---|
| 01 | default, uint8, `.shared::cta` | runtime PASS |
| 01 | `-DSUBBYTE`, 16U4, `.shared::cta` | runtime PASS |
| 01 | `-DSUBBYTE -DCLUSTER` | runtime PASS (sub-byte through the cluster form works when the destination is 128 B aligned) |
| 02 | default | runtime PASS (predicated zero-fill + group ordering) |
| 03 | default | runtime PASS (permutation, determinism) + prints the decode table |
| 04 | default | `nofence` arm may fail, `fence` arm must pass, 6000 iterations each |
| 05 | default | SASS must contain `QMMA`/`OMMA`; a `HMMA`-only or dropped atom is a FAIL |
| 06 | default | disassembly must show `BAR.SYNC.*DEFER_BLOCKING` and `SYNCS.PHASECHK.*TRYWAIT` |
| 07 | default (2x `st.shared.v4`) | runtime PASS, aligned and +16 B misaligned arms |
| 07 | `-DSTV8` | expected ptxas FAIL ("global state space expected") |
| 08 | default | aligned prefetch PASS, XOR unswizzle PASS, linear read must differ, misaligned prefetch must fault |
| 09 | default | capture, instantiate and two replays all PASS |
| 10 | default (plain gather4) | runtime PASS |
| 10 | `-DCTA_GROUP` | expected ptxas FAIL (`.cta_group::1` not supported at `sm_120a`) |
| 17 | default | runtime PASS, and multicast must beat the same fetch issued per CTA |
| 18 | default | runtime PASS; sparse must cost the same as two dense k32 per k64 tile, numeric arm must read 64 / 32 |
| 19 | default | runtime PASS; L2-pinned replay must be no slower than the plain one |

## Measured outcomes

Full run on IOTA GPU 1 (RTX PRO 6000, sm_120a, CUDA 13.3): 14/14 cells as
expected - see `RESULTS.md` for the constraint list the run produced. Probe 05
was also rebuilt with CUDA 12.9 (our production toolchain): the atom still
survives to `QMMA`, and 12.9 shows the same barrier lowering.

Compile-only checks were also run on a T400 host: everything assembles there
except the two expected-negative cells.

## Files

`common.h` - error checks, mbarrier/TMA/fence inline PTX, tensor-map builder.
`probeNN_*.cu` - one probe each, numbered as in the research note §6.
`rowNN_*_probe.py` - torch probes for a row's accuracy question (`row18_p_quant_probe.py`, `row19_s24_prune_probe.py`).
`run_all.sh` - build/run/check driver, results in `results/<probe>.<cell>.log`.
