# SM120 performance: remaining headroom and the legal instruction set (research note)

- **Date**: 2026-10-27
- **Status**: research complete, probe round executed (all 14 cells as expected), nothing implemented
- **Scope**: what a latency-bound kernel can still do on `sm_120a`, given the M5/M6 delivery state. Corrects two premises recorded in the plan: "sm_120a has no TMA" and "the mbarrier family is lowered to nothing".

## 1. Summary

The plan's ceiling analysis (§12.15/§12.17/§12.32) rests on "no TMA, no precise synchronization". Both are too pessimistic for `sm_120a`:

- `sm_120a` **does** support TMA (`cp.async.bulk.tensor.*.shared::cta`) and `cp.async`; CUTLASS ships sm120 GEMM mainloops built on TMA plus an **mbarrier-based multi-stage pipeline** (`PipelineTmaAsync`), and the sm120 builder selects **ldmatrix** smem->register copy atoms. That is NVIDIA's own blessed dataflow for this architecture.
- The 256-bit `ld.global.L1::no_allocate` load and `ldmatrix/stmatrix` (incl. `ldmatrix...b8.trans`) are enabled for `sm_120a`.
- `mma.sync.kind::f8f6f4` (m16n8k32) and the MX block-scaled variants exist for sm120, but there is **no bf16 x fp8 mixed atom**. That does not block fp8-resident smem: the register fragment can be dequantized to bf16 before a bf16 `mma.sync` (what SGLang does), so fp8 MMA is an optional extra on top, not a prerequisite.

Our kernels today copy with synchronous LDG+STS into a 72-stride, unswizzled smem layout and read operands via `cute::copy` (LDS). The biggest remaining costs - the in-page transpose (30%), the B-read bank conflicts inside both mma phases, and the exposed page copy - are all addressable with the primitives above.

This is not a research guess. SGLang's SM120 FA4 (#45871) keeps V resident as E4M3 in smem with fragment-level dequant and `ldmatrix...trans` and measures 2.03x decode on this chip class; CUTLASS PR #3030 carries an SM120 FlashAttention v2 with both a `cp.async` and a TMA/`PipelineTmaAsync` variant; the SM120 NVFP4 TMA mainloop reaches 279.6 TFLOP/s in QuACK at ~95% of the reference example. The same record calibrates the expectations down: TMA is 0.69-0.83x of `cp.async` at short sequences, FP8 is 0.81-0.88x of BF16. §4 collects it, including the three descriptor and proxy hazards it cost them.

## 2. Verified capability matrix (local evidence)

All items below were read in the pinned CUTLASS tree (`csrc/cutlass/`, referenced by `setup.py`) or measured with `ptxas`/`nvdisasm` at `-arch=sm_120a` (CUDA 13.3), not recalled from the docs.

| Capability | Verdict for `sm_120a` | Evidence |
|---|---|---|
| TMA tensor load, `.shared::cta` form | supported; CUTLASS has an sm120-specific asm branch (**no cluster**) | `cute/arch/copy_sm90_tma.hpp:59-75, 115-131` (SM120 branch drops `.shared::cluster`); `cute/arch/config.hpp:156` defines `CUTE_ARCH_TMA_SM120_ENABLED` |
| TMA completion via mbarrier / multi-stage pipeline | supported; official sm120 mainloop | `cutlass/gemm/collective/sm120_mma_tma.hpp:118` (`PipelineTmaAsync<Stages>`), same in all 7 sm120 collectives |
| `cp.async` (Ampere-style) | enabled for every arch >= 800; ptxas accepts it at `sm_120a` | `cute/arch/copy_sm80.hpp:38-40`; ptxas probe rc=0 |
| `ldmatrix` / `stmatrix`, incl. b8 `.trans` | enabled; ptxas accepts `ldmatrix.m16n16.x2.trans.shared.b8` (must be `.trans`, 4 destination registers) and `stmatrix.m16n8.x4.trans.shared.b8`; `m8n8.b8` is illegal | `cute/arch/config.hpp:131-132, 185-186`; `cute/arch/copy_sm100.hpp:107,138`; ptxas probe rc=0 |
| smem->register operand atoms chosen by NVIDIA's sm120 builder | ldmatrix (`SM75_U32x4_LDSM_N`, `SM100_SU4/SU6_DU8x16_x4_LDSM_N`) | `cutlass/gemm/collective/builders/sm120_common.inl:58-93` |
| 256-bit global load / store | enabled with CUDA >= 12.9 (PTX ISA 8.8, "requires sm_100 or higher"); no sm120 kernel in the tree uses it yet | `cute/arch/copy_sm100.hpp:50` (`ld.global.L1::no_allocate.v8.f32`), gated at `cute/arch/config.hpp:189-198`; usage only in `sm100_epilogue_nosmem.hpp:727,772` |
| TMA with sub-byte element types | works through both state spaces, but only with a 128 B-aligned smem destination; `16U4_ALIGN16B` encodes just one box width (128 elements = 64 B), so ALIGN8B is the usable mode | runtime probe `docs/20261027-sm120-perf-probes/probe01` |
| cluster > 1 / `multicast::cluster` | compiles, but NVIDIA's sm120 builder hard-asserts `cluster == 1` and no sm120 kernel ships multicast | `cutlass/gemm/collective/builders/sm120_mma_builder.inl:86`; docs `media/docs/cpp/blackwell_functionality.md:668-688` |
| precise arrival ordering | `bar.sync` lowers to `BAR.SYNC.DEFER_BLOCKING` (imprecise, the §12.26 finding); `mbarrier.try_wait.parity` is precise and works | nvdisasm probe; `mbarrier.try_wait.parity.shared::cta.b64` rc=0 |
| fp8/fp4 `mma.sync.kind::f8f6f4` m16n8k32 + MX block-scale variants | atoms exist for sm120 | `cute/arch/mma_sm120.hpp` (`SM120_16x8x32_TN*`, `SM120_*_VS`), `mma_sm120_sparse.hpp` |
| bf16/f16 x fp8 mixed MMA atom | **absent** (fp8-family operand pairs only) | `mma_sm120.hpp` struct list |
| 2 CTA/SM | blocked at our shape: 2 x 43 K registers > 64 K per SM (`rO` = 128 registers/thread). Reachable only after the tile and register budget shrink - FA2 needed a 48 KB tile | §12.32 |
| smem/SM | 101376 B | `cutlass/arch/arch.h:45` (`sm120_smem_capacity_bytes`) |

External confirmation of the same verdicts (details in §4): the PTX ISA lists `cp.async.bulk` / `cp.async.bulk.tensor` as `sm_90 or higher` and the tensor `gather4` form as explicitly "supported on `sm_120a`"; DeepGEMM ships SM120 TMA kernels; CUTLASS's open PR #3030 carries an SM120 FlashAttention v2 with a `cp.async` and a TMA/`PipelineTmaAsync` variant. The instruction set is not the constraint.

Two premises in the plan need correction on this basis:

1. §12.15/§12.28 "a bigger step would need inter-page overlap, but ... double buffering is infeasible" was derived under the synchronous-copy assumption. With TMA the copy no longer occupies register-file warps and the exposure becomes a pipeline question, not an smem-only one.
2. The §12.26 note "the whole mbarrier family is downgraded to nothing by ptxas" is half right. Measured with nvdisasm: `bar.sync` does become `BAR.SYNC.DEFER_BLOCKING`, but `mbarrier.try_wait.parity` compiles and stays precise, which is why `PipelineTmaAsync`'s arrive/wait is usable. The imprecision belongs to `bar.sync`, not to the mbarrier family.

## 3. What our kernels do today vs the sm120 dataflow

| Stage | SM120 kernels today | What the architecture offers |
|---|---|---|
| page copy | 16B `ld.global` (with L2 prefetch hint) -> `st.shared`, synchronous, warps blocked | `cp.async` (no register round-trip), TMA (no warp involvement, mbarrier completion, swizzle on store) |
| smem layout | `(SEG, TOKEN, COL64)` with `K_ROW_PAD=72` to dodge 8-way conflicts | TMA-written swizzled tiles (SW64/SW128) remove the conflict class and the padding |
| KV storage | fp8/fp4 dequantized to bf16 during the copy (sparse/fused) | raw fp8/fp4 in smem + ldmatrix b8 + dequant on read (what the SM100 kernels do), halves smem and enables stages |
| K -> V transpose | in-place scalar RMW, 18.3k clocks (~30% of a page), the only ULP-drift source | `ldmatrix...trans` renders the transpose implicit in the PV operand read, or a dim-major TMA view makes it unnecessary |
| operand reads | `cute::copy` = LDS + PRMT | ldmatrix (vectorized, conflict-free with swizzle) |
| pipeline | single buffer, copy -> QK -> transpose -> PV strictly serial | `PipelineTmaAsync`-style multi-stage |

## 4. External corroboration

The public record is consistent with §2 and adds runtime evidence: on SM120 these primitives work in shipped code, not only in headers.

**Legality, from NVIDIA's own ISA text.** The PTX ISA gives `cp.async.bulk` and `cp.async.bulk.tensor` as "sm_90 or higher"; the `.tile::gather4` / `.tile::scatter4` forms as "sm_100a or higher" and additionally "supported on `sm_120a`" (parallel-thread-execution guide, §cp.async.bulk.tensor). So `-arch=sm_120a` is a legal target for TMA including gather4 - the qualification the SM100 dequant path needs.

**TMA on SM120 is real in production code, with a caught-bug tail.**

- CUTLASS `[CuTeDSL] Add SM120 MXF4/NVFP4 native-TMA path` (PR #3273, merged; #3280 ptr-array variant, #3239 narrow warp path) - SM120 uses "native A/B TMA, native FP8 scale TMA, register A/B fragments, explicit FP8 scale fragments, and warp-level MXF4/NVFP4 MMA ... rather than force SM120 through the SM100 tcgen05/TMEM model".
- QuACK built on it (#145) reports 279.6 TFLOP/s, ~95% of the CUTLASS example - the TMA path reaches near-CUTLASS speed, not just correctness.
- DeepGEMM ships SM120 kernels that build host `make_tma_2d_desc` descriptors, issue `tma::copy`, and patch tensormaps (`tensor_map_replace_global_addr`) in `sm120_fp8_fp4_gemm_1d1d.cuh`.
- Its bug tail is exactly the risk list for us: descriptor allocated outside the 64 B alignment (`prefetch.tensormap` -> misaligned address on SM120, CUTLASS #2905); read-side un-swizzle mode hardcoded while descriptors swizzle per head-dim -> silently wrong, run-to-run nondeterministic results (DeepGEMM #433, 189/384 configs); group switch publishing new tensormaps without draining in-flight TMA -> race (DeepGEMM #430). All three are nondeterminism, i.e. our `out_drift == 0` gate is the right instrument.
- CuTe DSL's executable rank-2 TMA `cute.copy` lowering compiles and launches but then hangs on the mbarrier wait on SM120/SM120a (CUTLASS #3189). The fix there routed around the DSL lowering; a hand-rolled C++ asm path (what we would write) is not affected, but the probe in §6.1 must run at runtime, not stop at compile.

**Attention kernels: the community already ran our two designs.**

- CUTLASS PR #3030 ("Flash Attention v2 for SM120", open, 35 comments) carries both variants side by side: a `cp.async` kernel with all threads loading and computing, "tuned for SM120's 101 KB shared memory", and a TMA kernel with one dedicated DMA warp plus 3 MMA warps over a multi-stage KV pipeline on `PipelineTmaAsync` mbarriers. It uses `mma.sync.aligned.m16n8k16` because "SM120 lacks tcgen05 MMA". The author's own numbers (SM121a, CUDA 13.0, cutlass-dsl 4.5.2, all 16 configs passing): TMA is **0.69-0.83x** of the `cp.async` kernel at B=1 S=512 and only reaches **1.25x** at S=1024; the FP8 TMA kernel is 0.81-0.88x of BF16. So TMA is not a free win at short sequence, and fp8 MMA pays a B-operand layout-bridge tax of ~16 shuffles per `ldmatrix` in the PV path.
- Note one claim that conflicts with §2: #3030 writes "no `ldmatrix.b8` on SM120" and shuffles the 4x4 byte transpose with `prmt.b32` instead. That is stale. ptxas accepts `ldmatrix.m16n16.x2.trans.shared.b8` at `sm_120a`, SGLang's SM120 FA4 (#45871) uses `LdMatrix16x16x8bOp(transpose=True)` taken from quack's `GemmSm120`, and CUTLASS's own fp4/fp6 builder routes through `SM100_SU4/SU6_DU8x16_x4_LDSM_N`. Probe 3 in §6 is a formality by now.

**The closest published analogue to our kernel.** SGLang #45871 makes the FA4 SM120 path store V resident as E4M3 in smem (no bf16 copy at all, smem halved), dequantizes at fragment level, pre-multiplies P by 256 to dodge E4M3 subnormals, and selects `num_splits` for ~2k resident CTAs. Measured: decode HD256 2.03x, HD512 1.85x, end-to-end 20 -> 46 concurrency and 383 -> 864 tok/s. That is our #1 + #3 + #8 package, already run on this exact class of chip.

**The occupancy picture, measured.** FlashInfer #4990 publishes the ncu profile of an SM120 attention/MoE kernel shaped like ours: 168 regs, 89 KB smem, 384 threads, 1 CTA/SM, 22.9% active warps, 87.7% "no eligible" - neither bandwidth- nor compute-bound. That is our §12.15 diagnosis reproduced by a third party on a different kernel, and it is also why dragging occupancy up is only worth 1.05-1.086x even when FA2 cuts its tile to 48 KB to reach 2 CTA/SM.

**Budget tricks from NVIDIA's own sparse builder.** Because 128x128 leaves no room for two stages, the sm120 sparse builder half-splits the K tile, runs 3 stages as 1.5 buffers, and gives A and B independent pipelines (`sm120_sparse_mma_builder.inl:61-115`; two `PipelineTmaAsync` objects in `sm120_sparse_mma_tma.hpp:140-152`). This is the published answer to "no smem for stages", and it is directly applicable at our 64-token page.

**Two hazards the record paid for.** DeepGEMM carries 14 `fence_view_async_shared()` patches (generic-proxy WAR, `sm120_fp8_paged_mqa_logits.cuh:263,360` and elsewhere); without the fence it failed 2 runs in 6000, with it 0 in 6000. And its SM120 build hard-`#error`s below CUDA 13 because pre-13 ptxas silently drops `block_scale` (asserted by grepping the SASS for `QMMA.SF.16832` / `OMMA.SF.16864`). Our plain `f8f6f4` atom uses no block scale, but our toolchain is 12.9, so the SASS check belongs in the probe round (5) before #9 is attempted.

**Where TMA does not apply.** The sparse MQA-logits variant in DeepGEMM drops TMA for `cp.async.cg` + `commit_group` (`sm120_fp8_fp4_sparse_mqa_logits.cuh:247,251,280`) because the gather defeats a tensor map. Our sparse decode indexes whole 64-token pages, so a per-page 1D/2D TMA is still possible, but the sparse path must be measured separately rather than assumed.

**Concurrency and the field contract.**

- vLLM's SM12x sparse-MLA work shows what single-CTA collapse costs and what fixing it buys: the Triton `_fused_gather_dequant_attn_kernel` degenerated to one CTA per decode call (213 us/call at topk=2048, H=8), and a split-K variant at 32 then took single-stream TPOT from 44.9 to 23.84 ms (PR #54929). Relevant to us only where our splitkv grid under-fills, but it confirms the latency-bound diagnosis and that concurrency, not FLOPs, is the lever.
- The same class of hardware measured end-to-end puts decode at ~1 tok/s/request under eager execution with no graphs (vLLM #56892), and gives the working KV contract: per-64-token blocks, `[64 tok x 128 B fp8 | 64 tok x 4 B fp32 scale]`, block-appended, not per-token interleaved (vLLM #56700). That is exactly the raw-fp8 smem source layout in §5.1.
- Floor worth remembering: a `page_size=1` KV layout sustains only ~14 GB/s in production regardless of kernel quality - at that granularity the gather, not the copy, is the ceiling.

**Target naming.** CUTLASS gates the SM120A family on `__CUDA_ARCH_FEAT_SM120_ALL` (`cutlass/arch/config.h:158-166`), so plain `-arch=sm_120` silently loses all of the above; `sm_120a` is required and already what we build. `sm_120f` (CUDA >= 12.9, `CUDA_ARCH_FAMILY(1200)`) is the portable-family spelling, not needed for a single-device kernel family.

What the record does *not* contain: any SM120 kernel that is bandwidth- or MMA-bound, or any shipped attention kernel that beats its 1 CTA/SM sibling by more than ~1.09x through occupancy alone. The ceiling reading in §8 survives, with the one addition that 2 CTA/SM is reachable if the tile shrinks - and still not worth much on its own.

## 5. Ranked opportunities

Ordered by expected gain per unit of risk. Every item keeps the two hard gates used throughout M5/M6: full regression plus `out_drift == 0`.

**A is a package, not a step.** #1-#3 interlock: without fp8-resident smem there is no room for stages, without stages the copy stays exposed, and without ldmatrix.trans the transpose eats what the stages saved. Landing any one alone pays mostly in complexity.

| # | Change | Expected effect | Cost / risk | Comments |
|---|---|---|---|---|
| 1 | **KV resident in smem as raw fp8/fp4 + fragment-level dequant** | 36 KB/page instead of 72 -> double buffering fits in 99 KB; dequant leaves the copy path | M / medium | SGLang's SM120 FA4 (#45871) is this exact move on this exact chip: V resident E4M3, no bf16 copy, fragment dequant, P pre-multiplied by 256 against subnormals - 2.03x decode. Native for sparse/fused (KV already fp8/fp4 on the wire, per-64-token block, `[128 B data | 4 B fp32 scale]`, vLLM #56700). Dense is a **precision decision**: bf16 KV would be requantized to E4M3 in smem and needs the accuracy gate, not just the drift gate |
| 2 | **`cp.async` multi-stage copy + full/empty mbarrier** | hides the copy instead of serializing it; the producer half of #1's stages | S-M / low | kerutils ships `cp_async_cacheglobal` (`device/cuda/sm80/intrinsics.cuh:10`), predication covers the sentinel zero-fill; DeepGEMM's sparse path uses exactly `cp.async.cg` + `commit_group` because gather defeats a tensor map (`sm120_fp8_fp4_sparse_mqa_logits.cuh:247`) |
| 3 | **`ldmatrix...b8.trans` PV reads, delete the in-place transpose** | removes 30% of the page and the only ULP-drift source | M-L / medium | validated twice over: ptxas accepts `m16n16.x2.trans.b8`, SGLang runs `LdMatrix16x16x8bOp(transpose=True)` from quack. Plan B failed on the 72-stride layout, not on ldmatrix; §12.35's objection dies with a compact layout. Note `m16n16` must be `.trans` with 4 destination regs, `m8n8.b8` is illegal |
| 4 | **warp specialization: 128 producer / 256 math, `setmaxnreg` 40/232** | producer warps run the copy while math warps hold the mma pipeline; no register cliff | M / medium | DeepGEMM's SM120 decode is 384 threads with 40/232 and `__launch_bounds__(...,1)`; CUTLASS's sm120 GEMM uses 8 MMA + 2 load + 1 epilogue-load + 1 scheduler warp. Our 168 regs survive this split better than a naive CTA-wide bump |
| 5 | **TMA `.shared::cta` + `PipelineTmaAsync`** (dense first) | copy leaves the warp path entirely; swizzled tiles kill the B-read bank conflicts | L / medium-high | host-side `CUtensorMap` per call (pattern `legacy/sm100/dequant_utils.cuh:169`), 64 B-aligned, and the sparse path keeps cp.async. #3030's own data: 0.69-0.83x at 1-2 tiles, 1.25x at 16 - our ~216-page loop is the favorable regime. Multicast and cluster > 1 stay off (builder asserts `cluster == 1`) |
| 6 | **1.5-buffer, half-K, asymmetric A/B staging** | fits 3 stages where 2 full ones do not | M / medium | NVIDIA's own sm120 sparse builder does this at 128x128 (`sm120_sparse_mma_builder.inl:61-115`, two `PipelineTmaAsync`) - the published answer to the 99 KB budget |
| 7 | **256-bit `ld.global.L1::no_allocate` + Q prefetch** | fewer copy instructions, streaming KV stops evicting Q | S / low | 32 B alignment is mandatory; no sm120 kernel in CUTLASS uses it yet, so it is unverified on this part even though it compiles |
| 8 | **Concurrency floor check** (split-K / `sm_parts`, per launch shape) | vLLM measured 44.9 -> 23.84 ms TPOT from split-K=32 | S / low | our decode already splits over heads and `sm_parts`, so expect less; a one-off occupancy check, not a redesign |
| 9 | fp8 `mma.sync.kind::f8f6f4` QK | doubles QK math throughput | M-L / medium | needs fp8 Q and per-32-group scaling; #3030 measured FP8 *slower* than BF16 (0.81-0.88x) because of the ~16-shuffle B bridge. Our ptxas is 12.9 - verify the atom survives to SASS first (pre-13 ptxas silently drops `block_scale`, DeepGEMM hard-`#error`s below CUDA 13) |

Still closed: tcgen05/TMEM and wgmma (M1, both re-confirmed unavailable), `f32x2`, multicast/cluster > 1, non-tensor `SM90_BULK_COPY_G2S` (never specialized for sm120), PDL (CUTLASS disables it for the whole sm120 family).

2 CTA/SM: no longer listed as impossible - FA2 reaches it with a 48 KB tile - but the measured prize is 1.05-1.086x, and our 64-token page cannot shrink to 48 KB without #1 first. The FlashInfer ncu (168 regs, 22.9% warps, 87.7% no-eligible) says occupancy is a symptom, not the fix.

## 6. Probe round before implementation

Small probes in `docs/20261027-sm120-perf-probes/`, one per item, built with `-gencode arch=compute_120a,code=sm_120a`. All ten ran on IOTA GPU 1 and behaved as expected; the constraint list they produced is in that directory's `RESULTS.md`.

1. TMA 2D `cp.async.bulk.tensor.2d.shared::cta` end to end at run time (not just compile): data lands correctly, mbarrier `expect_tx`/`wait.parity` fires. Sub-byte through both state spaces needs a 128 B-aligned destination.
2. `cp.async` + commit/wait groups, including the predicated zero-fill form.
3. `ldmatrix...m16n16.x2.trans.shared.b8` on a compact fp8 layout: correctness of the decoded fragment table (legality is already known - this is about our table, not the instruction).
4. The generic-proxy WAR pair: TMA (async proxy) writes the tile, then `ldmatrix` (generic proxy) reads it, with and without `fence.proxy.async.shared::cta`. Reproduce DeepGEMM's 2-in-6000 failure mode as a run-to-run drift, with 6000 iterations.
5. Does our **CUDA 12.9** ptxas keep `mma.sync.kind::f8f6f4` in SASS (`QMMA`/`OMMA` strings, the check DeepGEMM automates), or silently drop it like pre-13 `block_scale`?
6. `bar.sync` -> `BAR.SYNC.DEFER_BLOCKING` versus `mbarrier.try_wait.parity` under nvdisasm, on our own barrier shapes, so §12.26 gets a precise replacement.
7. `ld.global.L1::no_allocate.v8.f32` load and the `st.shared` 256-bit counterpart, with 32 B alignment checked on every address.
8. The three descriptor hazards, on purpose: a descriptor in a 64 B-aligned struct (CUTLASS #2905), `prefetch.tensormap` on it, and a read-side un-swizzle that must match the descriptor's swizzle for every head-dim we ship (DeepGEMM #433).
9. Launch the TMA probe under CUDA graph capture and a capture replay - the SM120 field reports are full of graph-capture failures around these ops (vLLM #56700/#56892).
10. `tile::gather4` assembles at sm_120a **without** the `.cta_group::1` token our SM100 intrinsics carry; ptxas 13.3 rejects the token outright ("not supported on .target 'sm_120a'"). The plain form runs, and its x coordinate must be a 16 B multiple (col 4 works, col 2/3 faults); columns past the row are zero-filled.

## 7. Risks carried over from M5/M6

- Vector write-back plus DEFER-barrier semantics drifts (§12.27) - ldmatrix/TMA paths must take `out_drift == 0` as a gate, not racecheck.
- distutils does not track header mtimes; api-layer changes need `rm -rf build` (§12.30).
- Non-ASCII comments break ptxas under `--source-in-ptx`; keep kernel comments ASCII.
- TMA needs a host-side tensor map per call and a runtime null-check; the stable-ABI entry points must build it before launch (pattern: `legacy/sm100/dequant_utils.cuh:169`).
- Descriptor lifetime and alignment: the map must outlive the kernel and sit 64 B-aligned, or `prefetch.tensormap` faults on SM120 (#2905). Build it per call into a small aligned staging struct, not on the stack of a lambda.
- Swizzle agreement is a correctness contract, not a perf knob: descriptor swizzle, the TMA store into smem, and the read-side un-swizzle are three sites that must be derived from one constant. A mismatch is not a crash, it is silent garbage that changes run to run (#433).
- A replaced descriptor must not race in-flight loads: drain before publishing a new tensormap on a group switch (#430).
- `-arch=sm_120` alone compiles none of this; the SM120A gate needs `sm_120a` (`cutlass/arch/config.h:158-166`). Keep the existing gencode.
- **Async-proxy WAR fence.** A tile written by TMA lives in the async proxy; `ldmatrix`/LDS read through the generic one. Without `fence.proxy.async.shared::cta` between them the read is a race that shows up ~2 in 6000 runs (DeepGEMM needed 14 such patches). The `out_drift == 0` gate only catches it if the run count is high enough - use their 6000-iteration shape, not 30.
- **No sm120 bf16 tensor-core atom.** `rr_op_selector_sm120` returns `SM120_16x8x32_TN` unconditionally and static-asserts on bf16; bf16 stays on the SM80 `m16n8k16` atom (`mma_traits_sm120.hpp:36`). Do not wait for a bf16 atom that will not come - item #9 is fp8 Q only.
- **Sub-byte TMA packing.** A sub-byte TMA destination must sit on a 128 B boundary (a 64 B offset faults), so a 576 B page row cannot be tightly packed - it needs a 640 B stride, or the sub-byte path stays on `cp.async`. `16U4_ALIGN16B` also encodes only one box width (64 B).
- **gather4 coordinates.** The x coordinate must be a 16 B multiple; anything else is an illegal instruction at run time, not a compile error. Rows past the row end are zero-filled by OOB handling.
- **`ldmatrix.b8` shape rules.** `m16n16` requires `.trans` and exactly 4 destination registers; `m8n8.b8` does not exist. Wrong forms are ptxas errors, not silent wrong results - cheap, but they cap which fragment tables can be built.
- CUDA 13 only matters if #9 grows a block-scaled variant; plain `f8f6f4` is what probe 5 checks.

## 8. Effect on the ceiling conclusion

The previous conclusion - acceptance line unreachable, delivery under the "stated root cause" path - stands for 2 CTA/SM but not for the single-CTA in-page chain. The in-page chain is exactly what the #1-#3 package attacks: half the smem per page, the copy off the critical path, the transpose gone. SGLang's 2.03x on the same chip with the same package is the best available estimate of what that is worth, and our floor is lower because our page is 64 tokens, not a full KV tile. The honest range for the package is 20-50%, not the "double it" the SGLang number suggests; the cheap subset (#2 alone, #8) stays 5-10%.

The external record also sets the order of attack. Nobody found a way around 99 KB per block or the register budget, #3030 - the most complete public attempt - still loses TMA to `cp.async` at short tiles, and fp8 MMA loses to bf16 on both its measurements and ours. So TMA (#5) is the last structural item, not the first, and #9 stays optional. The 2 CTA/SM verdict is softened, not overturned: reachable at 48 KB, worth ~1.05x, and blocked behind #1 anyway.
