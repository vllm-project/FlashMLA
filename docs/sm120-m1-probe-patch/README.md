# M1 tcgen05 probe patch (throwaway)

Used on **IOTA (RTX PRO 6000)** to answer: can sm120 reuse FlashMLA's existing tcgen05 kernels? The verdict decides whether M3/M4 take "full reuse" or the "PR#189-style SM80-MMA fallback".

## Contents
- `0001-kerutils-sm100a-range.patch` -- `KERUTILS_ENABLE_SM100A` covers sm120 (`<1200` -> `<1300`).
- `0002-setup-py-nvcc-shim-define.patch` -- nvcc temporarily gets `-DCUTLASS_ARCH_MMA_SM100{,A,F}_ENABLED` (bypassing CUTLASS's `==1000` exact match).

## Execution sequence (IOTA)

```bash
# 0) M0 environment (CUDA >= 12.9 / 13 preferred + venv + torch cu129/130) already in place
source <venv>/bin/activate

# 0.5) Submodule (ghfast hangs on that host; on IOTA use the gh-proxy mirror instead, seconds)
git config url."https://gh-proxy.com/https://github.com/".insteadOf "https://github.com/"
git submodule update --init --depth 1 csrc/cutlass   # fetches pin 147f5673; setup.py also pulls it during the build

# 1) Branch (carrying the M2 setup.py / common.h groundwork)
git checkout sm120-support

# 2) Apply the probe patches
git apply docs/sm120-m1-probe-patch/0001-kerutils-sm100a-range.patch
git apply docs/sm120-m1-probe-patch/0002-setup-py-nvcc-shim-define.patch
git diff --stat   # should show 2 files

# 3) SM120-only minimal build (compute_120a only, fastest)
FLASH_MLA_DISABLE_SM100=1 FLASH_MLA_DISABLE_SM90=1 python setup.py build_ext --inplace

# 4) Smoke test: detect and load
python - <<'PY'
import torch, flash_mla
print("cc:", torch.cuda.get_device_capability(), torch.cuda.get_device_name(0))
print("device count:", torch.cuda.device_count())
PY

# 5) Numeric comparison: V32 d_qk=576 head64 sparse decode
python -m pytest tests/test_flash_mla_sparse_decoding.py -k "v32 and 576" -x -q
```

## Verdict criteria
1. **Compilation failure** (e.g. ptxas rejects tcgen05 at compute_120a) -> the probe fails -> fallback B (PR#189 style); record the offending primitive.
2. **Compiles but numerics are wrong / NaN** -> fallback B (or a targeted fix, recording the exact error).
3. **Compiles and numerics match** -> the probe passes -> full reuse -> drop this patch and proceed to M2 -> M3/M4.

## Wrap-up
When the probe is done: `git checkout -- setup.py csrc/3rdparty/kerutils/include/kerutils/device/cuda/common.h` (drop the throwaway changes), or `git apply -R` to revert. Fold the verdict back into this fork's `docs/20260919-sm120-tcgen05-probe.md`.

## Other files
`probe*.cu`, `sizeprobe.cu`, `dbg_*.py` are the ad-hoc probes behind the measurements (instruction survival, TMEM allocation, smem budget, fragment mapping): build them with `nvcc -arch=sm_120a` and run on the device. `lse-drift/` holds the 2026-09-22 lse-drift investigation record.
