#!/usr/bin/env bash
# Build and run the SM120 probes. Usage:
#   CUDA_HOME=~/cuda-13.3 ./run_all.sh [probe-substring]
# Runs on GPU 1 when a device is present, otherwise compile-only.
set -u

HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
CUDA_HOME=${CUDA_HOME:-$HOME/cuda-13.3}
NVCC=$CUDA_HOME/bin/nvcc
[ -x "$NVCC" ] || { echo "no nvcc at $NVCC (set CUDA_HOME)"; exit 2; }

FILTER=${1:-probe}
OUT=$HERE/results
mkdir -p "$OUT"

RUN=0
if command -v nvidia-smi >/dev/null 2>&1 &&
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | grep -q '^12\.'; then
  RUN=1
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
  echo "device run, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
else
  echo "no cc 12.x device, compile-only"
fi

FLAGS=(-gencode arch=compute_120a,code=sm_120a -O3 -lineinfo -lcuda)
fails=0
total=0

# build <log> <src> [extra flags...]
build() {
  local log=$1 src=$2
  shift 2
  "$NVCC" "${FLAGS[@]}" "$@" "$src" -o "${log%.log}.bin" >"$log" 2>&1
}

# cell <name> <src> <expect: pass|compile-fail> [extra flags...]
cell() {
  local name=$1 src=$2 expect=$3
  shift 3
  local log=$OUT/$name.log
  total=$((total + 1))
  if ! build "$log" "$src" "$@"; then
    if [ "$expect" = compile-fail ]; then
      echo "PASS $name (expected ptxas failure)"
    else
      echo "FAIL $name (compile)"; fails=$((fails + 1))
    fi
    return
  fi
  if [ "$expect" = compile-fail ]; then
    echo "FAIL $name (compiled, expected ptxas failure)"; fails=$((fails + 1)); return
  fi
  if [ "$RUN" = 1 ]; then
    if "${log%.log}.bin" >>"$log" 2>&1; then
      echo "PASS $name"
    else
      echo "FAIL $name (run, see $log)"; fails=$((fails + 1))
    fi
  else
    echo "SKIP-RUN $name (compiled)"
  fi
}

# sass_check <name> <src> <egrep pattern>
sass_check() {
  local name=$1 src=$2 pat=$3 log=$OUT/$1.log
  total=$((total + 1))
  if ! "$NVCC" -gencode arch=compute_120a,code=sm_120a -O3 -cubin "$src" \
       -o "${log%.log}.cubin" >>"$log" 2>&1; then
    echo "FAIL $name (compile)"; fails=$((fails + 1)); return
  fi
  if "$CUDA_HOME/bin/cuobjdump" -sass "${log%.log}.cubin" 2>/dev/null |
       grep -qE "$pat"; then
    echo "PASS $name"
  else
    echo "FAIL $name (pattern '$pat' absent)"; fails=$((fails + 1))
  fi
}

if [[ probe01 == *"$FILTER"* || "$FILTER" = probe ]]; then
  cell probe01.default probe01_tma_cta.cu pass
  cell probe01.subbyte probe01_tma_cta.cu pass -DSUBBYTE
  cell probe01.subbyte_cluster probe01_tma_cta.cu pass -DSUBBYTE -DCLUSTER
fi
if [[ probe02 == *"$FILTER"* || "$FILTER" = probe ]]; then
  cell probe02.default probe02_cpasync.cu pass
fi
if [[ probe03 == *"$FILTER"* || "$FILTER" = probe ]]; then
  cell probe03.default probe03_ldmatrix_b8trans.cu pass
fi
if [[ probe04 == *"$FILTER"* || "$FILTER" = probe ]]; then
  cell probe04.default probe04_proxy_fence.cu pass
fi
if [[ probe05 == *"$FILTER"* || "$FILTER" = probe ]]; then
  sass_check probe05.sass probe05_f8f6f4_sass.cu 'QMMA|OMMA'
fi
if [[ probe06 == *"$FILTER"* || "$FILTER" = probe ]]; then
  total=$((total + 1))
  log=$OUT/probe06.disasm.log
  if ! "$NVCC" -gencode arch=compute_120a,code=sm_120a -O3 -cubin \
       probe06_bar_lowering.cu -o "${log%.log}.cubin" >>"$log" 2>&1; then
    echo "FAIL probe06 (compile)"; fails=$((fails + 1))
  else
    dis=$("$CUDA_HOME/bin/nvdisasm" "${log%.log}.cubin" 2>/dev/null)
    echo "$dis" >"$log"
    if echo "$dis" | grep -qE 'BAR\.SYNC.*DEFER_BLOCKING' &&
       echo "$dis" | grep -qE 'SYNCS.*TRYWAIT'; then
      echo "PASS probe06.disasm"
    else
      echo "FAIL probe06.disasm"; fails=$((fails + 1))
    fi
  fi
fi
if [[ probe07 == *"$FILTER"* || "$FILTER" = probe ]]; then
  cell probe07.default probe07_load256.cu pass
  cell probe07.st_shared_v8 probe07_load256.cu compile-fail -DSTV8
fi
if [[ probe08 == *"$FILTER"* || "$FILTER" = probe ]]; then
  cell probe08.default probe08_descriptor.cu pass
fi
if [[ probe09 == *"$FILTER"* || "$FILTER" = probe ]]; then
  cell probe09.default probe09_graph_capture.cu pass
fi
if [[ probe10 == *"$FILTER"* || "$FILTER" = probe ]]; then
  cell probe10.default probe10_gather4.cu pass
  cell probe10.cta_group probe10_gather4.cu compile-fail -DCTA_GROUP
fi

echo "----"
echo "$((total - fails))/$total cells as expected"
[ "$fails" = 0 ] || { echo "$fails unexpected"; exit 1; }
