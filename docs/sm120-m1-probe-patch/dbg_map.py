# dbg_map.py: reverse-engineer the row/col mapping of the TMMA C fragment.
# the kernel uses a TEMP MAP: rO[s](g) = s*100+g, copy -> oBuf(SW128) -> TMA -> out.
# out[0,0,h,:] == the 512 cols of sOBuf logical row h (the col layout is restored to logical order through the TMA swizzle).
import sys, torch
sys.path.insert(0, "tests"); sys.path.insert(0, "tests/kernelkit")
torch.use_deterministic_algorithms(False)
torch.set_default_device("cuda"); torch.cuda.set_device(0)
import flash_mla, lib
from lib import RawTestParamForDecode
import kernelkit.generate as kkg, kernelkit as kk
from kernelkit.generate import gen_non_contiguous_randn_tensor as _orig_gen
def _gen_bf16(shape, *a, **k): k.setdefault("dtype", torch.bfloat16); return _orig_gen(shape, *a, **k)
kkg.gen_non_contiguous_randn_tensor = _gen_bf16
kk.gen_non_contiguous_randn_tensor = _gen_bf16

p = RawTestParamForDecode(b=1,h_q=64,s_q=1,h_kv=1,s_kv=2046,is_varlen=False,topk=64,
                          enable_attn_sink=True,block_size=64,d_qk=576,d_v=512,
                          check_correctness=True,num_runs=0,seed=123).to_test_param()
t = lib.generate_testcase_for_decode(p)
if t.q.dtype != torch.bfloat16: t.q = t.q.to(torch.bfloat16)
meta,_ = flash_mla.get_mla_metadata()
out,lse = lib.run_flash_mla_decode(p,t,meta,None)
torch.cuda.synchronize()
o = out.float()[0,0]  # (64, 512)

# every (row, col) -> (s, g)
grid = {}
nbad = 0
for h in range(64):
    for c in range(512):
        v = int(round(o[h,c].item()))
        s, g = v // 100, v % 100
        if 0 <= s < 8 and 0 <= g < 32:
            grid[(h,c)] = (s,g)
        else:
            grid[(h,c)] = None; nbad += 1
print("non-marker:", nbad, "/32768")

# row coverage: each row should hold every g in 0..31 per segment, 8 cols per g (2 col-pairs x 4?) - look at the data
from collections import defaultdict
per_row = defaultdict(set)
for (h,c),(s,g) in grid.items():
    if (s,g) is not None: per_row[h].add(g)
for h in range(8):
    gs = sorted(per_row[h])
    print("row%2d #g=%2d missing=%s" % (h, len(gs), [g for g in range(32) if g not in gs]))

# thread0's rows: base=0 and base+8. Check whether row0 and row8 agree on the (col,g) sequence (first few) and the col set
def dump(h, n=24):
    cols = sorted(c for c in range(512) if grid.get((h,c)) is not None)
    print("row%d cols=%d first%d=%s" % (h, len(cols), n, [grid[(h,c)] for c in cols[:n]]))
    return cols
c0 = dump(0); c8 = dump(8)
print("row0 cols == row8 cols?", c0 == c8)

# if each thread owns 16 cols, row0 should come from only a few threads; check the g distribution inside each row0 segment (col//64)
for s in range(2):
    gs = [g for (h,c),(ss,g) in grid.items() if h==0 and ss==s]
    print("row0 seg%d n=%d g=%s" % (s, len(gs), sorted(set(gs))))
