import sys, torch, math
sys.path.insert(0, "tests"); sys.path.insert(0, "tests/kernelkit")
torch.use_deterministic_algorithms(False)
torch.set_default_device("cuda"); torch.cuda.set_device(0)
import flash_mla, lib, ref
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
out_ref,lse_ref = ref.ref_sparse_attn_decode(p,t)
torch.cuda.synchronize()

db = t.attn_sink[48:64].float().tolist()

o = out.float(); orf = out_ref.float()
d = (o - orf).abs().max().item()
cos = torch.nn.functional.cosine_similarity(o.flatten(), orf.flatten(), dim=0).item()
print("=== out vs ref ===  max-abs=%.4g cos=%.5f" % (d, cos))
print("  ref  [h0 c0..3]  =", ["%.4g"%orf[0,0,0,c].item() for c in range(4)])
print("  ref  [h8 c0..3]  =", ["%.4g"%orf[0,0,8,c].item() for c in range(4)])
print("  out  [h0 c0..3]  =", ["%.4g"%o[0,0,0,c].item() for c in range(4)])
print("  out  [h8 c0..3]  =", ["%.4g"%o[0,0,8,c].item() for c in range(4)])

# ---- host reference math ----
kv = t.kv_scope
idx = kv.indices_in_kvcache
topk = idx.size(-1)
gi = torch.clamp_min(idx, 0)
gk = kv.blocked_k.view(-1, p.d_qk).index_select(0, gi.view(-1)).view(1,1,topk,p.d_qk).float()
invalid = (idx == -1)
if kv.topk_length is not None:
    invalid |= torch.arange(0, topk).view(1,1,topk).to(idx.device) >= kv.topk_length.view(1,1,1)
aw = (t.q.float() @ gk.transpose(-1,-2)) * t.sm_scale          # [1,1,h_q,topk]
aw[invalid.broadcast_to(aw.shape)] = float("-inf")
lse_h = aw.logsumexp(dim=-1)
w = aw - aw.max(dim=-1, keepdim=True).values
raw_p = w.exp()
print("=== host math (per head) ===")
for h in (0, 8):
    Lh = raw_p[0,0,h].sum().item(); Mh = aw[0,0,h].max().item()
    s = t.attn_sink[h].item()
    if math.isinf(s) and s > 0: fac = 0.0
    elif math.isinf(s) and s < 0: fac = 1.0/Lh
    else: fac = 1.0/(Lh + math.exp(s - Mh))
    print("  h%d: L=%.4g M=%.4g sink=%.4g o_scale=%.6g" % (h, Lh, Mh, s, fac))
    print("      p_norm h%d c0..3 %s" % (h, ["%.4g"%orf[0,0,h,c].item() for c in range(4)]))

vv = gk[0,0].transpose(0,1)  # [d_qk, topk]
# ---- hypotheses for kernel raw rO (c0~1.1, c1~-0.13, near-const across heads)
L0 = float(raw_p[0,0,0].sum())
print("=== hypotheses ===")
print("  V[0,0]=%.6g V[0,1]=%.6g  (first token row)" % (vv[0,0].item(), vv[0,1].item()))
print("  L0*V[0,*]  = %.6g %.6g   (L=%.4g)" % (L0*vv[0,0].item(), L0*vv[0,1].item(), L0))
print("  sum_t V[t,0]=%.6g sum_t V[t,1]=%.6g" % (vv[0,:].sum().item(), vv[1,:].sum().item()))
for blk in range(8):
    sl = slice(blk*8,(blk+1)*8)
    c0 = (raw_p[0,0,0,sl]*vv[0,sl]).sum(); c1 = (raw_p[0,0,0,sl]*vv[1,sl]).sum()
    print("  tok%02d-%02d partial: c0=%.5g c1=%.5g" % (blk*8, blk*8+7, c0, c1))

# ---- rO expected (host): sum_k raw_p(h0,k) * V(k,col) for full dims 0..127
vv = gk[0,0].transpose(0,1)  # [d_qk, topk]
rp0 = raw_p[0,0,0]           # [topk]
print("=== host rO_expected (h0) first 64 dims ===")
for c in range(0, 128, 8):
    e = float((rp0 * vv[c,:64]).sum())
    print("  col%3d : host_expected=%.6g" % (c, e))
# ---- rO register probe (attn_sink[16:48]): thread0's rO[0](0..15)+rO[1](0..15)
pts = [(0,0),(0,1),(8,0),(8,1),(0,8),(0,9),(8,8),(8,9),
       (0,16),(0,17),(8,16),(8,17),(0,24),(0,25),(8,24),(8,25)]
dbb = t.attn_sink[16:48].float().tolist()
Ls = {h: float(raw_p[0,0,h].sum()) for h in (0,8)}
Ts = {h: float((raw_p[0,0,h]*torch.arange(0,64,device="cuda")).sum()) for h in (0,8)}
# marker fill: seg sg value = 1000*(sg+1) + (sg even? t : d)
def expect(sg, row, gcol):
    """unshifted structured read: value at (t, global_col-64*sg), fill base+pat"""
    base = 1000.0*(sg+1)
    d = gcol - 64*sg                       # local dim (0..63)
    if sg % 2 == 0:                        # token marker
        return Ls[row]*base + Ts[row]
    else:                                  # dim marker
        return Ls[row]*base + Ls[row]*d
print("=== rO[0]/rO[1] regs vs marker hypotheses (fill: even=base+t, odd=base+d) ===")
print("  L0=%.4g L8=%.4g T0=%.4g T8=%.4g" % (Ls[0], Ls[8], Ts[0], Ts[8]))
for s in (0, 1):
    for i,(row,gcol) in enumerate(pts):
        k = dbb[s*16+i]
        exp = {}
        exp["H0(no shift)"]   = expect(s,   row, gcol)
        exp["H1(+1 seg)"]     = expect(s+1, row, gcol)
        exp["H1(-1 seg)"]     = expect(s-1, row, gcol)
        best = min(exp, key=lambda e: abs(k-exp[e]))
        print("  rO[%d] g%2d (r%d,c%3d) : kernel=%-10.4g  %-14s=%-10.4g  all=[%s]"
              % (s, i, row, gcol, k, best, exp[best],
                 " ".join("%s:%.0f"%(k2,exp[k2]) for k2 in exp)))
