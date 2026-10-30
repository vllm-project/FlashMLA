"""Row 18 feasibility: what does quantizing P (the softmax weights) to e4m3 cost
on top of the cache's own fp8 data? Simulates the mma as S' = S * sV quantized
to e4m3 (satfinite) times the raw e4m3 cache codes, and compares against the
same S times the dequantized (bf16) cache - identical S on both sides, so the
delta is P's quantization alone."""
import sys, torch
sys.path[:0] = ["tests", "."]
import lib, ref
from lib import RawTestParamForDecode

torch.set_default_dtype(torch.bfloat16); torch.set_default_device("cuda:0"); torch.cuda.set_device("cuda:0")

p = RawTestParamForDecode(2, 4096, 1, 1, 576, False, 128, block_size=64, seed=0).to_test_param()
tc = lib.generate_testcase_for_decode(p)
ks = tc.kv_scope
topk = ks.indices_in_kvcache.size(-1)
idx = torch.clamp_min(ks.indices_in_kvcache, 0)
u8 = ks.blocked_k_quantized.view(torch.uint8)                     # [nb, page, 1, 656]
pg = u8.size(1)
b, sq, d = p.decode.b, p.s_q, p.d_qk
gkv = ks.blocked_k.view(-1, d).index_select(0, idx.view(-1)).view(b, sq, topk, d).float()
inv = ks.indices_in_kvcache == -1
if ks.topk_length is not None:
    inv |= torch.arange(0, topk).view(1, 1, topk).broadcast_to(b, sq, topk) >= ks.topk_length.view(b, 1, 1)
gkv[gkv != gkv] = 0.0

# raw cache bytes and the in-row fp32 scales of the same rows
flat = idx.view(-1)
nope, sc_off, rope_off = 512, 512, 528
blk, rel = flat // pg, flat % pg
V8 = u8[blk, rel, 0, :nope].view(b, sq, topk, nope)
SV = u8[blk, rel, 0, sc_off:sc_off + 16].view(torch.float32).view(b, sq, topk, 4)

q = tc.q.float().view(b * sq, p.h_q, d)
W = q @ gkv.view(b * sq, topk, d).transpose(-1, -2) * tc.sm_scale
W[inv.view(b * sq, 1, -1).broadcast_to(b * sq, p.h_q, topk)] = float("-inf")
S = torch.exp(W - W.logsumexp(-1, keepdim=True))                   # [b*sq, h_q, topk]

# V32's V is exactly the nope dims (d_v == HEAD_DIM_NOPE); the rope dims feed
# QK only, so there is no bf16 V segment here. M flattens (b, s_q, h_q).
M = b * sq * p.h_q
Sf = S.reshape(M, topk)
Vbh = V8.reshape(b * sq, topk, nope).float().unsqueeze(1)\
        .expand(-1, p.h_q, -1, -1).reshape(M, topk, nope)
svf = SV.reshape(b * sq, topk, 4).unsqueeze(1).expand(-1, p.h_q, -1, -1)\
        .reshape(M, topk, 4).repeat_interleave(128, dim=-1)
Oa = (Sf.unsqueeze(-1) * Vbh * svf).sum(dim=1)   # dequantized V, exact S
# Kernel-shaped S: unnormalized, max exactly 1 per row (the kernel's S is
# exp(W - m)). A *fixed* power-of-two scale then only has to clear e4m3's
# 2^-9 subnormal floor, and it folds into the rL division for free.
Sn = Sf / Sf.amax(dim=-1, keepdim=True).clamp_min(1e-30)
Oa = (Sn.unsqueeze(-1) * Vbh * svf).sum(dim=1)
for K in (32, 128, 512):
    Ob = torch.zeros_like(Oa)
    for t in range(4):
        sl = slice(t * 128, (t + 1) * 128)
        sq_ = (Sn.unsqueeze(-1) * svf[..., sl] * K).to(torch.float8_e4m3fn).float()
        Ob[:, sl] = (sq_ * Vbh[..., sl]).sum(dim=1) / K
    # the kernel divides by rL at the end - gate metrics on the normalized form
    rs = Sn.sum(dim=1, keepdim=True)
    e = (Oa / rs - Ob / rs).abs()
    c = torch.nn.functional.cosine_similarity(Oa, Ob, dim=-1)
    print(f"K={K:4d}: abs worst {e.max():.3e} (gate 3e-2)  cos worst {1 - c.min():.3e} (gate 1e-2)")

err = (Oa - Ob).abs()
print(f"O scale: |Oa| max {Oa.abs().max():.3e}  mean {Oa.abs().mean():.3e}")

