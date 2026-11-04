"""Row 19 gate: can PV's A operand (S) be held 2:4 compressed?

Probe 18 says `mma.sp` k64 costs exactly the same as two k32 dense atoms, so the
only payoff is bytes: half the A operand and half the S stores. That makes the
question purely numerical - does dropping 2 of every 4 tokens out of S survive
the fp8 gate?

Kept values are rescaled so each group of four keeps its mass exactly
(q_i = S_i * sum(group) / sum(kept)), which leaves only the within-group
reordering as error. S is computed the way the kernel computes it (unnormalized,
max 1 per row) and the comparison is against the dequantized-V reference, both
normalized by the row's L, i.e. the metric is on the same form the suite uses.

The cache layout constants are the V32/V4 ones used by tests/_dbg_p8.py.
"""
import sys

import torch

sys.path[:0] = ["tests", "."]
import lib  # noqa: E402
from lib import RawTestParamForDecode  # noqa: E402

torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda:0")
torch.cuda.set_device("cuda:0")

p = RawTestParamForDecode(2, 4096, 1, 1, 576, False, 128, block_size=64,
                          seed=0).to_test_param()
tc = lib.generate_testcase_for_decode(p)
ks = tc.kv_scope
topk = ks.indices_in_kvcache.size(-1)
idx = torch.clamp_min(ks.indices_in_kvcache, 0)
u8 = ks.blocked_k_quantized.view(torch.uint8)
pg = u8.size(1)
b, sq, d = p.decode.b, p.s_q, p.d_qk
gkv = ks.blocked_k.view(-1, d).index_select(0, idx.view(-1)).view(b, sq, topk, d).float()
inv = ks.indices_in_kvcache == -1
if ks.topk_length is not None:
    inv |= torch.arange(0, topk).view(1, 1, topk).broadcast_to(b, sq, topk) >= ks.topk_length.view(b, 1, 1)
gkv[gkv != gkv] = 0.0

flat = idx.view(-1)
nope, sc_off = 512, 512
blk, rel = flat // pg, flat % pg
V8 = u8[blk, rel, 0, :nope].view(b, sq, topk, nope)
SV = u8[blk, rel, 0, sc_off:sc_off + 16].view(torch.float32).view(b, sq, topk, 4)

q = tc.q.float().view(b * sq, p.h_q, d)
W = q @ gkv.view(b * sq, topk, d).transpose(-1, -2) * tc.sm_scale
W[inv.view(b * sq, 1, -1).broadcast_to(b * sq, p.h_q, topk)] = float("-inf")
S = torch.exp(W - W.logsumexp(-1, keepdim=True))

M = b * sq * p.h_q
Sf = S.reshape(M, topk)
Vbh = V8.reshape(b * sq, topk, nope).float().unsqueeze(1)\
        .expand(-1, p.h_q, -1, -1).reshape(M, topk, nope)
svf = SV.reshape(b * sq, topk, 4).unsqueeze(1).expand(-1, p.h_q, -1, -1)\
        .reshape(M, topk, 4).repeat_interleave(128, dim=-1)
Oa = (Sf.unsqueeze(-1) * Vbh * svf).sum(dim=1)

Sn = Sf / Sf.amax(dim=-1, keepdim=True).clamp_min(1e-30)
K = 32


def s24(S_):
    """Keep 2 of every 4 along k, rescale the kept pair to hold the group mass."""
    g = S_.reshape(S_.size(0), -1, 4)
    top = g.topk(2, dim=-1).indices
    mask = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, top, True)
    s4 = g.sum(-1, keepdim=True)
    s2 = (g * mask).sum(-1, keepdim=True)
    scale = torch.where(s2 > 0, s4 / s2.clamp_min(1e-30), torch.ones_like(s2))
    return (g * scale).reshape(S_.size(0), -1)


def arm(S_):
    """The kernel's PV: four per-tile e4m3 S copies, each with its tile's V scale."""
    Ob = torch.zeros_like(Oa)
    for t in range(4):
        sl = slice(t * 128, (t + 1) * 128)
        Sq = (S_.unsqueeze(-1) * svf[..., sl] * K).to(torch.float8_e4m3fn).float()
        Ob[:, sl] = (Sq * Vbh[..., sl]).sum(dim=1) / K
    return Ob


def arm_exact(S_):
    """Control: the same loop with no e4m3 round trip."""
    Ob = torch.zeros_like(Oa)
    for t in range(4):
        sl = slice(t * 128, (t + 1) * 128)
        Ob[:, sl] = (S_.unsqueeze(-1) * svf[..., sl] * Vbh[..., sl]).sum(dim=1)
    return Ob


def s24_fixed(S_):
    """Keep positions 0 and 1 of every group of four - no selection in the softmax."""
    g = S_.reshape(S_.size(0), -1, 4)
    mask = torch.zeros_like(g, dtype=torch.bool)
    mask[..., :2] = True
    s4 = g.sum(-1, keepdim=True)
    s2 = (g * mask).sum(-1, keepdim=True)
    scale = torch.where(s2 > 0, s4 / s2.clamp_min(1e-30), torch.ones_like(s2))
    return (g * scale).reshape(S_.size(0), -1)


for name, use_sparse in (("exact  (control)", None), ("dense  (row 18)", False),
                         ("2:4 top2", True), ("2:4 fixed01", "fixed")):
    S_ = Sn if use_sparse is None else (
        s24_fixed(Sn) if use_sparse == "fixed" else (s24(Sn) if use_sparse else Sn))
    Ob = arm_exact(S_) if use_sparse is None else arm(S_)
    # the kernel divides by rL at the end - gate on the normalized form
    rs = S_.sum(dim=1, keepdim=True)
    e = (Oa / rs - Ob / rs).abs()
    c = torch.nn.functional.cosine_similarity(Oa, Ob, dim=-1)
    rel = (e / (Oa / rs).abs().clamp_min(1e-30)).max()
    print(f"{name}: abs worst {e.max():.3e} rel {rel:.3e}  cos worst {1 - c.min():.3e} (gates 3e-2 / 1e-2)")
print(f"O scale: |Oa| max {(Oa / Oa.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)).abs().max():.1f} normalized, "
      f"raw max {Oa.abs().max():.3e}")


# ---- second question: the same atom with A = Q (QK side) ----------------
# QK's A is Q, whose k axis is the head dim. Pruning dims is not the same
# problem as pruning tokens: the logits lose energy, so both arms below
# rescale each surviving pair by its group's mass ratio before the gemm.
def w_of(qq):
    W_ = qq.view(b * sq, p.h_q, d) @ gkv.view(b * sq, topk, d).transpose(-1, -2) * tc.sm_scale
    W_[inv.view(b * sq, 1, -1).broadcast_to(b * sq, p.h_q, topk)] = float("-inf")
    return W_


def q_prune(qq, top2):
    g = qq.view(-1, d).reshape(-1, d // 4, 4)
    if top2:
        idx2 = g.abs().topk(2, dim=-1).indices
        mask = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, idx2, True)
    else:
        mask = torch.zeros_like(g, dtype=torch.bool)
        mask[..., :2] = True
    s4 = (g * g).sum(-1, keepdim=True)
    s2 = (g * g * mask).sum(-1, keepdim=True)
    scale = torch.where(s2 > 0, (s4 / s2.clamp_min(1e-30)).sqrt(), torch.ones_like(s2))
    return (g * scale * mask).reshape(qq.view(-1, d).shape)


def s_of(W_):
    S_ = torch.exp(W_ - W_.logsumexp(-1, keepdim=True))
    return S_.view(M, topk)


def out_of(S_):
    return (S_.unsqueeze(-1) * Vbh * svf).sum(dim=1)


Oq = out_of(s_of(w_of(q)))
for name, top2 in (("Q 2:4 top2", True), ("Q 2:4 fixed01", False)):
    Op = out_of(s_of(w_of(q_prune(q, top2))))
    c = torch.nn.functional.cosine_similarity(Oq, Op, dim=-1)
    e = (Op - Oq).abs()
    print(f"{name}: abs worst {e.max():.3e}  cos worst {1 - c.min():.3e} (gates 3e-2 / 1e-2)")
