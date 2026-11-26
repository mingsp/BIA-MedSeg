"""Shared pieces for target-driven intervention selection (Fig. 13): model, candidate interventions, gradients,
randomised sketch and response-model footprints for repairing a missed tumour.

Imported by experiments/target/dump_val.py and experiments/target/fix_missed.py.
Reads results/base/e6_templates.npz (response prototypes).
"""
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.engine import P, NP
from bia.core import brain_mask_patches, fwd_grad, cell_masks

net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)
tm = np.load(base_path("e6_templates.npz"))
TM = {L: torch.tensor(tm[f"{L}__noise"], device=DEV, dtype=torch.float32) for L in LAYERS}
ALPHAS = (1.0, 2.0, 4.0)
R_SK = 32


def margin(logits):
    return torch.logsumexp(logits[:, 1:], 1) - logits[:, 0]


def dice(a, b):
    return float(2 * (a & b).sum() / max(1, int(a.sum() + b.sum())))


def dice_b(m, g):                                  # batched: m (B,H,W) bool, g (H,W) bool
    inter = (m & g).flatten(1).sum(1).float()
    return 2 * inter / (m.flatten(1).sum(1) + g.sum()).clamp_min(1).float()


def place(T, k):
    r, q = divmod(int(k), NP)
    return T[39 - 4 * r:79 - 4 * r, 39 - 4 * q:79 - 4 * q]


def grads_all(x, objs):
    out = []
    for fobj in objs:
        hs = {}

        def keep(L):
            def fn(h):
                h.retain_grad(); hs[L] = h
                return h
            return fn
        with torch.enable_grad():
            xg = x[None].clone().requires_grad_(True)
            f = fwd_grad(net, xg, edits={L: keep(L) for L in LAYERS})
            fobj(f["logits"]).backward()
        out.append({L: hs[L].grad.detach() for L in LAYERS})
    return out


def sketch(x, seed):
    gen = torch.Generator(device=DEV); gen.manual_seed(seed)
    out = {L: [] for L in LAYERS}
    for c0 in range(0, R_SK, 8):
        r = torch.randn((8, 4, GRID, GRID), generator=gen, device=DEV)
        hs = {}

        def keep(L):
            def fn(h):
                h.retain_grad(); hs[L] = h
                return h
            return fn
        with torch.enable_grad():
            xg = x[None].expand(8, *x.shape).clone().requires_grad_(True)
            f = fwd_grad(net, xg, edits={L: keep(L) for L in LAYERS})
            (f["logits"] * r).sum().backward()
        for L in LAYERS:
            out[L].append(hs[L].grad.detach())
    return {L: torch.cat(v) for L, v in out.items()}


def prepare(x, gt, si):
    cache = forward(net, x[None])
    s = margin(cache["logits"])[0]
    pred = s > 0
    gtt = gt > 0
    fn_mask = gtt & ~pred
    if int(fn_mask.sum()) < 30:
        return None
    ys, xs = torch.nonzero(fn_mask, as_tuple=True)
    pc = (torch.arange(NP, device=DEV) * P + P / 2 - 0.5)
    dist = torch.sqrt((ys[None, None] - pc[:, None, None]) ** 2 + (xs[None, None] - pc[None, :, None]) ** 2).min(-1)[0]
    bm = brain_mask_patches(x).view(NP, NP)
    okk = torch.nonzero(((dist <= 24) & bm).flatten()).flatten()
    if len(okk) == 0:
        return None
    okk = okk[torch.argsort(dist.flatten()[okk])[:16]]
    ks = sorted(int(k) for k in okk)
    gtf = gtt.float()

    def soft_dice(lg):
        pp = torch.sigmoid(margin(lg)[0])
        return 2 * (pp * gtf).sum() / (pp.sum() + gtf.sum())
    gT, gO, gD = grads_all(x, [lambda lg: (margin(lg)[0] * fn_mask).sum(), lambda lg: (margin(lg)[0] * ~gtt).sum(),
                               soft_dice])
    sk = sketch(x, 77 + si)
    # per-candidate footprints: per-pixel energy share on the 160x160 grid, and its square root
    fps = {}
    for L in LAYERS:
        f40 = torch.stack([place(TM[L], k) for k in ks])
        f40 = f40 / f40.flatten(1).sum(1).clamp_min(1e-30).view(-1, 1, 1)
        fps[L] = torch.sqrt(F.interpolate(f40[:, None], scale_factor=4, mode="nearest")[:, 0] / 16.0)
    return dict(cache=cache, s=s, pred=pred, gtt=gtt, fn=fn_mask, ks=ks, gT=gT, gO=gO, gD=gD, sk=sk, phi=fps,
                d0=dice(pred, gtt))


def candidates(st, L, alpha):
    M = cell_masks(L, st["ks"])
    g = st["gT"][L] * M
    hn = (st["cache"][L] * M).flatten(1).norm(dim=1).view(-1, 1, 1, 1)
    gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1).clamp_min(1e-12)
    delta = alpha * hn * g / gn
    E = (torch.einsum("rcij,bcij->rb", st["sk"][L], delta) ** 2).mean(0)          # estimate of ||J delta||^2 from the sketch
    q = torch.sqrt(E).view(-1, 1, 1) * st["phi"][L]                               # (B,160,160) amplitude map
    return delta, q



def energy(st, L, delta):
    return (torch.einsum("rcij,bcij->rb", st["sk"][L], delta) ** 2).mean(0)
