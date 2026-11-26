"""Core library: intervention types, localization network g_phi (InverseDecoder), attribution baselines, metrics.

Used by the prototype library, the localization experiments and the baseline comparisons.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bia.common import *
from bia.engine import P, NP

TYPES = ["noise", "ablate", "zero", "gain", "chansub"]
TRAIN_TYPES = ["noise", "ablate", "zero"]
TEST_TYPES = ["noise", "ablate", "gain", "chansub"]
fwd_grad = forward.__wrapped__          # differentiable version of common.forward


def brain_mask_patches(x, frac=0.5):
    fg = (x.abs().sum(0) > 1e-6).float()
    return fg.view(NP, P, NP, P).mean((1, 3)).flatten() >= frac          # (100,) bool


def cell_masks(L, ks):
    s = STRIDE[L]; n = GRID // s; c = P // s
    m = torch.zeros(len(ks), 1, n, n, device=DEV)
    for b, k in enumerate(ks):
        r, q = divmod(int(k), NP)
        m[b, 0, r * c:(r + 1) * c, q * c:(q + 1) * c] = 1
    return m


def make_delta(hc, L, ks, types, gen, kappa=None, alpha=None):
    """Additive delta (B,C,n,n) for B interventions on clean features hc (1,C,n,n)."""
    B = len(ks)
    m = cell_masks(L, ks)
    sig = hc.std(dim=(2, 3), keepdim=True)
    mu = hc.mean(dim=(2, 3), keepdim=True)
    h = hc.expand(B, *hc.shape[1:])
    eps = torch.randn(h.shape, generator=gen, device=DEV)
    C = hc.shape[1]
    out = torch.zeros_like(h)
    for b, t in enumerate(types):
        if t == "noise":
            k = 1.0 if kappa is None else float(kappa[b])
            out[b] = k * sig[0] * eps[b]
        elif t == "ablate":
            out[b] = mu[0] - h[b]
        elif t == "zero":
            out[b] = -h[b]
        elif t == "gain":
            a = 1.0 if alpha is None else float(alpha[b])
            out[b] = a * h[b]
        elif t == "chansub":
            cm = (torch.rand(C, 1, 1, generator=gen, device=DEV) < 0.25).float()
            out[b] = 2.0 * sig[0] * eps[b] * cm
    return out * m


def run_delta(net, x, cache, L, delta, keep=("logits",)):
    out = forward(net, x[None].expand(delta.shape[0], *x.shape), edits={L: (lambda h: h + delta)}, cache=cache)
    return {k: out[k] for k in keep}


# ----------------------------------------------------------------- localization network
def decoder_input(dy, clean_logits, x):
    """Output change dy (B,4,H,W) -> (B,14,H,W) scale-normalised input of the localization network."""
    B = dy.shape[0]
    rms = dy.pow(2).mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
    e = dy.pow(2).sum(1, keepdim=True) / rms.pow(2)
    feats = [dy / rms, torch.log1p(e),
             clean_logits.softmax(1).expand(B, -1, -1, -1),
             x[None].expand(B, -1, -1, -1),
             torch.log(rms).expand(B, 1, *dy.shape[2:]) / 5.0]
    return torch.cat(feats, 1)


def block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                         nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class InverseDecoder(nn.Module):
    """Localization network g_phi: output change -> scores over (9 layers x 100 patches)."""

    def __init__(self, cin=14, w=32):
        super().__init__()
        ch = [w, 2 * w, 4 * w, 8 * w, 8 * w]
        self.enc = nn.ModuleList([block(cin, ch[0]), block(ch[0], ch[1]), block(ch[1], ch[2]),
                                  block(ch[2], ch[3]), block(ch[3], ch[4])])
        self.glob = nn.Sequential(nn.Linear(ch[4], ch[4]), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(2 * ch[4], 9, 1)

    def forward(self, z):
        for i, b in enumerate(self.enc):
            z = b(z)
            if i < 4:
                z = F.max_pool2d(z, 2)            # 160 -> 10
        g = self.glob(z.mean((2, 3)))[:, :, None, None].expand_as(z)
        return self.head(torch.cat([z, g], 1)).flatten(2)   # (B, 9, 100)


# ----------------------------------------------------------------- attribution baselines
def patch_sum(m, L):
    s = STRIDE[L]; c = P // s
    return F.avg_pool2d(m[:, None], c)[:, 0].flatten(1) * c * c          # (B,100)


def grad_attributions(net, x, cache, dy, ig_steps=8):
    """Layer attributions for the objective J = <dy, logits>. Returns dict name -> (B,9,100)."""
    B = dy.shape[0]
    out = {k: [] for k in ("gxa", "gsig", "atp_abl", "gradcam", "layercam", "ig")}
    xb = x[None].expand(B, *x.shape)
    # one backward pass gives the gradient of every layer at the clean point
    with torch.enable_grad():
        xg = xb.clone().requires_grad_(True)
        hs = {}

        def keep(L):
            def fn(h):
                h.retain_grad()
                hs[L] = h
                return h
            return fn
        f = fwd_grad(net, xg, edits={L: keep(L) for L in LAYERS})
        (f["logits"] * dy).sum().backward()
    for L in LAYERS:
        h, g = hs[L].detach(), hs[L].grad.detach()
        sig = cache[L].std(dim=(2, 3), keepdim=True)
        mu = cache[L].mean(dim=(2, 3), keepdim=True)
        out["gxa"].append(patch_sum((h * g).abs().sum(1), L))
        out["gsig"].append(patch_sum(((sig * g) ** 2).sum(1), L))          # AtP, expected effect of noise
        out["atp_abl"].append(patch_sum(((mu - h) * g).sum(1), L).abs())     # AtP, linearised mean-ablation
        w = g.mean((2, 3), keepdim=True)
        out["gradcam"].append(patch_sum(F.relu((w * h).sum(1)).abs() + F.relu(-(w * h).sum(1)), L))
        out["layercam"].append(patch_sum((F.relu(g) * h).sum(1).abs(), L))
    # integrated gradients per layer (zero baseline on that layer, Riemann midpoint)
    for L in (LAYERS if ig_steps > 0 else []):
        acc = torch.zeros_like(cache[L].expand(B, *cache[L].shape[1:]))
        for a in (np.arange(ig_steps) + 0.5) / ig_steps:
            with torch.enable_grad():
                hL = (a * cache[L]).expand(B, *cache[L].shape[1:]).detach().requires_grad_(True)
                f = fwd_grad(net, xb, edits={L: (lambda h, hL=hL: hL)}, cache=cache)
                gL, = torch.autograd.grad((f["logits"] * dy).sum(), hL)
            acc += gL / ig_steps
        out["ig"].append(patch_sum((cache[L] * acc).abs().sum(1), L))
    return {k: torch.stack(v, 1).to(HI) for k, v in out.items() if v}


def normalise(S, mode, calib=None):
    """S (B,9,100) -> comparable scores: raw | sum | calib (divide by the per-layer validation median of the max)."""
    if mode == "raw":
        return S
    if mode == "sum":
        return S / S.sum(2, keepdim=True).clamp_min(1e-30)
    return S / torch.as_tensor(calib, device=S.device, dtype=S.dtype).view(1, 9, 1)


def e40(d, ref):
    e = (d.float() ** 2).sum(1) / ref
    return F.avg_pool2d(e[:, None], 4)[:, 0]


def cos_amp(obs, dic):
    a = obs.flatten(1).to(HI).clamp_min(0).sqrt(); b = dic.flatten(1).to(HI).clamp_min(0).sqrt()
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-30); b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return a @ b.T
