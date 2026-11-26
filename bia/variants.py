"""Variants for the fairness/ablation study (Table 2).

Defines the plain bilinear (non-invertible) alignment used as an ablation of the invertible one, a learned
decoder on top of gradient attributions, and the training variants of the localization network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from bia.common import *
from bia.engine import P, NP
from bia.core import block

GRADS5 = ["gxa", "gsig", "atp_abl", "gradcam", "layercam"]


def grid_masks(ks):
    m = torch.zeros(len(ks), 1, GRID, GRID, device=DEV)
    for b, k in enumerate(ks):
        r, q = divmod(int(k), NP)
        m[b, 0, r * P:(r + 1) * P, q * P:(q + 1) * P] = 1
    return m


def make_delta_interp(hc, L, ks, types, gen, kappa=None, alpha=None):
    """Same interventions, but through plain bilinear alignment (no exact inverse):
    g = up(h); g' = g + M*D on the 160 grid; h' = down(g'). Returns h' - h, which includes the
    round-trip error of the alignment itself."""
    n = hc.shape[-1]
    up = lambda h: F.interpolate(h, size=(GRID, GRID), mode="bilinear", align_corners=False)
    dn = lambda g: F.interpolate(g, size=(n, n), mode="bilinear", align_corners=False)
    g = up(hc)[0]
    sig = hc.std(dim=(2, 3), keepdim=True)[0]
    mu = hc.mean(dim=(2, 3), keepdim=True)[0]
    M = grid_masks(ks)
    C = hc.shape[1]
    out = []
    for b, t in enumerate(types):
        e = torch.randn(g.shape, generator=gen, device=DEV)
        if t == "noise":
            D = (1.0 if kappa is None else float(kappa[b])) * sig * e
        elif t == "ablate":
            D = mu - g
        elif t == "zero":
            D = -g
        elif t == "gain":
            D = (1.0 if alpha is None else float(alpha[b])) * g
        else:  # chansub
            cm = (torch.rand(C, 1, 1, generator=gen, device=DEV) < 0.25).float()
            D = 2.0 * sig * e * cm
        out.append(dn((g + M[b] * D)[None])[0] - hc[0])
    return torch.stack(out)


def grad_feats(S):
    """dict name -> (B,9,100)  ->  (B,45,10,10): log scores, shifted per sample and method so that the max is 0."""
    z = torch.stack([S[g].float() for g in GRADS5], 1).clamp_min(1e-30).log()     # B,5,9,100
    z = z - z.flatten(2).max(2).values[:, :, None, None]
    z = z.clamp_min(-30.0) / 10.0
    return z.reshape(z.shape[0], 45, NP, NP)


class GradDecoder(nn.Module):
    """Learned decoder on gradient attributions (plus pooled image and clean probabilities) -> (B,9,100)."""

    def __init__(self, cin=53, w=256):
        super().__init__()
        self.enc = nn.Sequential(block(cin, w), block(w, w), block(w, w), block(w, w))
        self.glob = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(2 * w, 9, 1)

    def forward(self, z):
        z = self.enc(z)
        g = self.glob(z.mean((2, 3)))[:, :, None, None].expand_as(z)
        return self.head(torch.cat([z, g], 1)).flatten(2)


def grad_in(gf, prob, x):
    """gf (B,45,10,10), prob/x (B,4,160,160) -> (B,53,10,10)."""
    return torch.cat([gf.float(), F.avg_pool2d(prob.float(), P), F.avg_pool2d(x.float(), P)], 1)


def dec_in(dy, prob, x):
    dy = dy.float(); prob = prob.float(); x = x.float()
    rms = dy.pow(2).mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
    e = dy.pow(2).sum(1, keepdim=True) / rms.pow(2)
    return torch.cat([dy / rms, torch.log1p(e), prob, x, torch.log(rms).expand_as(e) / 5.0], 1)


# InvNet input channels: 0-3 dy/rms | 4 log-energy | 5-8 clean prob | 9-12 image | 13 log rms
VARIANTS = {   # name: (buffer, channel subset, slice limit, types allowed, seed)
    "full_s2":   ("std", list(range(14)), 400, None, 2),
    "nomag":     ("std", list(range(13)), 400, None, 1),
    "dyonly":    ("std", list(range(5)), 400, None, 1),
    "noiseonly": ("std", list(range(14)), 400, ["noise"], 1),
    "n133":      ("std", list(range(14)), 133, None, 1),
    "n50":       ("std", list(range(14)), 50, None, 1),
    "interp":    ("interp", list(range(14)), 400, None, 1),
}
