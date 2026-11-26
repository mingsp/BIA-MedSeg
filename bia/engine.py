"""Patch-intervention engine on the common 160x160 grid.

An intervention is specified as (layer l, patch P of size p on the common grid, type). Because Q_l has
orthonormal columns and patches are cell aligned (p >= s_l), the native edit is Q_l^T applied to the
common-grid edit, i.e. it acts exactly on the native cells under P. B different interventions on the
same source layer run in one batched forward pass.
"""
import torch
from bia.common import *

P = 16                      # patch size on the common grid -> 10 x 10 patches
NP = GRID // P


def patch_rc(k):
    return divmod(int(k), NP)


def brain_patches(x, frac=0.5):
    """Patches whose input pixels are mostly inside the brain (non-background)."""
    fg = (x.abs().sum(0) > 1e-6).float()
    occ = fg.view(NP, P, NP, P).mean((1, 3)).flatten()
    return [int(k) for k in torch.nonzero(occ >= frac).flatten()]


def cell_mask(layer, ks):
    """(B,1,n,n) native-cell masks for a list of patch indices on source layer."""
    s = STRIDE[layer]; n = GRID // s; c = P // s
    m = torch.zeros(len(ks), 1, n, n, device=DEV)
    for b, k in enumerate(ks):
        r, q = patch_rc(k)
        m[b, 0, r * c:(r + 1) * c, q * c:(q + 1) * c] = 1
    return m


def make_edit(layer, ks, kind, h_clean, gen, kappa=1.0, alpha=0.5):
    """Return fn(h)->h implementing B interventions (one per patch index in ks)."""
    m = cell_mask(layer, ks)
    sig = h_clean.std(dim=(2, 3), keepdim=True)          # 1,C,1,1 per-channel spatial std of the clean feature
    mu = h_clean.mean(dim=(2, 3), keepdim=True)
    if kind == "noise":
        kap = torch.as_tensor(kappa, device=DEV, dtype=torch.float32).view(-1, 1, 1, 1)
        eps = torch.randn((len(ks),) + tuple(h_clean.shape[1:]), generator=gen, device=DEV)
        return lambda h: h + kap * sig * eps * m
    if kind == "ablate":
        return lambda h: h * (1 - m) + mu * m
    if kind == "gain":
        a = torch.as_tensor(alpha, device=DEV, dtype=torch.float32).view(-1, 1, 1, 1)
        return lambda h: h * (1 + a * m)
    raise ValueError(kind)


@torch.no_grad()
def run_interventions(net, x, cache, layer, ks, kind, gen, path=None, keep=("logits",), **kw):
    """Run len(ks) interventions on `layer`, optionally only on its 'skip' or 'deep' path."""
    key = layer if path is None else f"{layer}@{path}"
    fn = make_edit(layer, ks, kind, cache[layer], gen, **kw)
    out = forward(net, x[None].expand(len(ks), *x.shape), edits={key: fn}, cache=cache)
    return {k: out[k] for k in keep}


def logit_energy(logits, clean_logits):
    """Per-pixel output change energy (B,160,160)."""
    return ((logits - clean_logits) ** 2).sum(1)


def common_energy(feat, clean, layer):
    """Relative change energy of a feature map on the common grid (B,160,160).
    Block replication spreads a native cell's energy uniformly over its s x s pixels."""
    s = STRIDE[layer]
    e = ((feat - clean) ** 2).sum(1)                        # B,n,n
    ref = (clean ** 2).sum(1).mean()                        # mean clean energy per native cell
    e = e / ref
    return e.repeat_interleave(s, 1).repeat_interleave(s, 2) if s > 1 else e


def spread_stats(e, k):
    """Radius (energy-weighted RMS distance to patch centre, px) and energy outside the patch."""
    r, q = patch_rc(k)
    cy, cx = r * P + P / 2 - 0.5, q * P + P / 2 - 0.5
    yy, xx = torch.meshgrid(torch.arange(GRID, device=e.device, dtype=HI),
                            torch.arange(GRID, device=e.device, dtype=HI), indexing="ij")
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    e = e.to(HI)
    tot = e.sum((-2, -1)).clamp_min(1e-30)
    rad = torch.sqrt((e * d2).sum((-2, -1)) / tot)
    inside = e[..., r * P:(r + 1) * P, q * P:(q + 1) * P].sum((-2, -1)) / tot
    return rad, 1 - inside, tot
