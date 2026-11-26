"""Expressibility of interventions of any size and position (Fig. 5).

An intervention is specified on the common 160x160 grid: a square region R of side p at an arbitrary
position (not aligned to any layer's cells) and an edit rule (ablate: set to the channel mean;
gain: x(1+a)). It is realised as a native edit of layer l with one of three alignments:
star-block (nearest-up / sqrt(s), orthonormal and local), star-orthobil (orthonormalised bilinear:
exact and norm preserving but not local) and bilinear (hypercolumn-style bilinear up and down).
Metrics per layer and patch size: realisable (share of the intended edit the layer can express),
leak (share of the realised grid energy outside the cells touched by R), energy_ratio (realised native
edit norm relative to star-block) and out_cos (cosine of the output-change map with star-block's).
The output footprint radius against patch size is recorded as well.

Writes results/expressibility/b8_rows.json and b8_summary.json.
Run: python experiments/alignment/expressibility.py
"""
import json, time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *

OUT = set_out("expressibility")
T0 = time.time()
net = load_model("baseline", 123)
for p_ in net.parameters():
    p_.requires_grad_(False)
SIZES = (4, 8, 16, 32)
N_POS = 6
STARS = {"star-block": Star("block", dtype=torch.float32), "star-orthobil": Star("bilinear", dtype=torch.float32)}
METHODS = ["star-block", "star-orthobil", "bilinear"]


def to_grid(h, method):
    """Native (1,C,n,n) -> grid (1,C,160,160) for each method."""
    if method in STARS:
        return STARS[method].to_common(h)
    if method == "bilinear":
        return F.interpolate(h, size=(GRID, GRID), mode="bilinear", align_corners=False)
    return F.interpolate(h, size=(GRID, GRID), mode="nearest")


def to_native(z, n, method):
    if method in STARS:
        return STARS[method].to_native(z, n)
    if method == "bilinear":
        return F.interpolate(z, size=(n, n), mode="bilinear", align_corners=False)
    return F.avg_pool2d(z, GRID // n)


def grid_edit(z, M, rule):
    if rule == "ablate":
        mu = z.mean((2, 3), keepdim=True)
        return M * (mu - z)
    return M * 0.5 * z


names = pick_slices("test", 20, seed=0)
rows = []
rng = np.random.RandomState(7)
for si, nm in enumerate(names):
    x, _ = load_slice(nm, "test")
    cache = forward(net, x[None])
    fg = (x.abs().sum(0) > 1e-6)
    for p in SIZES:
        for _ in range(N_POS):
            for tries in range(100):          # arbitrary (unaligned) position, mostly inside the brain
                y0, x0 = rng.randint(0, GRID - p + 1, 2)
                if fg[y0:y0 + p, x0:x0 + p].float().mean() > 0.8:
                    break
            M = torch.zeros(1, 1, GRID, GRID, device=DEV); M[..., y0:y0 + p, x0:x0 + p] = 1
            for rule in ("ablate", "gain"):
                for li, L in enumerate(LAYERS):
                    h = cache[L]; n = h.shape[-1]; s = GRID // n
                    # cells touched by R (for leakage), on the grid
                    cellm = F.max_pool2d(M, s).repeat_interleave(s, 2).repeat_interleave(s, 3)
                    outs, dns = {}, {}
                    for meth in METHODS:
                        z = to_grid(h, meth)
                        D = grid_edit(z, M, rule)
                        dn = to_native(D, n, meth)                         # native edit (delta only)
                        dns[meth] = dn
                        back = STARS["star-block"].to_common(dn)           # common metric space: block star
                        e_back = float(back.pow(2).sum())
                        row = {"si": si, "p": p, "rule": rule, "L": li, "method": meth,
                               "realisable": float(STARS[meth].project(D, n).pow(2).sum() / D.pow(2).sum().clamp_min(1e-30))
                               if meth in STARS else float("nan"),
                               "leak": float((back.pow(2) * (1 - cellm)).sum() / max(e_back, 1e-30)),
                               "energy_ratio": float(dn.norm() / dns["star-block"].norm().clamp_min(1e-30))}
                        dy = forward(net, x[None], edits={L: (lambda hh, dn=dn: hh + dn)}, cache=cache)["logits"] - cache["logits"]
                        outs[meth] = dy
                        e = (dy ** 2).sum(1)[0]
                        cy, cx = y0 + p / 2 - 0.5, x0 + p / 2 - 0.5
                        yy, xx = torch.meshgrid(torch.arange(GRID, device=DEV, dtype=HI), torch.arange(GRID, device=DEV, dtype=HI), indexing="ij")
                        row["out_radius"] = float(torch.sqrt((e * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / e.sum().clamp_min(1e-30)))
                        row["out_energy"] = float(e.sum())
                        rows.append(row)
                    ref = outs["star-block"].flatten()
                    for r in rows[-len(METHODS):]:
                        o = outs[r["method"]].flatten()
                        r["out_cos"] = float(F.cosine_similarity(o[None], ref[None]))
    print(f"{si + 1}/{len(names)} rows={len(rows)} t={time.time()-T0:.0f}s", flush=True)

json.dump(rows, open(f"{OUT}/b8_rows.json", "w"))
summ = {}
for meth in METHODS:
    summ[meth] = {}
    for p in SIZES:
        rr = [r for r in rows if r["method"] == meth and r["p"] == p]
        summ[meth][p] = {k: [float(np.nanmean([r[k] for r in rr if r["L"] == li])) for li in range(9)]
                         for k in ("realisable", "leak", "energy_ratio", "out_cos", "out_radius")}
save_json({"summary": summ, "layers": LAYERS, "sizes": SIZES}, f"{OUT}/b8_summary.json")
for meth in METHODS:
    for k in ("leak", "energy_ratio", "out_cos"):
        print(meth, k, {p: [round(v, 2) for v in summ[meth][p][k]] for p in (4, 16)})
print(f"done t={time.time()-T0:.0f}s")
