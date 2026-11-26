"""Alignment exactness (Fig. 5).

Checks the star operator Q_l on every analysed layer: round-trip exactness and norm preservation
(block replication vs. orthonormalised bilinear), comparison with naive interpolation and pooling,
equivalence of the edited network to the original, commutation of grid-space interventions with the
native edit, and spatial alignment measured by effective-receptive-field centroids.

Writes results/base/e1_star.json.
Run: python experiments/alignment/exactness.py
"""
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
OUT = set_out("base")

torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
net = load_model("baseline", 123)
names = pick_slices("test", 20, seed=0)
res = {"slices": names}

# 1. exactness and isometry
stars = {m: Star(m) for m in ("block", "bilinear")}
res["cond"] = {m: {L: stars[m].cond(GRID // STRIDE[L]) for L in LAYERS} for m in stars}
rows = {m: {L: [] for L in LAYERS} for m in stars}
naive = {L: [] for L in LAYERS}
pooled = {L: [] for L in LAYERS}
for nm in names:
    x, _ = load_slice(nm)
    f = forward(net, x[None])
    for L in LAYERS:
        h = f[L][0].double()
        n = h.shape[-1]
        for m, S in stars.items():
            z = S.to_common(h)
            hr = S.to_native(z, n)
            rows[m][L].append([float((hr - h).abs().max()), float((hr - h).norm() / h.norm()),
                               float(z.norm() / h.norm())])
        # naive interpolation round trip: bilinear up (align_corners=True) then bilinear down
        if n < GRID:
            zu = F.interpolate(h[None], size=GRID, mode="bilinear", align_corners=True)
            hd = F.interpolate(zu, size=n, mode="bilinear", align_corners=True)[0]
            naive[L].append(float((hd - h).norm() / h.norm()))
        # non-invertible alternative: pool to the coarsest grid (10x10)
        if n > 10:
            hp = F.avg_pool2d(h[None], n // 10)
            hb = F.interpolate(hp, size=n, mode="nearest")[0]
            pooled[L].append(float((hb - h).norm() / h.norm()))
res["exactness"] = {m: {L: {"max_abs_err": max(r[0] for r in rows[m][L]),
                            "max_rel_err": max(r[1] for r in rows[m][L]),
                            "norm_ratio_min": min(r[2] for r in rows[m][L]),
                            "norm_ratio_max": max(r[2] for r in rows[m][L])} for L in LAYERS}
                    for m in stars}
res["naive_interp_roundtrip_rel_err"] = {L: float(np.mean(v)) for L, v in naive.items() if v}
res["pool_to_10x10_rel_err"] = {L: float(np.mean(v)) for L, v in pooled.items() if v}

# 2. network equivalence
S = stars["block"]
dmax = {"block": 0.0, "bilinear": 0.0}
flips = {"block": 0, "bilinear": 0}
for nm in names:
    x, _ = load_slice(nm)
    ref = forward(net, x[None])["logits"]
    for m, Sm in stars.items():
        ed = {L: (lambda h, Sm=Sm: Sm.to_native(Sm.to_common(h.double()), h.shape[-1]).float())
              for L in LAYERS}
        out = forward(net, x[None], edits=ed)["logits"]
        dmax[m] = max(dmax[m], float((out - ref).abs().max()))
        flips[m] += int((out.argmax(1) != ref.argmax(1)).sum())
res["network_equivalence"] = {"max_abs_logit_diff": dmax, "label_flips_total": flips}

# 3. intervention commutation
x, _ = load_slice(names[0])
f = forward(net, x[None])
comm = {}
g = torch.Generator().manual_seed(0)
for L in LAYERS:
    s = STRIDE[L]; n = GRID // s
    h = f[L][0].double()
    z = S.to_common(h)
    out = {}
    for p in (4, 8, 16, 32):
        m = patch_mask(p, (GRID // p) // 2, (GRID // p) // 2)
        delta = torch.randn(z.shape, generator=g, dtype=torch.float64).to(DEV) * m
        h_new = h + S.to_native(delta, n)            # native edit obtained from the common-grid edit
        z_new = S.to_common(h_new)
        rep = float(S.project(delta, n).norm() ** 2 / delta.norm() ** 2)
        resid_vs_proj = float((z_new - (z + S.project(delta, n))).abs().max())
        supp = (S.to_native(delta, n).abs().sum(0) > 0)
        cells_expected = native_cells(m, s) if p >= s else None
        out[p] = {"representable_energy_fraction": rep,
                  "commutation_max_err": resid_vs_proj,
                  "native_support_cells": int(supp.sum()),
                  "support_matches_patch_cells": (bool((supp == cells_expected).all())
                                                  if cells_expected is not None else None)}
    comm[L] = out
res["commutation"] = comm

# 4. effective-receptive-field alignment
def erf_stats(L, nm, cells):
    x, _ = load_slice(nm)
    xg = x[None].clone().requires_grad_(True)
    feats = {}
    with torch.enable_grad():
        x1 = net.inc(xg); x2 = net.down1(x1); x3 = net.down2(x2); x4 = net.down3(x3); x5 = net.down4(x4)
        u1 = net.up1(x5, x4); u2 = net.up2(u1, x3); u3 = net.up3(u2, x2); u4 = net.up4(u3, x1)
        feats = dict(zip(LAYERS, [x1, x2, x3, x4, x5, u1, u2, u3, u4]))
        h = feats[L][0]
        out = []
        for (i, j) in cells:
            gr, = torch.autograd.grad(h[:, i, j].sum(), xg, retain_graph=True)
            w = gr[0].abs().sum(0).double()
            w = w / w.sum()
            yy, xx = torch.meshgrid(torch.arange(GRID, dtype=torch.float64, device=DEV),
                                    torch.arange(GRID, dtype=torch.float64, device=DEV), indexing="ij")
            cy, cx = float((w * yy).sum()), float((w * xx).sum())
            rad = float(torch.sqrt((w * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum()))
            out.append((cy, cx, rad))
    return out

erf = {}
rng = np.random.RandomState(1)
for L in LAYERS:
    s = STRIDE[L]; n = GRID // s
    lo, hi = max(1, n // 4), min(n - 1, 3 * n // 4)
    cells = [(int(rng.randint(lo, hi)), int(rng.randint(lo, hi))) for _ in range(8)]
    e_half, e_ac, rads = [], [], []
    for nm in names[:4]:
        for (i, j), (cy, cx, rad) in zip(cells, erf_stats(L, nm, cells)):
            py, px = (i + 0.5) * s - 0.5, (j + 0.5) * s - 0.5          # block / half-pixel geometry
            qy = i * (GRID - 1) / max(n - 1, 1); qx = j * (GRID - 1) / max(n - 1, 1)  # align_corners
            e_half.append(np.hypot(cy - py, cx - px)); e_ac.append(np.hypot(cy - qy, cx - qx))
            rads.append(rad)
    erf[L] = {"centroid_err_px_halfpixel": float(np.mean(e_half)),
              "centroid_err_px_aligncorners": float(np.mean(e_ac)),
              "cell_size_px": s, "erf_radius_px": float(np.mean(rads))}
res["erf_alignment"] = erf
save_json(res, f"{OUT}/e1_star.json")
print(json.dumps({k: res[k] for k in ("cond", "network_equivalence", "erf_alignment",
                                       "naive_interp_roundtrip_rel_err", "pool_to_10x10_rel_err")}, indent=1))
print(json.dumps(res["exactness"]["block"], indent=0)[:1500])
print(json.dumps(res["exactness"]["bilinear"], indent=0)[:1500])
print(json.dumps(res["commutation"]["down4"], indent=0))
