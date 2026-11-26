"""Summary of the footprint statistics (Sec. 4.3.1).

Aggregates the outputs of footprints.py: footprint radius and gain per (source layer -> destination),
linearity in the intervention strength, skip-connection removal, and the skip/deep path split.

Reads results/base/e2_*.npy; writes results/base/a2_summary.json.
Run: python experiments/response/summarize_footprints.py
"""
import json
import numpy as np
from bia.common import LAYERS, set_out
OUT = set_out("base")

DST = LAYERS + ["logits"]
st = np.load(f"{OUT}/e2_stats.npy")   # split, si, Lsrc, k, dst, total, radius, src_tot
res = {}
te = st[st[:, 0] == 1]
tab = {}
for i, L in enumerate(LAYERS):
    row = {}
    for j, D in enumerate(DST):
        m = (te[:, 2] == i) & (te[:, 4] == j)
        if not m.any():
            continue
        gain = te[m, 5] / np.maximum(te[m, 7], 1e-30)
        row[D] = {"radius_px_median": float(np.median(te[m, 6])),
                  "gain_median": float(np.median(gain)), "n": int(m.sum())}
    tab[L] = row
res["footprint"] = tab

lin = np.load(f"{OUT}/e2_linearity.npy")  # si, L, kap, tot, ref_tot, cos
lt = {}
for i, L in enumerate(LAYERS):
    d = {}
    for kap in (0.25, 0.5, 1.0, 2.0):
        m = (lin[:, 1] == i) & (lin[:, 2] == kap)
        ratio = lin[m, 3] / np.maximum(lin[m, 4], 1e-30)
        d[str(kap)] = {"energy_ratio_vs_k1": float(np.median(ratio)), "ideal": kap ** 2,
                       "shape_cos": float(np.median(lin[m, 5]))}
    lt[L] = d
res["linearity"] = lt

sk = np.load(f"{OUT}/e2_skip.npy")  # split si e up1 up2 up3 up4 logits dice dice0 flip
names = ["inc", "down1", "down2", "down3"]
skt = {}
for e in range(4):
    m = (sk[:, 0] == 1) & (sk[:, 2] == e)
    skt[names[e]] = {"mean_energy": {d: float(sk[m, 3 + q].mean()) for q, d in enumerate(["up1", "up2", "up3", "up4", "logits"])},
                     "dice_tumor": float(sk[m, 8].mean()), "dice_tumor_clean": float(sk[m, 9].mean()),
                     "pixel_flip_frac": float(sk[m, 10].mean())}
res["skip_removal"] = skt

pa = np.load(f"{OUT}/e2_path.npy")  # si L path tot rad
pt = {}
for L in names:
    i = LAYERS.index(L)
    d = {}
    for p, nm in ((0, "skip"), (1, "deep")):
        m = (pa[:, 1] == i) & (pa[:, 2] == p)
        d[nm] = {"out_energy_median": float(np.median(pa[m, 3])), "radius_px_median": float(np.median(pa[m, 4]))}
    pt[L] = d
res["path_split"] = pt
json.dump(res, open(f"{OUT}/a2_summary.json", "w"), indent=1)

print("src -> logits: radius(px) / gain")
for L in LAYERS:
    r = tab[L]["logits"]
    print(f"{L:6s} r={r['radius_px_median']:6.1f}  g={r['gain_median']:.3g}")
print("\nsrc -> each dst radius")
for L in LAYERS:
    print(L, " ".join(f"{D}:{v['radius_px_median']:.0f}" for D, v in tab[L].items()))
print("\nlinearity (energy ratio vs ideal k^2, cos)")
for L in LAYERS:
    print(L, " ".join(f"{k}:{v['energy_ratio_vs_k1']:.2f}/{v['ideal']:.2f}/{v['shape_cos']:.2f}" for k, v in lt[L].items()))
print("\nskip removal", json.dumps(skt, indent=0))
print("\npath split", json.dumps(pt, indent=0))
