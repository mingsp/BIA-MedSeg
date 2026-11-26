"""Summary of the localization results (Table 1, Fig. 8).

Per-method accuracy by intervention type (layer, patch, patch within 1, joint, MRR) with patient-level
bootstrap confidence intervals, paired differences against the best gradient/attribution baseline,
and accuracy by layer.

Reads results/base/e6_rows.pkl; writes results/base/e6_summary.json.
Run: python experiments/localization/summarize.py
"""
import json, pickle
import numpy as np
from bia.common import LAYERS, set_out
from bia.engine import NP
OUT = set_out("base")

d = pickle.load(open(f"{OUT}/e6_rows.pkl", "rb"))
rows = d["rows"]
methods = [k for k in rows[0] if k not in ("si", "type", "L", "k")]
rng = np.random.RandomState(0)
LIT = ["Grad×Act", "AtP-noise", "AtP-ablate", "Grad-CAM", "LayerCAM", "IG"]
TYPES = ["noise", "ablate", "gain", "chansub"]


def cheb(a, b):
    return max(abs(a // NP - b // NP), abs(a % NP - b % NP))


def metrics(r, m):
    pl, pk, rank = r[m]
    return {"layer": pl == r["L"], "patch": pk == r["k"], "patch1": cheb(pk, r["k"]) <= 1,
            "joint": pl == r["L"] and pk == r["k"], "joint1": pl == r["L"] and cheb(pk, r["k"]) <= 1,
            "mrr": 1.0 / (1 + rank)}


def boot(per, B=4000):
    v = np.asarray(per, float); idx = rng.randint(len(v), size=(B, len(v)))
    m = v[idx].mean(1)
    return [round(float(v.mean()), 4), round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


out = {"n_rows": len(rows), "n_patients": len({r["si"] for r in rows}), "norm": d["norm"], "by_type": {}}
for t in TYPES + ["all"]:
    R = [r for r in rows if t == "all" or r["type"] == t]
    pats = sorted({r["si"] for r in R})
    res = {}
    for m in methods:
        mm = [metrics(r, m) for r in R]
        res[m] = {k: round(float(np.mean([x[k] for x in mm])), 4) for k in mm[0]}
        res[m]["layer_ci"] = boot([np.mean([metrics(r, m)["layer"] for r in R if r["si"] == p]) for p in pats])
        res[m]["joint_ci"] = boot([np.mean([metrics(r, m)["joint"] for r in R if r["si"] == p]) for p in pats])
    best_lit = max(LIT, key=lambda m: res[m]["joint"])
    for ours in ("InvNet", "PM-mix"):
        diff = [np.mean([metrics(r, ours)["joint"] for r in R if r["si"] == p]) -
                np.mean([metrics(r, best_lit)["joint"] for r in R if r["si"] == p]) for p in pats]
        res[f"{ours}_minus_{best_lit}_joint"] = boot(diff)
    res["best_literature"] = best_lit
    res["by_layer_InvNet"] = {L: round(float(np.mean([metrics(r, "InvNet")["layer"] for r in R if r["L"] == i])), 3)
                              for i, L in enumerate(LAYERS)}
    res["by_layer_bestlit"] = {L: round(float(np.mean([metrics(r, best_lit)["layer"] for r in R if r["L"] == i])), 3)
                               for i, L in enumerate(LAYERS)}
    out["by_type"][t] = res
json.dump(out, open(f"{OUT}/e6_summary.json", "w"), indent=1, ensure_ascii=False)

order = ["InvNet", "InvNet+verify-9", "InvNet+verify-45", "PM-mix", "PM-noise(v1)"] + LIT + \
        ["Occlusion-9", "Occlusion-45", "Occlusion-90", "Occlusion-180", "Brute-noise", "Brute-sametype"]
for t in TYPES + ["all"]:
    res = out["by_type"][t]
    print(f"\n== {t}  (best lit: {res['best_literature']})")
    print(f"{'method':18s} layer  patch  patch1 joint  joint1 MRR")
    for m in order:
        if m in res:
            v = res[m]
            print(f"{m:18s} {v['layer']:.3f}  {v['patch']:.3f}  {v['patch1']:.3f}  {v['joint']:.3f}  {v['joint1']:.3f}  {v['mrr']:.3f}  L{v['layer_ci'][1:]}")
    for k, v in res.items():
        if "_minus_" in k:
            print(k, v)
print("\nby layer InvNet (all):", out["by_type"]["all"]["by_layer_InvNet"])
print("by layer best lit (all):", out["by_type"]["all"]["by_layer_bestlit"])
