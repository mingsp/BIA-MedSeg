"""Fixing missed tumour with the response model as a simulator of the goal metric (Fig. 13b,c).

For every test slice with a missed-tumour (false-negative) region T and every edit strength alpha, all candidate
(layer, patch) edits are scored without running them and the top choice is compared with the real outcome. The
layer hardly matters here; the patch does, because a pixel flips only if its current tumour margin s(u) is
close to 0, which a plain footprint-share score ignores.

Rule VE-sim (no extra forward passes):
    predicted margin after the edit   s_hat(u) = s(u) + c[L,alpha] * sqrt(E_b * fp_b(u))
      s(u)      clean tumour margin (from the clean pass)
      fp_b(u)   response-model footprint of candidate b (noise prototype, per-pixel share)
      E_b       output-change energy ||J delta_b||^2 from the shared Jacobian sketch (R=32 backward passes)
      c[L,a]    one scalar per (layer, alpha), least-squares fitted on validation slices
    predicted dDice = Dice(s_hat > 0, target) - Dice(s > 0); choose the argmax.
VE-sim(lin) uses the first-order target gain g_T.delta as amplitude instead of the sketch.
Baselines (same candidates and edit, differing only in where the edit goes): Random, GradNorm (g_T.delta),
AtP-sel (g_T.delta - [g_O.delta]+), AtP-Dice (first-order gain of the soft Dice, 1 backward pass), VE (unsigned
footprint share inside T), and Oracle (all candidates run on the real network).

Writes results/fix_missed/d4b_calibration.json and d4b_summary.json.
Run: python experiments/target/fix_missed.py
Environment: D4B_NTUNE (validation slices for calibration), D4B_NTEST (test slices).
"""
import os, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from bia.common import *
from bia.core import run_delta
from bia.target import *

OUT = set_out("fix_missed")
T0 = time.time()
MS = (1, 3, 5, 10)
N_TUNE = int(os.environ.get("D4B_NTUNE", 40))
METHODS = ("GradNorm", "AtP-sel", "AtP-Dice", "VE", "VE-sim", "VE-sim(lin)")

# ------------------------------------------------------------------ 1) calibrate c[L,alpha] on validation slices
num = {(L, a): 0.0 for L in LAYERS for a in ALPHAS}
den = {(L, a): 0.0 for L in LAYERS for a in ALPHAS}
n_tune = 0
for si, nm in enumerate(pick_slices("val", 124, seed=1)):
    if n_tune >= N_TUNE:
        break
    x, gt = load_slice(nm, "val")
    st = prepare(x, gt, 10_000 + si)
    if st is None:
        continue
    n_tune += 1
    for a in ALPHAS:
        for L in LAYERS:
            delta, q = candidates(st, L, a)
            ds = margin(run_delta(net, x, st["cache"], L, delta)["logits"]) - st["s"]
            num[(L, a)] += float((ds * q).sum()); den[(L, a)] += float((q * q).sum())
    if DEV.type == "mps":
        torch.mps.empty_cache()
CAL = {f"{L}@{a}": num[(L, a)] / max(den[(L, a)], 1e-30) for L in LAYERS for a in ALPHAS}
save_json({"n_tune_slices": n_tune, "c": CAL}, f"{OUT}/d4b_calibration.json")
print(f"calibrated on {n_tune} val slices t={time.time()-T0:.0f}s", {k: round(v, 3) for k, v in CAL.items()}, flush=True)

# ------------------------------------------------------------------ 2) test slices
names = pick_slices("test", 243, seed=2)[:int(os.environ.get("D4B_NTEST", 243))]
rows = []
for si, nm in enumerate(names):
    x, gt = load_slice(nm, "test")
    st = prepare(x, gt, si)
    if st is None:
        continue
    s, gtt, fn_mask, d0, ks = st["s"], st["gtt"], st["fn"], st["d0"], st["ks"]
    for alpha in ALPHAS:
        cands = []
        for li, L in enumerate(LAYERS):
            delta, q = candidates(st, L, alpha)
            m = margin(run_delta(net, x, st["cache"], L, delta)["logits"]) > 0
            real = dice_b(m, gtt) - d0
            atpT = (st["gT"][L] * delta).flatten(1).sum(1)
            atpO = (st["gO"][L] * delta).flatten(1).sum(1)
            atpD = (st["gD"][L] * delta).flatten(1).sum(1)
            # simulator: predicted margin -> predicted Dice (hard), soft Dice as tie-break
            def sim(amp):
                sh = s[None] + amp
                return dice_b(sh > 0, gtt) - d0 + 1e-3 * (2 * (torch.sigmoid(sh) * gtt).flatten(1).sum(1)
                                                          / (torch.sigmoid(sh).flatten(1).sum(1) + gtt.sum()))
            ve_sim = sim(CAL[f"{L}@{alpha}"] * q)
            lin_amp = (atpT / (q * fn_mask).flatten(1).sum(1).clamp_min(1e-12)).view(-1, 1, 1) * q
            ve_lin = sim(lin_amp)
            for b, k in enumerate(ks):
                fp = place(TM[L], k); fp = fp / fp.sum().clamp_min(1e-30)
                T40 = F.avg_pool2d(fn_mask.float()[None, None], 4)[0, 0]
                O40 = F.avg_pool2d((~gtt).float()[None, None], 4)[0, 0]
                shT, shO = float((fp * T40).sum()), float((fp * O40).sum())
                mb = m[b]
                cands.append({"L": li, "k": k, "dDice": float(real[b]),
                              "fixed": float((mb & fn_mask).sum() / fn_mask.sum()),
                              "newFP": int((mb & ~gtt & ~st["pred"]).sum()),
                              "lostTP": int((~mb & gtt & st["pred"]).sum()),
                              "GradNorm": float(atpT[b]), "AtP-sel": float(atpT[b] - atpO[b].clamp_min(0)),
                              "AtP-Dice": float(atpD[b]),
                              "VE": float(q[b].pow(2).sum()) * (shT - shO),
                              "VE-sim": float(ve_sim[b]), "VE-sim(lin)": float(ve_lin[b])})
        dd = np.array([c["dDice"] for c in cands])
        row = {"si": si, "name": nm, "alpha": alpha, "n_cand": len(cands), "fn_px": int(fn_mask.sum()), "dice0": d0,
               "oracle": float(dd.max()), "random": float(dd.mean()),
               "oracle_fixed": cands[int(dd.argmax())]["fixed"], "random_fixed": float(np.mean([c["fixed"] for c in cands])),
               "oracle_layer": cands[int(dd.argmax())]["L"]}
        for meth in METHODS:
            sc = np.array([c[meth] for c in cands])
            order = np.argsort(-sc, kind="stable")
            c0 = cands[int(order[0])]
            row[meth] = float(dd[order[0]])
            row[meth + "_fixed"] = c0["fixed"]; row[meth + "_newFP"] = c0["newFP"]; row[meth + "_layer"] = c0["L"]
            row[meth + "_rho"] = float(spearmanr(sc, dd).correlation) if np.std(sc) > 0 else float("nan")
            for m_ in MS:
                row[f"{meth}_top{m_}"] = float(dd[order[:m_]].max())
        rng = np.random.RandomState(si)
        for m_ in MS:
            row[f"random_top{m_}"] = float(np.mean([dd[rng.choice(len(dd), min(m_, len(dd)), replace=False)].max()
                                                    for _ in range(50)]))
        rows.append(row)
    if DEV.type == "mps":
        torch.mps.empty_cache()
    if si % 20 == 0:
        print(f"{si}/{len(names)} used={len(rows)//len(ALPHAS)} t={time.time()-T0:.0f}s", flush=True)


def boot_diff(a, b, B=2000):
    rng = np.random.RandomState(0); a, b = np.asarray(a), np.asarray(b)
    d = [np.mean((a - b)[rng.randint(0, len(a), len(a))]) for _ in range(B)]
    return [float(np.mean(a - b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


summ = {}
for alpha in ALPHAS:
    rr = [r for r in rows if r["alpha"] == alpha]
    sm = {"n_slices": len(rr)}
    for meth in ("random",) + METHODS + ("oracle",):
        sm[meth] = {"dDice": float(np.mean([r[meth] for r in rr])), "fixed": float(np.mean([r[meth + "_fixed"] for r in rr]))}
        if meth in METHODS:
            sm[meth].update(newFP=float(np.mean([r[meth + "_newFP"] for r in rr])),
                            rho=float(np.nanmean([r[meth + "_rho"] for r in rr])),
                            layer_hist=np.bincount([r[meth + "_layer"] for r in rr], minlength=9).tolist(),
                            minus_random=boot_diff([r[meth] for r in rr], [r["random"] for r in rr]))
        if meth != "oracle":
            sm[meth]["topm"] = {m_: float(np.mean([r[f"{meth}_top{m_}"] for r in rr])) for m_ in MS}
    for ours in ("VE-sim", "VE-sim(lin)"):
        for other in ("AtP-sel", "AtP-Dice", "GradNorm", "VE"):
            sm[f"{ours}_minus_{other}"] = boot_diff([r[ours] for r in rr], [r[other] for r in rr])
            sm[f"{ours}_minus_{other}_top10"] = boot_diff([r[f"{ours}_top10"] for r in rr], [r[f"{other}_top10"] for r in rr])
    summ[str(alpha)] = sm
save_json({"summary": summ, "calibration": CAL, "rows": rows}, f"{OUT}/d4b_summary.json")
for alpha in ALPHAS:
    sm = summ[str(alpha)]
    print(f"alpha={alpha} n={sm['n_slices']} " + " | ".join(f"{m} {sm[m]['dDice']:+.4f}" for m in ("random",) + METHODS + ("oracle",)))
    print("   rho " + " ".join(f"{m} {sm[m]['rho']:.2f}" for m in METHODS))
    print("   VE-sim - AtP-sel", [round(v, 4) for v in sm["VE-sim_minus_AtP-sel"]],
          " - AtP-Dice", [round(v, 4) for v in sm["VE-sim_minus_AtP-Dice"]])
print(f"done t={time.time()-T0:.0f}s")
