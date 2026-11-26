"""Forward fidelity of the intervention response model (Fig. 7).

Given a new image and an intervention (layer L, patch k, type t), how well is the output change
predicted without running the network? Truth is the output-logit change energy map (40x40, 4-px cells)
of the real intervention through the U-Net.
Predictors:
  PM             prototype per (L, type) placed at the patch, scaled by a per-(L, type) gain calibrated
                 on 24 tune slices x source energy. No network forward.
  Radius         structure-only prior: isotropic Gaussian at the patch, one width and one gain per
                 (L, type) fitted on the tune slices.
  Uniform        energy spread uniformly over the brain (floor).
  PM+sketch      prototype shape with total energy from a randomised Jacobian sketch,
                 ||J d||^2 ~ mean_r <J^T r, d>^2, r ~ N(0, I) on the output (R = 32 backward passes
                 per image, shared by all candidate interventions of that image).
  PM+sketch-cal  the same times a per-(L, type) bias factor calibrated on the tune slices (the gradient
                 misses one-sided max-pool / ReLU responses).
  Linear         first-order Taylor expansion (finite-difference JVP, eps = 1e-2); costs one forward
                 per query, so it measures network linearity rather than serving as a cheap predictor.
Metrics per query: shape cosine of sqrt-energy maps, |log10 energy ratio|, top-5% region IoU, with
patient-level bootstrap confidence intervals. Unseen types (gain, chansub) use the noise prototype.
Test set: the 60 held-out test patients, query RNG seed 2026 as in evaluate.py.

Reads results/base/e6_templates.npz; writes results/fidelity/ (b6_summary.json, b6_rows.json,
b6_calibration.json).
Run: python experiments/response/fidelity.py
"""
import json, time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.engine import NP
from bia.core import TRAIN_TYPES, TEST_TYPES, brain_mask_patches, make_delta, run_delta, e40, fwd_grad

OUT = set_out("fidelity")
T0 = time.time()
net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)
G = 40
PER = 4
tm = np.load(base_path("e6_templates.npz"))
TM = {(L, t): torch.tensor(tm[f"{L}__{t}"], device=DEV, dtype=torch.float32) for L in LAYERS for t in TRAIN_TYPES}
TMAP = {"noise": "noise", "ablate": "ablate", "gain": "noise", "chansub": "noise"}
yy, xx = torch.meshgrid(torch.arange(79, device=DEV, dtype=torch.float32),
                        torch.arange(79, device=DEV, dtype=torch.float32), indexing="ij")
D2 = (yy - 39) ** 2 + (xx - 39) ** 2                     # template frame: patch centre at ~(39.5, 39.5)


def place(T, k):
    r, q = divmod(int(k), NP)
    return T[39 - 4 * r:79 - 4 * r, 39 - 4 * q:79 - 4 * q]


def gauss(sig):
    g = torch.exp(-D2 / (2 * sig ** 2))
    return g / g.sum()


def queries(names, split, seed):
    """Yield per-slice query batches with the same RNG stream as evaluate.py."""
    rng = np.random.RandomState(seed)
    gen = torch.Generator(device=DEV)
    for si, nm in enumerate(names):
        x, _ = load_slice(nm, split)
        cache = forward(net, x[None])
        ref = float((cache["logits"] ** 2).sum(1).mean())
        bm = brain_mask_patches(x)
        ks_all = torch.nonzero(bm).flatten().cpu().numpy().tolist()
        for t in TEST_TYPES:
            for li, L in enumerate(LAYERS):
                ks = rng.choice(ks_all, PER)
                gen.manual_seed(int(rng.randint(1 << 30)))
                d = make_delta(cache[L], L, ks, [t] * PER, gen)
                yield si, x, cache, ref, bm, t, li, L, ks, d


def src_energy(d, L, ref_feat):
    """Source energy of the edit, relative to the clean feature energy per native cell (as in footprints.py)."""
    return (d.to(HI) ** 2).sum((1, 2, 3)) / float((ref_feat.to(HI) ** 2).sum(1).mean())


R_SK = 32
_sk = {}


def sketch(si, x):
    """Per-image Jacobian sketch: for every layer, J_L^T r for R random output probes (R, C, n, n)."""
    if si in _sk:
        return _sk[si]
    _sk.clear()
    gen = torch.Generator(device=DEV); gen.manual_seed(1000 + si)
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
    _sk[si] = {L: torch.cat(v) for L, v in out.items()}
    return _sk[si]


# calibration on the 24 tune slices
tune = pick_slices("val", 124, seed=1)[100:124]
cal = {}            # (L, t) -> list of (log gain, energy map 40x40 normalised, k)
skc = {}            # (L, t) -> list of log(true total / sketch total)   (bias of the gradient sketch)
for si, x, cache, ref, bm, t, li, L, ks, d in queries(tune, "val", 123):
    dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
    e = e40(dy, ref)
    es = src_energy(d, L, cache[L])
    skt = (torch.einsum("rcij,bcij->rb", sketch(10000 + si, x)[L], d) ** 2).mean(0) / (16.0 * ref)
    for b in range(PER):
        if float(e[b].sum()) > 1e-12 and float(skt[b]) > 0:
            skc.setdefault((L, t), []).append(float(np.log(float(e[b].sum()) / float(skt[b]))))
    for b in range(PER):
        tot = float(e[b].sum())
        if tot <= 1e-12 or float(es[b]) <= 0:
            continue
        cal.setdefault((L, t), []).append((np.log(tot / float(es[b])), (e[b] / tot), int(ks[b])))
print(f"calibration queries {sum(len(v) for v in cal.values())} t={time.time()-T0:.0f}s", flush=True)

GAIN = {k: float(np.median([a[0] for a in v])) for k, v in cal.items()}
SKB = {k: float(np.median(v)) for k, v in skc.items()}
SIGMAS = np.array([1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20], dtype=np.float32)
SIG = {}
for (L, t), v in cal.items():          # best Gaussian width (units of 4-px cells) by mean cosine
    best = None
    for s in SIGMAS:
        g = gauss(float(s))
        c = np.mean([float(F.cosine_similarity(place(g, k).flatten().sqrt()[None], m.flatten().sqrt()[None]))
                     for _, m, k in v])
        if best is None or c > best[0]:
            best = (c, float(s))
    SIG[(L, t)] = best[1]
save_json({"gain_log": {f"{L}|{t}": g for (L, t), g in GAIN.items()},
           "sigma_cells": {f"{L}|{t}": s for (L, t), s in SIG.items()},
           "sketch_log_bias": {f"{L}|{t}": v for (L, t), v in SKB.items()}}, f"{OUT}/b6_calibration.json")
print("sigma (4px cells):", {f"{L}|{t}": s for (L, t), s in SIG.items() if t == "noise"}, flush=True)


def metrics(pred, true, bmask40):
    """pred/true (40,40) energy maps (true >= 0)."""
    a, b = pred.clamp_min(0).flatten().sqrt(), true.clamp_min(0).flatten().sqrt()
    cos = float(F.cosine_similarity(a[None], b[None]))
    mag = abs(float(torch.log10(pred.sum().clamp_min(1e-30) / true.sum().clamp_min(1e-30))))
    n = max(1, int(0.05 * true.numel()))
    tp = set(torch.topk(pred.flatten(), n).indices.tolist()); tt = set(torch.topk(true.flatten(), n).indices.tolist())
    iou = len(tp & tt) / len(tp | tt)
    return cos, mag, iou


# test on the 60 held-out patients
used = set(pick_slices("test", 40, seed=2))
test_names = [n for n in pick_slices("test", 243, seed=2) if n not in used][:60]
rows = []
EPS = 1e-2
for si, x, cache, ref, bm, t, li, L, ks, d in queries(test_names, "test", 2026):
    dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
    true = e40(dy, ref)
    dyl = (run_delta(net, x, cache, L, EPS * d)["logits"] - cache["logits"]) / EPS
    lin = e40(dyl, ref)
    es = src_energy(d, L, cache[L])
    gs = sketch(si, x)[L]                                            # (R, C, n, n)
    sk_tot = (torch.einsum("rcij,bcij->rb", gs, d) ** 2).mean(0) / (16.0 * ref)   # predicted sum of e40
    tt = TMAP[t]
    bm40 = F.interpolate(bm.float().view(1, 1, NP, NP), size=(G, G), mode="nearest")[0, 0]
    for b in range(PER):
        if float(true[b].sum()) <= 1e-12:
            continue
        mag = float(np.exp(GAIN[(L, tt)]) * float(es[b]))
        pm = place(TM[(L, tt)], ks[b]); pm = pm / pm.sum().clamp_min(1e-30) * mag
        rad = place(gauss(SIG[(L, tt)]), ks[b]); rad = rad / rad.sum().clamp_min(1e-30) * mag
        uni = bm40 / bm40.sum() * mag
        pms = pm / pm.sum().clamp_min(1e-30) * float(sk_tot[b])
        pmc = pm / pm.sum().clamp_min(1e-30) * float(sk_tot[b]) * float(np.exp(SKB[(L, tt)]))
        row = {"si": si, "type": t, "L": li, "k": int(ks[b])}
        for name, pr in (("PM", pm), ("PM+sketch", pms), ("PM+sketch-cal", pmc), ("Radius", rad), ("Uniform", uni), ("Linear", lin[b])):
            row[name] = metrics(pr, true[b], bm40)
        rows.append(row)
    if li == 8 and t == TEST_TYPES[-1] and si % 10 == 0:
        print(f"test {si}/60 rows={len(rows)} t={time.time()-T0:.0f}s", flush=True)

# summary with patient bootstrap CI
def boot(vals_by_pat, B=2000, seed=0):
    rng = np.random.RandomState(seed)
    pats = list(vals_by_pat)
    means = [np.mean(np.concatenate([vals_by_pat[p] for p in rng.choice(pats, len(pats))])) for _ in range(B)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


summ = {}
for name in ("PM", "PM+sketch", "PM+sketch-cal", "Radius", "Uniform", "Linear"):
    summ[name] = {}
    for grp in TEST_TYPES + ["all"]:
        rr = [r for r in rows if grp == "all" or r["type"] == grp]
        d = {}
        for mi, met in enumerate(("cos", "mag_err_log10", "top5_iou")):
            byp = {}
            for r in rr:
                byp.setdefault(r["si"], []).append(r[name][mi])
            byp = {p: np.array(v) for p, v in byp.items()}
            d[met] = [float(np.mean([r[name][mi] for r in rr]))] + boot(byp)
        summ[name][grp] = d
    summ[name]["by_layer_cos"] = [float(np.mean([r[name][0] for r in rows if r["L"] == li])) for li in range(9)]
save_json({"summary": summ, "n_queries": len(rows), "test": test_names}, f"{OUT}/b6_summary.json")
json.dump(rows, open(f"{OUT}/b6_rows.json", "w"))
for name in summ:
    a = summ[name]["all"]
    print(f"{name:8s} cos {a['cos'][0]:.3f} [{a['cos'][1]:.3f},{a['cos'][2]:.3f}]  "
          f"|log10 mag err| {a['mag_err_log10'][0]:.3f}  top5 IoU {a['top5_iou'][0]:.3f}")
print(f"done t={time.time()-T0:.0f}s")
