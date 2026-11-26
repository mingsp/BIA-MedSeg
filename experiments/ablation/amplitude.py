# -*- coding: utf-8 -*-
"""Sensitivity to the intervention amplitude (Fig. 10).

During training the noise amplitude kappa is drawn log-uniformly from [0.25, 2] and gain interventions are
not used. Here the amplitude is fixed at test time:
  noise  kappa in {0.125, 0.25, 0.5, 1, 2, 4}  (0.125 and 4 lie outside the training range)
  gain   alpha in {0.25, 0.5, 1, 2}            (h -> (1+alpha) h; the main experiment uses alpha = 1)
Same 60 test patients, 9 layers, 4 grid-aligned 16 px patches per layer. Methods as in the main
localization experiment, with no retraining or recalibration: localization network and prototypes from
results/base, AtP (noise), Grad x Act and LayerCAM with the normalisation chosen there.
Writes results/sensitivity/s4c_amplitude.json (S4_NPAT sets the number of test patients).
Run: python experiments/ablation/amplitude.py
"""
import os, pickle, time
import numpy as np
import torch
from bia.common import *
from bia.engine import NP
from bia.core import (brain_mask_patches, make_delta, run_delta, grad_attributions, normalise, e40, cos_amp,
                    InverseDecoder, TRAIN_TYPES)

OUT = set_out("sensitivity")
T0 = time.time()
net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)
leg = pickle.load(open(base_path("e6_rows.pkl"), "rb"))
norm_choice, calib = leg["norm"], leg["calib"]
test_names = leg["test"][:int(os.environ.get("S4_NPAT", 60))]
model = InverseDecoder(cin=14, w=32).to(DEV)
model.load_state_dict(torch.load(base_path("e6_invnet.pt"), map_location=DEV)); model.eval()
tm = np.load(base_path("e6_templates.npz"))
TM = {(L, t): torch.tensor(tm[f"{L}__{t}"], device=DEV, dtype=torch.float32) for L in LAYERS for t in TRAIN_TYPES}
PER = 4
CONDS = [("noise", k) for k in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)] + [("gain", a) for a in (0.25, 0.5, 1.0, 2.0)]
METH = ["Localization network", "Prototype matching", "AtP-noise", "Grad×Act", "LayerCAM"]
GKEY = {"AtP-noise": "gsig", "Grad×Act": "gxa", "LayerCAM": "layercam"}


def dec_in(dy, prob, x):
    rms = dy.pow(2).mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
    e = dy.pow(2).sum(1, keepdim=True) / rms.pow(2)
    return torch.cat([dy / rms, torch.log1p(e), prob.expand(dy.shape[0], -1, -1, -1),
                      x[None].expand(dy.shape[0], -1, -1, -1), torch.log(rms).expand_as(e) / 5.0], 1)


def place(T, ks):
    return torch.stack([T[39 - 4 * (k // NP):79 - 4 * (k // NP), 39 - 4 * (k % NP):79 - 4 * (k % NP)] for k in ks])


rows = []
gen = torch.Generator(device=DEV)
for si, nm in enumerate(test_names):
    x, _ = load_slice(nm, "test")
    cache = forward(net, x[None])
    ref = float((cache["logits"] ** 2).sum(1).mean())
    prob = cache["logits"].softmax(1)
    bm = brain_mask_patches(x)
    ks_all = torch.nonzero(bm).flatten().cpu().numpy().tolist(); nK = len(ks_all)
    D_mix = {t: torch.cat([place(TM[(L, t)], ks_all) for L in LAYERS]) for t in TRAIN_TYPES}
    for ci, (t, amp_v) in enumerate(CONDS):
        rng = np.random.RandomState(1000 * si + ci)
        for li, L in enumerate(LAYERS):
            ks = rng.choice(ks_all, PER).tolist()
            gen.manual_seed(int(rng.randint(1 << 30)))
            vv = np.full(PER, amp_v)
            d = make_delta(cache[L], L, ks, [t] * PER, gen, kappa=vv if t == "noise" else None,
                           alpha=vv if t == "gain" else None)
            dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
            keep = (dy.pow(2).mean((1, 2, 3)).sqrt() > 1e-6).cpu().numpy()
            obs = e40(dy, ref)
            sc = {}
            with torch.no_grad(), amp():
                s = model(dec_in(dy, prob, x)).float()
            sc["Localization network"] = s[:, :, ks_all].flatten(1).to(HI)
            sc["Prototype matching"] = torch.stack([cos_amp(obs, D_mix[tt]) for tt in TRAIN_TYPES]).max(0).values
            S = grad_attributions(net, x, cache, dy, ig_steps=0)
            for nmth, g in GKEY.items():
                sc[nmth] = normalise(S[g], norm_choice[g], calib[g])[:, :, ks_all].flatten(1)
            for b in range(PER):
                if not keep[b]:
                    continue
                row = {"si": si, "type": t, "amp": amp_v, "L": li, "k": ks[b]}
                for key, v in sc.items():
                    j = int(v[b].argmax())
                    row[key] = (j // nK, ks_all[j % nK])
                rows.append(row)
    if DEV.type == "mps":
        torch.mps.empty_cache()
    print(f"{si} {nm} K={nK} rows={len(rows)} t={time.time()-T0:.0f}s", flush=True)


def boot(vals_by_pat, B=1000):
    rng = np.random.RandomState(0); pats = list(vals_by_pat)
    m = float(np.mean(np.concatenate([vals_by_pat[p] for p in pats])))
    bs = [np.mean(np.concatenate([vals_by_pat[p] for p in rng.choice(pats, len(pats))])) for _ in range(B)]
    return [m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


summ = {}
for t, a in CONDS:
    rr = [r for r in rows if r["type"] == t and r["amp"] == a]
    d = {"n": len(rr)}
    for m in METH:
        joint = {}
        for r in rr:
            joint.setdefault(r["si"], []).append(r[m][0] == r["L"] and r[m][1] == r["k"])
        d[m] = {"joint": boot({k: np.array(v, float) for k, v in joint.items()}),
                "layer": float(np.mean([r[m][0] == r["L"] for r in rr]))}
    summ[f"{t}_{a}"] = d
    print(f"{t} {a}: n={len(rr)} " + " | ".join(f"{m} {d[m]['joint'][0]:.3f} layer {d[m]['layer']:.3f}" for m in METH), flush=True)
save_json({"summary": summ, "rows": rows}, f"{OUT}/s4c_amplitude.json")
print(f"done t={time.time()-T0:.0f}s")
