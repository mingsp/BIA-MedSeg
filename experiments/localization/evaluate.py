"""Localization on the held-out test patients (Table 1, Fig. 8).

Evaluates all solvers and baselines on the 60 test patients (never used for calibration or tuning)
with four intervention types: noise and ablate (seen in training), gain and chansub (unseen).
Methods:
  InvNet           localization network g_phi (amortised, no extra forwards)
  PM-mix           prototype matching, max over types
  PM-noise(v1)     prototype matching with the single-type footprint prototypes from footprints.py
  Grad x Act, AtP-noise, AtP-ablate, Grad-CAM, LayerCAM, IG (8 steps)   gradient/attribution baselines
  Occlusion-k      k real candidate interventions on the image (candidates = top-k of the AtP prior,
                   k in 9, 45, 90, 180), matched against the observed footprint
  InvNet+verify-k  forward re-check of the top-k InvNet candidates
  Brute-noise / Brute-sametype   exhaustive dictionaries (the latter built with the true type)
The per-layer normalisation of each gradient method ('raw', 'sum', 'calib') is chosen on the tune set.

Reads results/base/e6_invnet.pt, e6_templates.npz, e2_templates.npz, e6_tuneQ.pt;
writes results/base/e6_rows.pkl (one row per query; summarised by summarize.py).
Run: python experiments/localization/evaluate.py
"""
import time
import numpy as np
import torch
from bia.common import *
from bia.engine import NP
from bia.core import *
OUT = set_out("base")

net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)
T0 = time.time()
used = set(pick_slices("test", 40, seed=2))                      # slices used by the footprint experiments are excluded
pool = [n for n in pick_slices("test", 243, seed=2) if n not in used]
test_names = pool[:60]
PER = 4                                                          # per (slice, type, layer)
G = 40

model = InverseDecoder(cin=14, w=32).to(DEV)
model.load_state_dict(torch.load(f"{OUT}/e6_invnet.pt", map_location=DEV)); model.eval()
tm = np.load(f"{OUT}/e6_templates.npz")
TM = {(L, t): torch.tensor(tm[f"{L}__{t}"], device=DEV, dtype=torch.float32) for L in LAYERS for t in TRAIN_TYPES}
t1 = np.load(f"{OUT}/e2_templates.npz")
T1 = {L: torch.tensor(t1[f"{L}__logits"], device=DEV, dtype=torch.float32) for L in LAYERS}


def dec_in(dy, prob, x):
    rms = dy.pow(2).mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
    e = dy.pow(2).sum(1, keepdim=True) / rms.pow(2)
    return torch.cat([dy / rms, torch.log1p(e), prob.expand(dy.shape[0], -1, -1, -1),
                      x[None].expand(dy.shape[0], -1, -1, -1), torch.log(rms).expand_as(e) / 5.0], 1)


def place(T, ks):
    return torch.stack([T[39 - 4 * (k // NP):79 - 4 * (k // NP), 39 - 4 * (k % NP):79 - 4 * (k % NP)] for k in ks])


def run_cands(x, cache, cands, kind, gen, ref):
    """Footprints (n,40,40) of candidates [(layer index, patch)] by real forwards, grouped by layer."""
    out = torch.zeros(len(cands), G, G, device=DEV)
    by = {}
    for i, (li, k) in enumerate(cands):
        by.setdefault(li, []).append((i, k))
    for li, lst in by.items():
        L = LAYERS[li]
        for c0 in range(0, len(lst), 96):
            part = lst[c0:c0 + 96]
            d = make_delta(cache[L], L, [k for _, k in part], [kind] * len(part), gen)
            o = run_delta(net, x, cache, L, d)["logits"]
            out[[i for i, _ in part]] = e40(o - cache["logits"], ref)
    return out


# choose the gradient normalisation on the tune set
tq = torch.load(f"{OUT}/e6_tuneQ.pt")
tune_names = tq["names"]
GRADS = ["gxa", "gsig", "atp_abl", "gradcam", "layercam", "ig"]
tune_acc = {g: {m: [] for m in ("raw", "sum", "calib")} for g in GRADS}
calib_max = {g: [[] for _ in range(9)] for g in GRADS}
tune_scores = []
for si, (split, nm) in enumerate(tune_names):
    x, _ = load_slice(nm, split)
    cache = forward(net, x[None])
    bm = brain_mask_patches(x)
    idx = [i for i, m in enumerate(tq["meta"]) if m[0] == si]
    for c0 in range(0, len(idx), 36):
        part = idx[c0:c0 + 36]
        dy = tq["dy"][part].to(DEV).float()
        S = grad_attributions(net, x, cache, dy, ig_steps=8)
        for g in GRADS:
            for li in range(9):
                calib_max[g][li] += S[g][:, li].max(1).values.cpu().tolist()
        tune_scores.append((part, {g: S[g].cpu() for g in GRADS}, bm.cpu()))
calib = {g: [float(np.median(v)) for v in calib_max[g]] for g in GRADS}
for part, Sd, bm in tune_scores:
    for g in GRADS:
        for mode in ("raw", "sum", "calib"):
            S = normalise(Sd[g].to(DEV), mode, calib[g]).masked_fill(~bm.to(DEV)[None, None], -1e30)
            j = S.flatten(1).argmax(1).cpu().numpy()
            for p_, jj in zip(part, j):
                m = tq["meta"][p_]
                tune_acc[g][mode].append(jj // 100 == m[2])
norm_choice = {g: max(("raw", "sum", "calib"), key=lambda m: np.mean(tune_acc[g][m])) for g in GRADS}
print("grad normalisation on tune:", {g: (norm_choice[g], round(float(np.mean(tune_acc[g][norm_choice[g]])), 3)) for g in GRADS},
      f"t={time.time()-T0:.0f}s", flush=True)

# main evaluation
rng = np.random.RandomState(2026)
gen = torch.Generator(device=DEV)
rows = []
KS = (9, 45, 90, 180)
for si, nm in enumerate(test_names):
    x, _ = load_slice(nm, "test")
    cache = forward(net, x[None])
    ref = float((cache["logits"] ** 2).sum(1).mean())
    prob = cache["logits"].softmax(1)
    bm = brain_mask_patches(x)
    ks_all = torch.nonzero(bm).flatten().cpu().numpy().tolist()
    nK = len(ks_all)
    # per-image dictionaries
    D_mix = {t: torch.cat([place(TM[(L, t)], ks_all) for L in LAYERS]) for t in TRAIN_TYPES}
    D_v1 = torch.cat([place(T1[L], ks_all) for L in LAYERS])
    allc = [(li, k) for li in range(9) for k in ks_all]
    gen.manual_seed(si * 7 + 1)
    D_bru = {"noise": run_cands(x, cache, allc, "noise", gen, ref)}
    for t in ("ablate", "gain", "chansub"):
        gen.manual_seed(si * 7 + 2 + TEST_TYPES.index(t))
        D_bru[t] = run_cands(x, cache, allc, t, gen, ref)
    for t in TEST_TYPES:
        for li, L in enumerate(LAYERS):
            ks = rng.choice(ks_all, PER)
            gen.manual_seed(int(rng.randint(1 << 30)))
            d = make_delta(cache[L], L, ks, [t] * PER, gen)
            o = run_delta(net, x, cache, L, d)["logits"]
            dy = o - cache["logits"]
            keep = (dy.pow(2).mean((1, 2, 3)).sqrt() > 1e-6).cpu().numpy()
            obs = e40(dy, ref)
            sc = {}
            with torch.no_grad(), amp():
                s = model(dec_in(dy, prob, x)).float()
            s = s.masked_fill(~bm[None, None], -1e9)
            sc["InvNet"] = s[:, :, ks_all].flatten(1).double()
            sc["PM-mix"] = torch.stack([cos_amp(obs, D_mix[tt]) for tt in TRAIN_TYPES]).max(0).values
            sc["PM-noise(v1)"] = cos_amp(obs, D_v1)
            sc["Brute-noise"] = cos_amp(obs, D_bru["noise"])
            sc["Brute-sametype"] = cos_amp(obs, D_bru[t])
            S = grad_attributions(net, x, cache, dy, ig_steps=8)
            names_g = {"gxa": "Grad×Act", "gsig": "AtP-noise", "atp_abl": "AtP-ablate",
                       "gradcam": "Grad-CAM", "layercam": "LayerCAM", "ig": "IG"}
            for g, lab in names_g.items():
                sc[lab] = normalise(S[g], norm_choice[g], calib[g])[:, :, ks_all].flatten(1)
            # occlusion with budget k: the AtP-noise prior proposes candidates, verified by real forwards
            prior = sc["AtP-noise"]
            for k in KS:
                top = prior.topk(min(k, prior.shape[1]), dim=1).indices.cpu().numpy()
                Sk = torch.full_like(prior, -1.0)
                for b in range(PER):
                    cs = [allc[j] for j in top[b]]
                    fp = D_bru["noise"][top[b]]               # equals running these k candidates
                    Sk[b, top[b]] = cos_amp(obs[b:b + 1], fp)[0]
                sc[f"Occlusion-{k}"] = Sk
            # InvNet top-k verified by real forwards
            for k in (9, 45):
                top = sc["InvNet"].topk(k, dim=1).indices.cpu().numpy()
                Sk = torch.full_like(prior, -1.0)
                for b in range(PER):
                    Sk[b, top[b]] = cos_amp(obs[b:b + 1], D_bru["noise"][top[b]])[0] + \
                        1e-3 * torch.arange(k, 0, -1, device=DEV, dtype=torch.float64)
                sc[f"InvNet+verify-{k}"] = Sk
            for b in range(PER):
                if not keep[b]:
                    continue
                true_j = li * nK + ks_all.index(int(ks[b]))
                row = {"si": si, "type": t, "L": li, "k": int(ks[b])}
                for key, v in sc.items():
                    j = int(v[b].argmax())
                    row[key] = (j // nK, ks_all[j % nK], int((v[b] > v[b, true_j]).sum()))
                rows.append(row)
    print(f"{si} {nm} K={nK} rows={len(rows)} t={time.time()-T0:.0f}s", flush=True)

import pickle
pickle.dump({"rows": rows, "test": test_names, "norm": norm_choice, "calib": calib}, open(f"{OUT}/e6_rows.pkl", "wb"))
print("done", time.time() - T0)
