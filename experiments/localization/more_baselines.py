# -*- coding: utf-8 -*-
"""Additional baselines for Table 1: LRP and Seg-HiRes-Grad CAM, plus type-matched exhaustive matching.

Evaluated on exactly the queries of evaluate.py (same seeds; the reproduction is checked row by row
against e6_rows.pkl for AtP-noise, InvNet and Brute-noise).
  LRP              zennit EpsilonPlusFlat (flat first layer, z+ for convolutions, epsilon elsewhere;
                   BatchNorm merged into the convolutions); relevance of every analysed layer for the
                   output change dy, the same objective as the gradient methods.
  Seg-HiRes-Grad CAM   HiResCAM for segmentation: ReLU(sum_c g_c * h_c).
  Exhaustive matching  with a reference per training type (noise, mean ablation, zero) and the best
                   match over the three (3 x ~509 forwards per image).
Normalisation and aggregation of the new attribution maps are chosen on the tune queries, as in evaluate.py.

Reads results/base/ (e6_rows.pkl, e6_invnet.pt, e6_tuneQ.pt); writes results/more_baselines/
(m4_rows.pkl, m4_summary.json). Environment variables: SMOKE=1 (2 patients only, writes to
results/more_baselines_smoke/), GPU_FRAC (GPU memory fraction, default 0.4).
Run: python experiments/localization/more_baselines.py
"""
import copy, json, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bia.common import *
from bia.engine import NP
from bia.core import TEST_TYPES, brain_mask_patches, make_delta, run_delta, InverseDecoder, decoder_input, patch_sum, normalise, e40, cos_amp
from bia.core import fwd_grad
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm

BASE_DIR = os.path.join(RESULTS, "base")
SMOKE = os.environ.get("SMOKE") == "1"
OUT = set_out("more_baselines" + ("_smoke" if SMOKE else ""))
T0 = time.time()
G = 40


def log(*a):
    print(*a, f"t={time.time()-T0:.0f}s", flush=True)


if DEV.type == "cuda":                          # cap the GPU memory fraction
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.4")))
net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)
net_lrp = copy.deepcopy(net)
for m in net_lrp.modules():
    if isinstance(m, nn.ReLU):
        m.inplace = False                       # zennit hooks do not allow in-place activations

E6 = pickle.load(open(f"{BASE_DIR}/e6_rows.pkl", "rb"))
rows6, norm6, calib6 = E6["rows"], E6["norm"], E6["calib"]
used = set(pick_slices("test", 40, seed=2))
test_names = [n for n in pick_slices("test", 243, seed=2) if n not in used][:60]
assert test_names == list(E6["test"]), "test patients differ from evaluate.py"
inv = InverseDecoder(cin=14, w=32).to(DEV)
inv.load_state_dict(torch.load(f"{BASE_DIR}/e6_invnet.pt", map_location=DEV)); inv.eval()
PER = 4
if SMOKE:
    test_names = test_names[:2]

NEW = ["lrp_abs", "lrp_net", "hires"]


def new_attr(x, cache, dy):
    """(B,9,100) maps for the new methods, plus AtP-noise for the reproduction check."""
    B = dy.shape[0]
    xb = x[None].expand(B, *x.shape)
    out = {k: [] for k in NEW + ["gsig"]}
    # plain gradients: Seg-HiRes-Grad CAM and AtP-noise
    hs = {}

    def keep(L):
        def fn(h):
            h.retain_grad(); hs[L] = h
            return h
        return fn
    with torch.enable_grad():
        f = fwd_grad(net, xb.clone().requires_grad_(True), edits={L: keep(L) for L in LAYERS})
        (f["logits"] * dy).sum().backward()
    for L in LAYERS:
        h, g = hs[L].detach(), hs[L].grad.detach()
        sig = cache[L].std(dim=(2, 3), keepdim=True)
        out["hires"].append(patch_sum(F.relu((g * h).sum(1)), L))
        out["gsig"].append(patch_sum(((sig * g) ** 2).sum(1), L))
    # LRP: in zennit, the gradient reaching a module output during backward is its relevance
    hs = {}
    comp = EpsilonPlusFlat(canonizers=[SequentialMergeBatchNorm()])
    with comp.context(net_lrp) as mod, torch.enable_grad():
        f = fwd_grad(mod, xb.clone().requires_grad_(True), edits={L: keep(L) for L in LAYERS})
        (f["logits"] * dy).sum().backward()
    for L in LAYERS:
        R = hs[L].grad.detach()
        out["lrp_abs"].append(patch_sum(R.abs().sum(1), L))
        out["lrp_net"].append(patch_sum(R.sum(1), L).abs())
    return {k: torch.stack(v, 1).to(HI) for k, v in out.items()}


# tune set: normalisation of the new maps
tq = torch.load(f"{BASE_DIR}/e6_tuneQ.pt")
tune_names = tq["names"][:2] if SMOKE else tq["names"]
calib_max = {g: [[] for _ in range(9)] for g in NEW}
tune_scores = []
for si, (split, nm) in enumerate(tune_names):
    x, _ = load_slice(nm, split)
    cache = forward(net, x[None])
    bm = brain_mask_patches(x)
    idx = [i for i, m in enumerate(tq["meta"]) if m[0] == si]
    for c0 in range(0, len(idx), 12):          # chunk size does not change any score (per-query maps)
        part = idx[c0:c0 + 12]
        S = new_attr(x, cache, tq["dy"][part].to(DEV).float())
        for g in NEW:
            for li in range(9):
                calib_max[g][li] += S[g][:, li].max(1).values.cpu().tolist()
        tune_scores.append((part, {g: S[g].cpu() for g in NEW}, bm.cpu()))
calib = {g: [float(np.median(v)) for v in calib_max[g]] for g in NEW}
tune_acc = {g: {m: [] for m in ("raw", "sum", "calib")} for g in NEW}
for part, Sd, bm in tune_scores:
    for g in NEW:
        for mode in ("raw", "sum", "calib"):
            S = normalise(Sd[g], mode, calib[g]).masked_fill(~bm[None, None], -1e30)
            j = S.flatten(1).argmax(1).numpy()
            tune_acc[g][mode] += [jj // 100 == tq["meta"][p_][2] for p_, jj in zip(part, j)]
tune_tab = {g: {m: float(np.mean(v)) for m, v in d.items()} for g, d in tune_acc.items()}
norm_choice = {g: max(("raw", "sum", "calib"), key=lambda m: tune_tab[g][m]) for g in NEW}
log("tune layer acc:", json.dumps(tune_tab))
del tune_scores, tq


# test: regenerate the evaluate.py queries
def place(T, ks):
    return torch.stack([T[39 - 4 * (k // NP):79 - 4 * (k // NP), 39 - 4 * (k % NP):79 - 4 * (k % NP)] for k in ks])


@torch.no_grad()
def run_cands(x, cache, cands, kind, gen, ref):
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


rng = np.random.RandomState(2026)
gen = torch.Generator(device=DEV)
rows = []
for si, nm in enumerate(test_names):
    x, _ = load_slice(nm, "test")
    cache = forward(net, x[None])
    ref = float((cache["logits"] ** 2).sum(1).mean())
    bm = brain_mask_patches(x)
    ks_all = torch.nonzero(bm).flatten().cpu().numpy().tolist()
    nK = len(ks_all)
    allc = [(li, k) for li in range(9) for k in ks_all]
    gen.manual_seed(si * 7 + 1)
    D = {"noise": run_cands(x, cache, allc, "noise", gen, ref)}          # identical to 'Brute-noise' in evaluate.py
    D["ablate"] = run_cands(x, cache, allc, "ablate", gen, ref)            # deterministic types
    D["zero"] = run_cands(x, cache, allc, "zero", gen, ref)
    for t in TEST_TYPES:
        for li, L in enumerate(LAYERS):
            ks = rng.choice(ks_all, PER)
            gen.manual_seed(int(rng.randint(1 << 30)))
            d = make_delta(cache[L], L, ks, [t] * PER, gen)
            dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
            keep = (dy.pow(2).mean((1, 2, 3)).sqrt() > 1e-6).cpu().numpy()
            obs = e40(dy, ref)
            sc = {}
            with torch.no_grad(), amp():
                s = inv(decoder_input(dy, cache["logits"], x)).float()
            sc["check_InvNet"] = s.masked_fill(~bm[None, None], -1e9)[:, :, ks_all].flatten(1).double()
            sc["check_Brute-noise"] = cos_amp(obs, D["noise"])
            sc["Exhaustive matching (3 types)"] = torch.stack([cos_amp(obs, D[tt]) for tt in ("noise", "ablate", "zero")]).max(0).values
            S = new_attr(x, cache, dy)
            sc["check_AtP-noise"] = normalise(S["gsig"], norm6["gsig"], calib6["gsig"])[:, :, ks_all].flatten(1)
            for g in NEW:
                sc[g] = normalise(S[g], norm_choice[g], calib[g])[:, :, ks_all].flatten(1)
            for b in range(PER):
                if not keep[b]:
                    continue
                true_j = li * nK + ks_all.index(int(ks[b]))
                row = {"si": si, "type": t, "L": li, "k": int(ks[b])}
                for key, v in sc.items():
                    j = int(v[b].argmax())
                    row[key] = (j // nK, ks_all[j % nK], int((v[b] > v[b, true_j]).sum()))
                rows.append(row)
    log(f"{si} {nm} K={nK} rows={len(rows)}")
pickle.dump({"rows": rows, "norm": norm_choice, "calib": calib, "tune": tune_tab},
            open(os.path.join(OUT, "m4_rows.pkl"), "wb"))


# summary
def boot(correct, si, n=1000, seed=0):
    r = np.random.RandomState(seed)
    pats = np.unique(si)
    by = {p: correct[si == p] for p in pats}
    vals = [np.concatenate([by[p] for p in r.choice(pats, len(pats))]).mean() for _ in range(n)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


summ = {"n_rows": len(rows), "norm": norm_choice, "tune_layer_acc": tune_tab, "methods": {}, "reproduction": {}}
n6 = len(rows6) if not SMOKE else len(rows)
same_q = all(rows[i]["L"] == rows6[i]["L"] and rows[i]["k"] == rows6[i]["k"] and rows[i]["type"] == rows6[i]["type"]
             for i in range(min(len(rows), n6)))
summ["reproduction"]["same_query_positions"] = bool(same_q and len(rows) == n6)
for key, k6 in (("check_InvNet", "InvNet"), ("check_Brute-noise", "Brute-noise"), ("check_AtP-noise", "AtP-noise")):
    summ["reproduction"][k6] = float(np.mean([rows[i][key][:2] == tuple(rows6[i][k6][:2]) for i in range(len(rows))]))
si_arr = np.array([r["si"] for r in rows])
for meth in [k for k in rows[0] if k not in ("si", "type", "L", "k")]:
    J = np.array([(r[meth][0] == r["L"]) and (r[meth][1] == r["k"]) for r in rows], float)
    Ly = np.array([r[meth][0] == r["L"] for r in rows], float)
    d = {"joint": float(J.mean()), "joint_ci": boot(J, si_arr), "layer": float(Ly.mean()), "layer_ci": boot(Ly, si_arr)}
    for t in TEST_TYPES:
        d[t] = float(J[np.array([r["type"] == t for r in rows])].mean())
    d["per_layer_joint"] = [float(J[np.array([r["L"] == li for r in rows])].mean()) for li in range(9)]
    summ["methods"][meth] = d
save_json(summ, os.path.join(OUT, "m4_summary.json"))
log("reproduction:", json.dumps(summ["reproduction"]))
for meth, d in summ["methods"].items():
    print(f"{meth:18s} " + " ".join(f"{d[t]:.3f}" for t in TEST_TYPES) +
          f" | all {d['joint']:.3f} [{d['joint_ci'][0]:.3f},{d['joint_ci'][1]:.3f}] layer {d['layer']:.3f}", flush=True)
log("done")
