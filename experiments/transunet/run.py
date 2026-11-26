# -*- coding: utf-8 -*-
"""BIA on TransUNet (ResNet-50 + ViT-B/16 hybrid): Table 3 and the TransUNet part of Sec. 4.3.1.

Same data, splits, patients, query protocol and training budget as the U-Net experiments:
  - 300 train + 100 val patients: 24 mixed interventions (noise / mean ablation / zero, kappa log-uniform
    in [0.25, 2]) per layer per slice, giving the response prototypes (Eq. 7) and the training set of the
    localization network;
  - 24 other val patients: tune set (checkpoint selection, gradient-map normalisation);
  - the same 60 fresh test patients: 4 types (noise, ablate: seen; gain, chansub: unseen) x 9 layers x 4 patches.
Parts
  A  network description on 40 held-out test slices: tumour Dice, Dice when one skip is cut (channel mean),
     per-layer output radius and gain of kappa = 1 noise on every brain patch
  B  intervention buffers (in RAM) and response prototypes
  C  training of the localization network (14-channel input, same optimiser / steps / batch as the U-Net)
  D  evaluation of every method on the 8,640 test queries; patient-level bootstrap CIs
Writes results/transunet/. Run: python experiments/transunet/run.py
Environment: SMOKE=1 (tiny end-to-end check), PART_A_ONLY=1 (rerun part A only), GPU_FRAC (GPU memory fraction).
"""
import bia.transunet  # noqa: F401  (must come first: rebinds LAYERS / STRIDE / forward to TransUNet)
import json, os, pickle, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.engine import P, NP
from bia.core import (TRAIN_TYPES, TEST_TYPES, brain_mask_patches, make_delta, run_delta, InverseDecoder,
                    grad_attributions, normalise, e40, cos_amp)
from bia.variants import dec_in

assert LAYERS[0] == "R1" and forward is arch_transunet.forward_tu
SMOKE = os.environ.get("SMOKE") == "1"
OUT = set_out("transunet" + ("_smoke" if SMOKE else ""))
T0 = time.time()
STEPS = 40 if SMOKE else 24 * (86400 // 192)
BS = 32 if SMOKE else 192
G40, C79 = 40, 79


def log(*a):
    print(*a, f"t={time.time()-T0:.0f}s", flush=True)


if DEV.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.5")))
net = load_model("transunet", 123)
for p in net.parameters():
    p.requires_grad_(False)

e2_test = pick_slices("test", 40, seed=2)
val = pick_slices("val", 124, seed=1)
train_names = [("train", n) for n in pick_slices("train", 300, seed=3)] + [("val", n) for n in val[:100]]
tune_names = [("val", n) for n in val[100:124]]
used = set(e2_test)
test_names = [("test", n) for n in pick_slices("test", 243, seed=2) if n not in used][:60]
if SMOKE:
    e2_test, train_names, tune_names, test_names = e2_test[:2], train_names[:3], tune_names[:2], test_names[:2]


# A. network description
def dice(pred, gt):
    p, g = pred > 0, gt > 0
    return float(2 * (p & g).sum() / (p.sum() + g.sum()).clamp_min(1))


def e40g(d, ref):
    e = (d.float() ** 2).sum(1) / ref
    n = e.shape[-1]
    if n > G40:
        e = F.avg_pool2d(e[:, None], n // G40)[:, 0]
    elif n < G40:
        e = e.repeat_interleave(G40 // n, 1).repeat_interleave(G40 // n, 2)
    return e


def radius_out(maps, centers):
    yy, xx = np.meshgrid(np.arange(G40) * 4 + 1.5, np.arange(G40) * 4 + 1.5, indexing="ij")
    m = maps.double().cpu().numpy()
    tot = m.sum((1, 2)) + 1e-30
    rad = np.sqrt((m * ((yy[None] - centers[:, 0, None, None]) ** 2 +
                        (xx[None] - centers[:, 1, None, None]) ** 2)).sum((1, 2)) / tot)
    return rad, tot


@torch.no_grad()
def part_a():
    from bia.models.unet import UNetBaseline
    unet = UNetBaseline(4, 4, bilinear=False)
    unet.load_state_dict(torch.load(f"{CKPT}/unet_baseline/seed_123/best_model.pth", map_location="cpu"))
    unet = unet.eval().to(DEV)
    gen = torch.Generator(device=DEV)
    res = {"dice": {"transunet": [], "unet": []}, "skip_cut": {e: [] for e in ("R1", "R2", "R3")}, "stats": []}
    for si, nm in enumerate(e2_test):
        x, y = load_slice(nm, "test")
        cache = forward(net, x[None])
        res["dice"]["transunet"].append(dice(cache["logits"].argmax(1)[0], y))
        res["dice"]["unet"].append(dice(unet(x[None]).argmax(1)[0], y))
        for e in ("R1", "R2", "R3"):
            mu = cache[e].mean(dim=(2, 3), keepdim=True)
            out = forward(net, x[None], edits={f"{e}@skip": (lambda h, mu=mu: mu.expand_as(h))}, cache=cache)
            res["skip_cut"][e].append(dice(out["logits"].argmax(1)[0], y))
        ks = torch.nonzero(brain_mask_patches(x)).flatten().cpu().numpy()
        tl = np.array([(divmod(int(k), NP)[0] * 4, divmod(int(k), NP)[1] * 4) for k in ks])
        cen = tl * 4 + P / 2 - 0.5
        rl = float((cache["logits"] ** 2).sum(1).mean())
        for li, L in enumerate(LAYERS):
            gen.manual_seed(1000 * si + 17 * li)
            d = make_delta(cache[L], L, ks, ["noise"] * len(ks), gen)
            out = forward(net, x[None].expand(len(ks), *x.shape), edits={L: (lambda h, d=d: h + d)}, cache=cache)
            src = e40g(out[L] - cache[L], float((cache[L] ** 2).sum(1).mean())).sum((1, 2)).double().cpu().numpy()
            rad, tot = radius_out(e40g(out["logits"] - cache["logits"], rl), cen)
            for b in range(len(ks)):
                res["stats"].append((si, li, int(ks[b]), float(tot[b]), float(rad[b]), float(src[b])))
    st = np.array(res["stats"])
    summ = {"dice_tumour_mean": {k: float(np.mean(v)) for k, v in res["dice"].items()},
            "skip_cut_dice": {k: float(np.mean(v)) for k, v in res["skip_cut"].items()},
            "layers": {}}
    for li, L in enumerate(LAYERS):
        m = st[:, 1] == li
        summ["layers"][L] = {"stride": STRIDE[L], "radius_px_median": float(np.median(st[m, 4])),
                             "gain_median": float(np.median(st[m, 3] / np.maximum(st[m, 5], 1e-30))),
                             "n": int(m.sum())}
    np.save(os.path.join(OUT, "a_stats.npy"), st)
    save_json(summ, os.path.join(OUT, "a_summary.json"))
    log("A", json.dumps(summ["dice_tumour_mean"]), json.dumps(summ["skip_cut_dice"]))
    for L, v in summ["layers"].items():
        print(f"   {L:4s} s={v['stride']:2d} radius {v['radius_px_median']:5.1f}px  gain {v['gain_median']:.3g}", flush=True)
    del unet
    torch.cuda.empty_cache()


if os.environ.get("PART_A_ONLY") == "1":          # rerun only part A
    part_a(); sys.exit(0)
if not os.path.exists(os.path.join(OUT, "a_summary.json")):
    part_a()


# B. buffers and prototypes
@torch.no_grad()
def build(name, names, per, types_mode, seed0, type_list, tmpl=None):
    """Same RNG use as build_buffer.py (std realisation only); buffers are preallocated in RAM."""
    rng = np.random.RandomState(seed0)
    gen = torch.Generator(device=DEV)
    cap = len(names) * 9 * per * (len(type_list) if types_mode == "each" else 1)
    DYA = np.empty((cap, 4, GRID, GRID), np.float16)
    n = 0
    META, X, PROB, BM, REF = [], [], [], [], []
    for si, (split, nm) in enumerate(names):
        x, _ = load_slice(nm, split)
        cache = forward(net, x[None])
        ref = float((cache["logits"] ** 2).sum(1).mean())
        bm = brain_mask_patches(x)
        ks_all = torch.nonzero(bm).flatten().cpu().numpy()
        X.append(x.half()); PROB.append(cache["logits"].softmax(1)[0].half()); BM.append(bm); REF.append(ref)
        loops = [(t, li) for t in type_list for li in range(9)] if types_mode == "each" else [(None, li) for li in range(9)]
        for t, li in loops:
            L = LAYERS[li]
            ks = rng.choice(ks_all, per)
            if t is None:
                types = [type_list[i] for i in rng.randint(len(type_list), size=per)]
                kap = np.exp(rng.uniform(np.log(0.25), np.log(2.0), per))
            else:
                types, kap = [t] * per, None
            seed = int(rng.randint(1 << 30))
            gen.manual_seed(seed)
            d = make_delta(cache[L], L, ks, types, gen, kappa=kap)
            dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
            keep = (dy.pow(2).mean((1, 2, 3)).sqrt() > 1e-6).cpu().numpy()
            tix = [TEST_TYPES.index(tt) if tt in TEST_TYPES else 10 + TRAIN_TYPES.index(tt) for tt in types]
            meta = np.array([[si, li, int(ks[b]), tix[b], 1] for b in range(per)], np.int32)[keep]
            a = dy[torch.from_numpy(keep).to(DEV)]
            if tmpl is not None:                                  # response prototypes, Eq. (7)
                em = e40(a, ref).double().cpu().numpy()
                for j, b in enumerate(np.nonzero(keep)[0]):
                    r, q = divmod(int(ks[b]), NP)
                    y0, x0 = 39 - 4 * r, 39 - 4 * q
                    acc = tmpl[(L, types[b])]
                    acc[0][y0:y0 + 40, x0:x0 + 40] += em[j] / (em[j].sum() + 1e-30)
                    acc[1][y0:y0 + 40, x0:x0 + 40] += 1
            a = a.half().cpu().numpy()
            DYA[n:n + len(a)] = a; n += len(a); META.append(meta)
        if si % 50 == 0 or si == len(names) - 1:
            log(f"[build {name}] {si + 1}/{len(names)} n={n}")
    meta = np.concatenate(META)
    return {"n": len(meta), "dy": DYA[:n], "meta": meta, "X": torch.stack(X), "PROB": torch.stack(PROB),
            "BM": torch.stack(BM), "REF": REF, "names": names}


TE = build("test", test_names, 4, "each", 2026, TEST_TYPES)
TU = build("tune", tune_names, 4, "each", 123, TEST_TYPES)
TMPL = {(L, t): [np.zeros((C79, C79)), np.zeros((C79, C79))] for L in LAYERS for t in TRAIN_TYPES}
TR = build("train", train_names, 24, "mixed", 0, TRAIN_TYPES, tmpl=TMPL)
TM_np = {f"{L}__{t}": v[0] / np.maximum(v[1], 1) for (L, t), v in TMPL.items()}
np.savez_compressed(os.path.join(OUT, "t1_templates.npz"), **TM_np)
TM = {(L, t): torch.tensor(TM_np[f"{L}__{t}"], device=DEV, dtype=torch.float32) for L in LAYERS for t in TRAIN_TYPES}
log(f"buffers: train {TR['n']} tune {TU['n']} test {TE['n']} ({TR['dy'].nbytes/1e9:.1f} GB)")


# C. localization network
def inputs(S, idx, sel):
    m = S["meta"][idx]
    si = torch.from_numpy(m[:, 0].astype(np.int64)).to(DEV)
    dy = torch.from_numpy(S["dy"][idx]).to(DEV, non_blocking=True)
    return dec_in(dy, S["PROB"][si], S["X"][si])[:, sel], m, si


@torch.no_grad()
def predict_scores(model, S, sel):
    model.eval()
    out = []
    for i in range(0, S["n"], 256):
        idx = np.arange(i, min(i + 256, S["n"]))
        z, m, si = inputs(S, idx, sel)
        with amp():
            s = model(z).float()
        out.append(s.masked_fill(~S["BM"][si][:, None, :], -1e9).cpu())
    model.train()
    return torch.cat(out)                                         # (n, 9, 100)


def score(S, pred):
    m = S["meta"]
    out = {}
    for ti, t in enumerate(TEST_TYPES):
        sel = m[:, 3] == ti
        out[t] = {"layer": float(np.mean(pred[sel, 0] == m[sel, 1])),
                  "joint": float(np.mean((pred[sel, 0] == m[sel, 1]) & (pred[sel, 1] == m[sel, 2])))}
    out["all"] = {"layer": float(np.mean(pred[:, 0] == m[:, 1])),
                  "joint": float(np.mean((pred[:, 0] == m[:, 1]) & (pred[:, 1] == m[:, 2])))}
    return out


def argmax_lk(s):
    j = s.flatten(1).argmax(1).numpy()
    return np.stack([j // 100, j % 100], 1)


SEL = list(range(14))
ck = os.path.join(OUT, "t1_invnet.pt")
model = InverseDecoder(cin=14, w=32).to(DEV)
if os.path.exists(ck) and not SMOKE:
    model.load_state_dict(torch.load(ck, map_location=DEV))
    log("[locator] loaded")
else:
    torch.manual_seed(1); np.random.seed(1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.1)
    idx = np.arange(TR["n"])
    per_ep = len(idx) // BS
    n_ep = int(np.ceil(STEPS / per_ep))
    ev_every = max(1, n_ep // 24)
    best, best_state, tlog, step = -1, None, [], 0
    for ep in range(n_ep):
        order = idx[np.random.permutation(len(idx))]
        for b in range(per_ep):
            if step >= STEPS:
                break
            z, mb, si = inputs(TR, np.sort(order[b * BS:(b + 1) * BS]), SEL)
            with amp():
                s = model(z).float()
            s = s.masked_fill(~TR["BM"][si][:, None, :], -1e9).flatten(1)
            lab = torch.from_numpy(mb[:, 1].astype(np.int64)).to(DEV)
            kk = torch.from_numpy(mb[:, 2].astype(np.int64)).to(DEV)
            loss = F.cross_entropy(s, lab * 100 + kk) + 0.5 * F.cross_entropy(torch.logsumexp(s.view(-1, 9, 100), 2), lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
        if ep % ev_every == 0 or step >= STEPS:
            tu = score(TU, argmax_lk(predict_scores(model, TU, SEL)))
            sc = float(np.mean([tu[t]["joint"] for t in TEST_TYPES]))
            tlog.append({"epoch": ep, "step": step, "loss": float(loss), "tune": tu})
            log(f"[locator] ep {ep}/{n_ep} step {step} loss {float(loss):.3f} tune {sc:.3f}")
            if sc > best:
                best, best_state = sc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step >= STEPS:
            break
    model.load_state_dict(best_state)
    torch.save(best_state, ck)
    save_json({"best_tune_joint": best, "steps": step, "batch": BS, "n_train_queries": int(TR["n"]), "log": tlog},
              os.path.join(OUT, "t1_invnet_train.json"))
model.eval()
INV_TE = predict_scores(model, TE, SEL)                          # (n,9,100), masked
log("[locator] TEST " + json.dumps({k: round(v["joint"], 3) for k, v in score(TE, argmax_lk(INV_TE)).items()}))
del TR
import gc; gc.collect()


# D. all methods on the test queries
GRADS = ["gxa", "gsig", "atp_abl", "gradcam", "layercam", "ig"]
GNAME = {"gxa": "Grad×Act", "gsig": "AtP-noise", "atp_abl": "AtP-ablate", "gradcam": "Grad-CAM",
         "layercam": "LayerCAM", "ig": "IG"}


def slice_of(S, si):
    split, nm = S["names"][si]
    x, _ = load_slice(nm, split)
    return x, forward(net, x[None])


# gradient-map normalisation chosen on the tune set (best layer accuracy among raw / sum / calib)
tune_scores, calib_max = [], {g: [[] for _ in range(9)] for g in GRADS}
for si in range(len(TU["names"])):
    x, cache = slice_of(TU, si)
    rows = np.nonzero(TU["meta"][:, 0] == si)[0]
    for c0 in range(0, len(rows), 36):
        part = rows[c0:c0 + 36]
        dy = torch.from_numpy(TU["dy"][part]).to(DEV).float()
        S = grad_attributions(net, x, cache, dy, ig_steps=8)
        for g in GRADS:
            for li in range(9):
                calib_max[g][li] += S[g][:, li].max(1).values.cpu().tolist()
        tune_scores.append((part, {g: S[g].cpu() for g in GRADS}, TU["BM"][si].cpu()))
calib = {g: [float(np.median(v)) for v in calib_max[g]] for g in GRADS}
tune_acc = {g: {m: [] for m in ("raw", "sum", "calib")} for g in GRADS}
for part, Sd, bm in tune_scores:
    for g in GRADS:
        for mode in ("raw", "sum", "calib"):
            S = normalise(Sd[g], mode, calib[g]).masked_fill(~bm[None, None], -1e30)
            j = S.flatten(1).argmax(1).numpy()
            tune_acc[g][mode] += list(j // 100 == TU["meta"][part, 1])
norm_choice = {g: max(("raw", "sum", "calib"), key=lambda m: np.mean(tune_acc[g][m])) for g in GRADS}
log("grad normalisation on tune:", json.dumps({g: norm_choice[g] for g in GRADS}))
del tune_scores


def place(T, ks):
    return torch.stack([T[39 - 4 * (k // NP):79 - 4 * (k // NP), 39 - 4 * (k % NP):79 - 4 * (k % NP)] for k in ks])


@torch.no_grad()
def run_cands(x, cache, cands, kind, gen, ref):
    out = torch.zeros(len(cands), G40, G40, device=DEV)
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


rows_out = []
gen = torch.Generator(device=DEV)
for si in range(len(TE["names"])):
    x, cache = slice_of(TE, si)
    ref = TE["REF"][si]
    bm = TE["BM"][si]
    ks_all = torch.nonzero(bm).flatten().cpu().numpy().tolist()
    nK = len(ks_all)
    allc = [(li, k) for li in range(9) for k in ks_all]
    D_mix = {t: torch.cat([place(TM[(L, t)], ks_all) for L in LAYERS]) for t in TRAIN_TYPES}
    D_bru = {}
    for i, t in enumerate(("noise", "ablate", "gain", "chansub", "zero")):
        gen.manual_seed(si * 7 + 1 + i)
        D_bru[t] = run_cands(x, cache, allc, t, gen, ref)
    rows = np.nonzero(TE["meta"][:, 0] == si)[0]
    for c0 in range(0, len(rows), 36):
        part = rows[c0:c0 + 36]
        m = TE["meta"][part]
        dy = torch.from_numpy(TE["dy"][part]).to(DEV).float()
        obs = e40(dy, ref)
        sc = {"BIA localization network": INV_TE[part].to(DEV)[:, :, ks_all].flatten(1).double(),
              "BIA prototype matching": torch.stack([cos_amp(obs, D_mix[tt]) for tt in TRAIN_TYPES]).max(0).values,
              "Exhaustive matching (noise)": cos_amp(obs, D_bru["noise"]),
              "Exhaustive matching (3 types)": torch.stack([cos_amp(obs, D_bru[tt]) for tt in ("noise", "ablate", "zero")]).max(0).values}
        same = torch.zeros(len(part), len(allc), device=DEV, dtype=HI)
        for b in range(len(part)):
            same[b] = cos_amp(obs[b:b + 1], D_bru[TEST_TYPES[m[b, 3]]])[0]
        sc["Exhaustive matching (known type)"] = same
        S = grad_attributions(net, x, cache, dy, ig_steps=8)
        for g in GRADS:
            sc[GNAME[g]] = normalise(S[g], norm_choice[g], calib[g])[:, :, ks_all].flatten(1)
        for b in range(len(part)):
            true_j = int(m[b, 1]) * nK + ks_all.index(int(m[b, 2]))
            row = {"si": si, "type": TEST_TYPES[m[b, 3]], "L": int(m[b, 1]), "k": int(m[b, 2])}
            for key, v in sc.items():
                j = int(v[b].argmax())
                row[key] = (j // nK, ks_all[j % nK], int((v[b] > v[b, true_j]).sum()))
            rows_out.append(row)
    if si % 10 == 0 or si == len(TE["names"]) - 1:
        log(f"[eval] {si + 1}/{len(TE['names'])} rows={len(rows_out)}")
pickle.dump({"rows": rows_out, "test": TE["names"], "norm": norm_choice, "calib": calib},
            open(os.path.join(OUT, "t1_rows.pkl"), "wb"))


# summary with patient-level bootstrap
def boot(correct, si, n=1000, seed=0):
    r = np.random.RandomState(seed)
    pats = np.unique(si)
    by = {p: correct[si == p] for p in pats}
    vals = []
    for _ in range(n):
        smp = r.choice(pats, len(pats))
        vals.append(np.concatenate([by[p] for p in smp]).mean())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


methods = [k for k in rows_out[0] if k not in ("si", "type", "L", "k")]
si_arr = np.array([r["si"] for r in rows_out])
summ = {"n_rows": len(rows_out), "n_patients": int(len(np.unique(si_arr))), "norm": norm_choice, "methods": {}}
for meth in methods:
    J = np.array([(r[meth][0] == r["L"]) and (r[meth][1] == r["k"]) for r in rows_out], float)
    Ly = np.array([r[meth][0] == r["L"] for r in rows_out], float)
    d = {"joint": float(J.mean()), "joint_ci": boot(J, si_arr), "layer": float(Ly.mean()), "layer_ci": boot(Ly, si_arr)}
    for t in TEST_TYPES:
        s = np.array([r["type"] == t for r in rows_out])
        d[t] = float(J[s].mean())
    d["per_layer_joint"] = [float(J[np.array([r["L"] == li for r in rows_out])].mean()) for li in range(9)]
    summ["methods"][meth] = d
best_base = max([GNAME[g] for g in GRADS], key=lambda k: summ["methods"][k]["joint"])
Jo = np.array([(r["BIA localization network"][0] == r["L"]) and (r["BIA localization network"][1] == r["k"]) for r in rows_out], float)
Jb = np.array([(r[best_base][0] == r["L"]) and (r[best_base][1] == r["k"]) for r in rows_out], float)
summ["diff_locator_minus_best_gradient"] = {"baseline": best_base, "diff": float((Jo - Jb).mean()),
                                            "ci": boot(Jo - Jb, si_arr)}
save_json(summ, os.path.join(OUT, "t1_summary.json"))
for meth in methods:
    d = summ["methods"][meth]
    print(f"{meth:14s} " + " ".join(f"{d[t]:.3f}" for t in TEST_TYPES) +
          f" | all {d['joint']:.3f} [{d['joint_ci'][0]:.3f},{d['joint_ci'][1]:.3f}] layer {d['layer']:.3f}", flush=True)
log("done")
