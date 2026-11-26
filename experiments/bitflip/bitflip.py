"""Hardware-fault localization from the output change alone: simulated single-bit flips (Table 4).

Fault model (fp32 inference): one activation value h[L, c, i, j] at the output of layer L has bit b flipped;
L is uniform over the 9 layers, (i, j) uniform over native cells inside brain patches, c uniform, b uniform in 0..31.
Faults that do not change the segmentation are 'masked'; faults with inf/NaN output are 'crash'; both are counted
but not localized. For the remaining 'visible' faults the target is (layer, 16-px patch containing the flipped cell).
No method is trained on bit flips unless stated; all use the same 60 test patients as the main localization table.

Methods: response-model prototypes (zero-shot), the localization network and the gradient map + training
(trained on synthetic patch interventions), Grad x Act / AtP-noise / LayerCAM, fault-calibrated prototypes
(same matching, prototypes built from simulated exponent-bit flips on training slices; the sqrt or log map
transform is chosen on validation slices), fault-finetuned networks, and the forward re-check with M simulations
on the top-3 candidate layers.

Reads results/base/e6_*, results/ablation/{full,gradlearned}.pt; writes results/<H1_OUT>/h1_summary.json.
Main Table 4 run: H1_VERIFY=1 H1_BUDGETS=2,5,10,20,40 python experiments/bitflip/bitflip.py
Environment: H1_OUT, H1_LOCLOG, H1_VERIFY, H1_BUDGETS, H1_NPAT, H1_PER_LAYER, H1_NCAL, H1_NVAL, H1_FT_STEPS.
"""
import os, pickle, time
import numpy as np
import torch
from bia.common import *
from bia.engine import P, NP
from bia.core import brain_mask_patches, run_delta, grad_attributions, normalise, e40, cos_amp, InverseDecoder, TRAIN_TYPES
from bia.variants import GradDecoder, grad_in, dec_in, grad_feats

OUT = set_out(os.environ.get("H1_OUT", "bitflip"))
LOCLOG = os.environ.get("H1_LOCLOG") == "1"   # extra variant: localization network with a log-support energy channel
VERIFY = os.environ.get("H1_VERIFY") == "1"   # extra method: response-model candidates checked by re-simulation (forward re-check)
VK, VMS = 3, (8, 16, 32, 64)                  # candidate layers and simulated flips per candidate
BUDGETS = [int(v) for v in os.environ.get("H1_BUDGETS", "").split(",") if v]   # numbers of calibration slices for the budget curve
T0 = time.time()
N_PAT = int(os.environ.get("H1_NPAT", 60))
PER_LAYER = int(os.environ.get("H1_PER_LAYER", 5))        # visible faults per layer per patient
TRY_BATCH, MAX_TRIES = 64, 6
net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)

leg = pickle.load(open(base_path("e6_rows.pkl"), "rb"))
norm_choice, calib = leg["norm"], leg["calib"]
test_names = leg["test"][:N_PAT]
tm = np.load(base_path("e6_templates.npz"))
TM = {(L, t): torch.tensor(tm[f"{L}__{t}"], device=DEV, dtype=torch.float32) for L in LAYERS for t in TRAIN_TYPES}
E7 = os.path.join(RESULTS, "ablation")
inv = InverseDecoder(cin=14, w=32).to(DEV); inv.load_state_dict(torch.load(f"{E7}/full.pt", map_location=DEV)); inv.eval()
gdec = GradDecoder().to(DEV); gdec.load_state_dict(torch.load(f"{E7}/gradlearned.pt", map_location=DEV)); gdec.eval()
G_NAMES = {"gxa": "Grad×Act", "gsig": "AtP-noise", "layercam": "LayerCAM"}
METHODS = ((["Response model + re-simulation (M=%d)" % m for m in VMS] if VERIFY else [])
           + [f"Response model (fault-cal., n={n})" for n in BUDGETS] + [f"Grad+training (fault-ft., n={n})" for n in BUDGETS]
           + ["Response model (fault-cal.)"] + (["Localisation net (fault-ft., log input)"] if LOCLOG else []) + ["Localisation net (fault-ft.)", "Grad+training (fault-ft.)", "Response model", "Localisation net", "Grad+training", "Grad×Act", "AtP-noise", "LayerCAM"])


def place(T, ks):
    return torch.stack([T[39 - 4 * (k // NP):79 - 4 * (k // NP), 39 - 4 * (k % NP):79 - 4 * (k % NP)] for k in ks])


def flip(vals, bits):
    """vals float32 (n,), bits int (n,) -> values with that bit flipped (CPU, exact IEEE-754)."""
    iv = vals.detach().cpu().float().contiguous().view(torch.int32)
    m = torch.tensor([(1 << int(b)) if b < 31 else -(1 << 31) for b in bits], dtype=torch.int32)
    return torch.bitwise_xor(iv, m).view(torch.float32)



def gen_faults(x, cache, lab0, L, ks_all, n_target, bit_lo, rng, stats=None, bitstat=None):
    """Random single-bit flips at layer L (bits uniform in [bit_lo, 31]); returns visible dy list and (patch, bit)."""
    s = STRIDE[L]; n = GRID // s; C = cache[L].shape[1]
    got_d, got_meta = [], []
    for _ in range(MAX_TRIES):
        if len(got_meta) >= n_target:
            break
        k = rng.choice(ks_all, TRY_BATCH)
        c = rng.randint(0, C, TRY_BATCH)
        i = (k // NP) * P // s + rng.randint(0, max(1, P // s), TRY_BATCH)
        j = (k % NP) * P // s + rng.randint(0, max(1, P // s), TRY_BATCH)
        b = rng.randint(bit_lo, 32, TRY_BATCH)
        v = cache[L][0, c, i, j]
        nv = flip(v, b).to(DEV)
        d = torch.zeros(TRY_BATCH, C, n, n, device=DEV)
        d[torch.arange(TRY_BATCH), c, i, j] = nv - v.float()
        fin = torch.isfinite(d).flatten(1).all(1)
        d[~fin] = 0
        lo = run_delta(net, x, cache, L, d)["logits"]
        ok = torch.isfinite(lo).flatten(1).all(1) & fin
        vis = ok & (lo.argmax(1) != lab0).flatten(1).any(1)
        if stats is not None:
            for t_ in range(TRY_BATCH):
                st = 1 if not ok[t_] else (2 if vis[t_] else 0)
                bitstat[b[t_], st] += 1
                stats[L]["tried"] += 1
                stats[L][["masked", "crash", "visible"][st]] += 1
        for t_ in torch.nonzero(vis).flatten().tolist():
            if len(got_meta) < n_target:
                got_d.append(lo[t_] - cache["logits"][0])
                got_meta.append((int(k[t_]), int(b[t_])))
    return got_d, got_meta


def rob(dy):
    """Scale each output change to max |value| = 1 (exponent flips reach 1e19; squaring overflows fp32).
    The matching and attribution argmax used below are invariant to a positive global scale."""
    return dy / dy.abs().flatten(1).amax(1).clamp_min(1e-30).view(-1, 1, 1, 1)


def dec_in64(dy, prob, x):
    """Localization-network input computed in float64 on CPU (keeps the true log-magnitude channel finite)."""
    B = dy.shape[0]
    z = dec_in(dy.cpu().double(), prob.cpu().double().expand(B, -1, -1, -1), x.cpu().double()[None].expand(B, -1, -1, -1))
    return z.float().to(DEV)


def log_support(dyr):
    """(B,4,H,W) output change scaled to max 1 -> (B,H,W) log-energy support over 8 decades, in [0, 1].
    Bit-flip responses are one huge peak plus a long faint tail; the linear / log1p channels of dec_in only see
    the peak, while the 'log' transform also sees where the faint tail reaches."""
    e = dyr.float().pow(2).sum(1)
    mx = e.flatten(1).max(1).values.clamp_min(1e-30).view(-1, 1, 1)
    return ((torch.log10(e.clamp_min(1e-30) / mx) + 8.0).clamp_min(0) / 8.0)


def dec_in_log(dy, prob, x):
    z = dec_in64(dy, prob, x)
    z[:, 4] = log_support(rob(dy))
    return z


def sim_impulse(x, cache, L, k, M, rs):
    """Re-simulation: M exponent-MSB-like flips (nonzero value x 2^100) at random channels / cells of patch k of
    layer L, one batched forward.  Returns (m, 4, H, W) output changes (finite ones only) or None."""
    s_ = STRIDE[L]; n = GRID // s_; h = cache[L][0]; C = h.shape[0]; w = max(1, P // s_)
    r0, q0 = (k // NP) * P // s_, (k % NP) * P // s_
    nz = torch.nonzero(h[:, r0:r0 + w, q0:q0 + w] > 0)
    if len(nz) == 0:
        return None
    pick = nz[torch.from_numpy(rs.randint(0, len(nz), M)).to(nz.device)]
    c, i, j = pick[:, 0], pick[:, 1] + r0, pick[:, 2] + q0
    d = torch.zeros(M, C, n, n, device=DEV)
    d[torch.arange(M, device=DEV), c, i, j] = h[c, i, j] * (2.0 ** 100)
    lo = run_delta(net, x, cache, L, d)["logits"]
    ok = torch.isfinite(lo).flatten(1).all(1)
    return (lo[ok] - cache["logits"][0]) if ok.any() else None


def cosm(a, b):
    a = a.flatten(1).float(); b = b.flatten(1).float()
    return (a / a.norm(dim=1, keepdim=True).clamp_min(1e-30)) @ (b / b.norm(dim=1, keepdim=True).clamp_min(1e-30)).T


def tf(e, mode):
    """energy map (B,40,40) -> matching representation."""
    if mode == "sqrt":
        return e.clamp_min(0).sqrt()
    mx = e.flatten(1).max(1).values.clamp_min(1e-30).view(-1, 1, 1)
    return (torch.log10(e.clamp_min(1e-30) / mx) + 8.0).clamp_min(0)          # support over 8 decades


def match(obs_t, D):
    a = obs_t.flatten(1).to(HI); b = D.flatten(1).to(HI)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-30); b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return a @ b.T


# ------------------------------------------------------------------ calibration of fault templates (train slices)
N_CAL = int(os.environ.get("H1_NCAL", 80)); CAL_PER = 6
FT = {m: {L: torch.zeros(79, 79, device=DEV) for L in LAYERS} for m in ("sqrt", "log")}
FC = {L: torch.zeros(79, 79, device=DEV) for L in LAYERS}
FTB = {n: {m: {L: torch.zeros(79, 79, device=DEV) for L in LAYERS} for m in ("sqrt", "log")} for n in BUDGETS}
FCB = {n: {L: torch.zeros(79, 79, device=DEV) for L in LAYERS} for n in BUDGETS}
rng_c = np.random.RandomState(11)
CAL = {"dyn": [], "logrms": [], "gf": [], "si": [], "L": [], "k": [], "X": [], "PROB": [], "BM": []}
for ci, nm in enumerate(pick_slices("train", N_CAL, seed=3)):
    x, _ = load_slice(nm, "train")
    cache = forward(net, x[None]); lab0 = cache["logits"].argmax(1)
    ref = float((cache["logits"] ** 2).sum(1).mean())
    bmc = brain_mask_patches(x)
    ks_all = torch.nonzero(bmc).flatten().cpu().numpy().tolist()
    CAL["X"].append(x.cpu()); CAL["PROB"].append(cache["logits"].softmax(1)[0].cpu()); CAL["BM"].append(bmc.cpu())
    for li, L in enumerate(LAYERS):
        dl, meta = gen_faults(x, cache, lab0, L, ks_all, CAL_PER, 23, rng_c)
        if not dl:
            continue
        dys = torch.stack(dl)
        CAL["dyn"].append(rob(dys).half().cpu())
        CAL["logrms"].append(torch.log(dys.cpu().double().pow(2).mean((1, 2, 3)).sqrt()).float())
        CAL["gf"].append(grad_feats(grad_attributions(net, x, cache, rob(dys), ig_steps=0)).half().cpu())
        CAL["si"] += [ci] * len(meta); CAL["L"] += [li] * len(meta); CAL["k"] += [m_[0] for m_ in meta]
        e = e40(rob(dys), 1.0)
        for mode in ("sqrt", "log"):
            g = tf(e, mode); g = g / g.flatten(1).norm(dim=1).clamp_min(1e-30).view(-1, 1, 1)
            for bi, (k, _) in enumerate(meta):
                r, q = divmod(k, NP)
                FT[mode][L][39 - 4 * r:79 - 4 * r, 39 - 4 * q:79 - 4 * q] += g[bi]
                if mode == "sqrt":
                    FC[L][39 - 4 * r:79 - 4 * r, 39 - 4 * q:79 - 4 * q] += 1
                for nb in BUDGETS:
                    if ci < nb:
                        FTB[nb][mode][L][39 - 4 * r:79 - 4 * r, 39 - 4 * q:79 - 4 * q] += g[bi]
                        if mode == "sqrt":
                            FCB[nb][L][39 - 4 * r:79 - 4 * r, 39 - 4 * q:79 - 4 * q] += 1
    if DEV.type == "mps":
        torch.mps.empty_cache()
for mode in FT:
    for L in LAYERS:
        FT[mode][L] = FT[mode][L] / FC[L].clamp_min(1)
        for nb in BUDGETS:
            FTB[nb][mode][L] = FTB[nb][mode][L] / FCB[nb][L].clamp_min(1)
print(f"fault templates calibrated on {N_CAL} train slices ({len(CAL['L'])} simulated faults) t={time.time()-T0:.0f}s", flush=True)


# ------------------------------------------------------------------ fair counterpart: fine-tune both decoders on the SAME simulated faults
def finetune(kind, steps=int(os.environ.get("H1_FT_STEPS", 800)), bs=64, lr=3e-4, nmax=None):
    import copy
    import torch.nn.functional as F_
    model = copy.deepcopy(gdec if kind == "grad" else inv).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dyn = torch.cat(CAL["dyn"]); lr_ = torch.cat(CAL["logrms"]); gf = torch.cat(CAL["gf"])
    si = torch.tensor(CAL["si"]); Ls = torch.tensor(CAL["L"]); ks = torch.tensor(CAL["k"])
    X = torch.stack(CAL["X"]).to(DEV); PR = torch.stack(CAL["PROB"]).to(DEV); BM = torch.stack(CAL["BM"]).to(DEV)
    g = torch.Generator().manual_seed(5)
    pool = torch.arange(len(Ls)) if nmax is None else torch.nonzero(si < nmax).flatten()
    for st in range(steps):
        idx = pool[torch.randint(0, len(pool), (bs,), generator=g)]
        s_ = si[idx].to(DEV)
        if kind in ("loc", "loclog"):
            z = dec_in(dyn[idx].to(DEV).float(), PR[s_], X[s_])
            z[:, 13] = (lr_[idx].to(DEV) / 5.0).view(-1, 1, 1)                 # true log-magnitude channel
            if kind == "loclog":
                z[:, 4] = log_support(dyn[idx].to(DEV))
        else:
            z = grad_in(gf[idx].to(DEV), PR[s_], X[s_])
        with amp():
            out = model(z).float()
        out = out.view(bs, 9, NP * NP).masked_fill(~BM[s_][:, None, :], -1e9).flatten(1)
        lab, kk = Ls[idx].to(DEV), ks[idx].to(DEV)
        loss = F_.cross_entropy(out, lab * 100 + kk) + 0.5 * F_.cross_entropy(torch.logsumexp(out.view(bs, 9, -1), 2), lab)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    print(f"fine-tuned {kind} on {len(pool)} simulated faults, {steps} steps, last loss {float(loss):.3f} t={time.time()-T0:.0f}s", flush=True)
    return model


inv_ft, gdec_ft = finetune("loc"), finetune("grad")
inv_ftlog = finetune("loclog") if LOCLOG else None
gdec_b = {nb: finetune("grad", nmax=nb) for nb in BUDGETS}


def fault_scores(dy, ref, ks_all, mode, T=None):
    T = FT if T is None else T
    D = torch.cat([place(T[mode][L], ks_all) for L in LAYERS])
    return match(tf(e40(rob(dy), 1.0), mode), D)


# ------------------------------------------------------------------ choose the map transform on validation slices
N_VAL = int(os.environ.get("H1_NVAL", 20))
hits = {"sqrt": [], "log": []}
rng_v = np.random.RandomState(13)
for nm in pick_slices("val", N_VAL, seed=1):
    x, _ = load_slice(nm, "val")
    cache = forward(net, x[None]); lab0 = cache["logits"].argmax(1)
    ref = float((cache["logits"] ** 2).sum(1).mean())
    ks_all = torch.nonzero(brain_mask_patches(x)).flatten().cpu().numpy().tolist(); nK = len(ks_all)
    for li, L in enumerate(LAYERS):
        dl, meta = gen_faults(x, cache, lab0, L, ks_all, 3, 0, rng_v)
        if not dl:
            continue
        for mode in hits:
            s_ = fault_scores(torch.stack(dl), ref, ks_all, mode)
            for bi, (k, _) in enumerate(meta):
                jx = int(s_[bi].argmax()); hits[mode].append(jx // nK == li and ks_all[jx % nK] == k)
MODE = max(hits, key=lambda m: np.mean(hits[m]))
VAL_SEL = {m: float(np.mean(v)) for m, v in hits.items()}
print("validation joint acc by transform", VAL_SEL, "-> use", MODE, flush=True)

rng = np.random.RandomState(7)
rs_v = np.random.RandomState(31)                          # re-simulation draws (separate random stream)
TIME = {"Response model (fault-cal.)": 0.0, "Grad+training (fault-ft.)": 0.0, "re-simulation (M=64)": 0.0}
NFWD = []


def sync():
    if DEV.type == "mps":
        torch.mps.synchronize()
rows, stats = [], {L: {"tried": 0, "masked": 0, "crash": 0, "visible": 0} for L in LAYERS}
bitstat = np.zeros((32, 3), int)                           # per bit: masked / crash / visible
for si, nm in enumerate(test_names):
    x, _ = load_slice(nm, "test")
    cache = forward(net, x[None])
    lab0 = cache["logits"].argmax(1)
    ref = float((cache["logits"] ** 2).sum(1).mean())
    prob = cache["logits"].softmax(1)
    bm = brain_mask_patches(x)
    ks_all = torch.nonzero(bm).flatten().cpu().numpy().tolist()
    nK = len(ks_all)
    D_mix = {t: torch.cat([place(TM[(L, t)], ks_all) for L in LAYERS]) for t in TRAIN_TYPES}
    for li, L in enumerate(LAYERS):
        got_d, got_meta = gen_faults(x, cache, lab0, L, ks_all, PER_LAYER, 0, rng, stats, bitstat)
        if not got_meta:
            continue
        dy = torch.stack(got_d)
        sc = {}
        with torch.no_grad(), amp():
            sc["Localisation net"] = inv(dec_in64(dy, prob, x)).float()
        obs = e40(rob(dy), 1.0)
        pm = torch.stack([cos_amp(obs, D_mix[t]) for t in TRAIN_TYPES]).max(0).values      # (B, 9*nK) over ks_all
        sync(); t_g = time.time()
        S = grad_attributions(net, x, cache, rob(dy), ig_steps=0)
        gin = grad_in(grad_feats(S), prob.expand(len(dy), -1, -1, -1), x[None].expand(len(dy), -1, -1, -1))
        with torch.no_grad(), amp():
            sc["Grad+training"] = gdec(gin).float()
            sc["Grad+training (fault-ft.)"] = gdec_ft(gin).float()
            sync(); TIME["Grad+training (fault-ft.)"] += time.time() - t_g
            sc["Localisation net (fault-ft.)"] = inv_ft(dec_in64(dy, prob, x)).float()
            if LOCLOG:
                sc["Localisation net (fault-ft., log input)"] = inv_ftlog(dec_in_log(dy, prob, x)).float()
        for key in ("Localisation net", "Grad+training", "Localisation net (fault-ft.)", "Grad+training (fault-ft.)") + (("Localisation net (fault-ft., log input)",) if LOCLOG else ()):                 # (B,9,100) -> restrict to brain patches
            sc[key] = sc[key].view(-1, 9, NP * NP)[:, :, ks_all].flatten(1)
        sc["Response model"] = pm
        sync(); t_ = time.time()
        sc["Response model (fault-cal.)"] = fault_scores(dy, ref, ks_all, MODE)
        sync(); TIME["Response model (fault-cal.)"] += time.time() - t_
        for nb in BUDGETS:
            sc[f"Response model (fault-cal., n={nb})"] = fault_scores(dy, ref, ks_all, MODE, FTB[nb])
            with torch.no_grad(), amp():
                sc[f"Grad+training (fault-ft., n={nb})"] = gdec_b[nb](gin).float().view(-1, 9, NP * NP)[:, :, ks_all].flatten(1)
        if VERIFY:
            base = sc["Response model (fault-cal.)"].view(len(dy), 9, nK)
            ver = {m: torch.full_like(sc["Response model (fault-cal.)"], -1e9) for m in VMS}
            for bi in range(len(dy)):
                sync(); t_ = time.time()
                lv, kv = base[bi].max(1)                                     # best patch per layer
                obs_l = log_support(rob(dy[bi:bi + 1]))
                nf = 0
                for lc in torch.topk(lv, VK).indices.tolist():
                    sims = sim_impulse(x, cache, LAYERS[lc], ks_all[int(kv[lc])], max(VMS), rs_v)
                    nf += max(VMS)
                    if sims is None:
                        continue
                    cl = cosm(obs_l, log_support(rob(sims)))[0]
                    for m in VMS:
                        c_ = cl[:m]
                        ver[m][bi, lc * nK + int(kv[lc])] = c_.topk(min(3, len(c_))).values.mean()
                sync(); TIME["re-simulation (M=64)"] += time.time() - t_
                NFWD.append(nf)
            for m in VMS:
                sc["Response model + re-simulation (M=%d)" % m] = ver[m]
        for g, lab in G_NAMES.items():
            sc[lab] = normalise(S[g], norm_choice[g], calib[g])[:, :, ks_all].flatten(1)
        for bi, (kk, bb) in enumerate(got_meta):
            row = {"si": si, "L": li, "k": kk, "bit": bb, "log10_rms": float(torch.log10(dy[bi].cpu().double().pow(2).mean().sqrt())),
                   "npix": int((cache["logits"].argmax(1)[0] != (cache["logits"][0] + dy[bi]).argmax(0)).sum())}
            for m in METHODS:
                jx = int(sc[m][bi].argmax())
                row[m] = (jx // nK, ks_all[jx % nK])
            rows.append(row)
    if DEV.type == "mps":
        torch.mps.empty_cache()
    if si % 10 == 0:
        print(f"{si}/{len(test_names)} rows={len(rows)} t={time.time()-T0:.0f}s", flush=True)


def boot(vals_by_pat, B=2000):
    rng_ = np.random.RandomState(0); pats = sorted(vals_by_pat)
    allv = np.concatenate([vals_by_pat[p] for p in pats])
    bs = []
    for _ in range(B):
        pick = rng_.choice(pats, len(pats))
        bs.append(np.mean(np.concatenate([vals_by_pat[p] for p in pick])))
    return [float(allv.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def acc(m, kind, rr):
    out = {}
    for r in rr:
        hit = (r[m][0] == r["L"]) if kind == "layer" else (r[m][0] == r["L"] and r[m][1] == r["k"])
        out.setdefault(r["si"], []).append(float(hit))
    return {p: np.array(v) for p, v in out.items()}


summ = {"val_transform_acc": VAL_SEL, "transform": MODE, "n_visible": len(rows), "per_layer_counts": stats,
        "bit_outcome_share": {int(b): (bitstat[b] / max(1, bitstat[b].sum())).round(3).tolist() for b in range(32)}}
for m in METHODS:
    summ[m] = {"joint": boot(acc(m, "joint", rows)), "layer": boot(acc(m, "layer", rows)),
               "per_layer_joint": [float(np.mean([r[m][0] == r["L"] and r[m][1] == r["k"] for r in rows if r["L"] == li]))
                                   if any(r["L"] == li for r in rows) else None for li in range(9)]}


def diff(a, b_, kind="joint"):
    A, B_ = acc(a, kind, rows), acc(b_, kind, rows)
    return boot({p: A[p] - B_[p] for p in A})


summ["Localisation net - Grad+training"] = diff("Localisation net", "Grad+training")
summ["Fault-cal. RM - Grad+training"] = diff("Response model (fault-cal.)", "Grad+training")
summ["Fault-cal. RM - Grad×Act"] = diff("Response model (fault-cal.)", "Grad×Act")
summ["Fault-cal. RM - LocNet (fault-ft.)"] = diff("Response model (fault-cal.)", "Localisation net (fault-ft.)")
summ["Fault-cal. RM - Grad+training (fault-ft.)"] = diff("Response model (fault-cal.)", "Grad+training (fault-ft.)")
summ["Response model - best untrained grad"] = {g: diff("Response model", g) for g in G_NAMES.values()}
if LOCLOG:
    summ["LocNet log input (fault-ft.) - Grad+training (fault-ft.)"] = diff("Localisation net (fault-ft., log input)", "Grad+training (fault-ft.)")
    summ["LocNet log input (fault-ft.) - Fault-cal. RM"] = diff("Localisation net (fault-ft., log input)", "Response model (fault-cal.)")
    print("LocNet log (ft) - Grad(ft)", [round(v, 3) for v in summ["LocNet log input (fault-ft.) - Grad+training (fault-ft.)"]],
          " LocNet log (ft) - FaultCal RM", [round(v, 3) for v in summ["LocNet log input (fault-ft.) - Fault-cal. RM"]])
summ["seconds_per_query"] = {k: v / max(1, len(rows)) for k, v in TIME.items()}
if VERIFY:
    summ["re-simulation forwards per query"] = float(np.mean(NFWD))
    for m in VMS:
        summ[f"RM+resim(M={m}) - Grad+training (fault-ft.)"] = diff("Response model + re-simulation (M=%d)" % m, "Grad+training (fault-ft.)")
        summ[f"RM+resim(M={m}) - Fault-cal. RM"] = diff("Response model + re-simulation (M=%d)" % m, "Response model (fault-cal.)")
    print("RM+resim(M=64) - Grad(ft)", [round(v, 3) for v in summ["RM+resim(M=64) - Grad+training (fault-ft.)"]],
          " - RM", [round(v, 3) for v in summ["RM+resim(M=64) - Fault-cal. RM"]], " forwards/query", summ["re-simulation forwards per query"])
for nb in BUDGETS:
    summ[f"RM(n={nb}) - Grad(ft, n={nb})"] = diff(f"Response model (fault-cal., n={nb})", f"Grad+training (fault-ft., n={nb})")
    print(f"budget n={nb}: RM - Grad(ft)", [round(v, 3) for v in summ[f"RM(n={nb}) - Grad(ft, n={nb})"]])
print("seconds per query", {k: round(v, 4) for k, v in summ["seconds_per_query"].items()})
save_json({"summary": summ, "rows": rows}, f"{OUT}/h1_summary.json")
print("visible faults", len(rows), {L: v for L, v in stats.items()})
for m in METHODS:
    print(f"{m:18s} joint {summ[m]['joint'][0]:.3f} [{summ[m]['joint'][1]:.3f},{summ[m]['joint'][2]:.3f}]  layer {summ[m]['layer'][0]:.3f}")
print("LocNet - Grad+training", [round(v, 3) for v in summ["Localisation net - Grad+training"]],
      " FaultCal RM - Grad+training", [round(v, 3) for v in summ["Fault-cal. RM - Grad+training"]])
print("FaultCal RM - LocNet(ft)", [round(v, 3) for v in summ["Fault-cal. RM - LocNet (fault-ft.)"]],
      " FaultCal RM - Grad(ft)", [round(v, 3) for v in summ["Fault-cal. RM - Grad+training (fault-ft.)"]])
print(f"done t={time.time()-T0:.0f}s")
