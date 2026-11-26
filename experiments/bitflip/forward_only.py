# -*- coding: utf-8 -*-
"""Forward-only controls for bit-flip localization on the same faults as bitflip.py (Table 4).

Same fault model, same 60 test patients and same random streams as bitflip.py (at most 5 visible faults per
layer and patient). Methods compared on the identical faults:
  fault-calibrated prototype matching   log-energy matching against fault prototypes (80 training slices,
                                        exponent-bit flips); no network run
  prototype matching + forward re-check top-3 layers by prototype score, best patch of each, M=64 simulated
                                        flips per candidate (192 forward passes)
  AtP candidates + forward re-check     candidates from AtP, identical re-check and forward count
  fault dictionary (exhaustive)         M_d simulated flips for every brain patch of every layer, most similar
                                        entry wins (about 509 x M_d forward passes per query)
Re-check and dictionary share one similarity: cosine between the observed and simulated log-energy maps,
averaged over the 3 most similar simulations.

Reads results/base/e6_rows.pkl; writes results/bitflip_forward/h2_summary.json
(per-patient checkpoints h2_partial_<i>of<n>.json; a restarted run resumes from them).
Run: CUDNN_OFF=1 python experiments/bitflip/forward_only.py
Environment: H2_OUT, H2_NPAT, GPU_FRAC, CUDNN_OFF (flipped activations reach ~1e19, so cuDNN fast
convolution algorithms are disabled), H2_SHARD=i/n (process only patients with index % n == i).
"""
import os, json, pickle, time
import numpy as np
import torch
from bia.common import *
from bia.engine import P, NP
from bia.core import brain_mask_patches, run_delta, grad_attributions, normalise, e40

if DEV.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.3")))
    if os.environ.get("CUDNN_OFF") == "1":      # direct convolution avoids rounding error of fast algorithms when the tail is ~1e-8 of the peak
        torch.backends.cudnn.enabled = False
OUT_NAME = os.environ.get("H2_OUT", "bitflip_forward")
OUT = set_out(os.environ.get("H2_OUT", "bitflip_forward"))
BASE_DIR = os.path.join(RESULTS, "base")
T0 = time.time()
N_PAT = int(os.environ.get("H2_NPAT", 60))
PER_LAYER, TRY_BATCH, MAX_TRIES, VK, VM = 5, 64, 6, 3, 64
MDS = (16, 64)
net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)
leg = pickle.load(open(f"{BASE_DIR}/e6_rows.pkl", "rb"))
norm_choice, calib = leg["norm"], leg["calib"]
test_names = leg["test"][:N_PAT]


def log(*a):
    print(*a, f"t={time.time()-T0:.0f}s", flush=True)


# ------------------------------------------------------------------ identical to bitflip.py
def place(T, ks):
    return torch.stack([T[39 - 4 * (k // NP):79 - 4 * (k // NP), 39 - 4 * (k % NP):79 - 4 * (k % NP)] for k in ks])


def flip(vals, bits):
    iv = vals.detach().cpu().float().contiguous().view(torch.int32)
    m = torch.tensor([(1 << int(b)) if b < 31 else -(1 << 31) for b in bits], dtype=torch.int32)
    return torch.bitwise_xor(iv, m).view(torch.float32)


def gen_faults(x, cache, lab0, L, ks_all, n_target, bit_lo, rng):
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
        for t_ in torch.nonzero(vis).flatten().tolist():
            if len(got_meta) < n_target:
                got_d.append(lo[t_] - cache["logits"][0])
                got_meta.append((int(k[t_]), int(b[t_])))
    return got_d, got_meta


def rob(dy):
    return dy / dy.abs().flatten(1).amax(1).clamp_min(1e-30).view(-1, 1, 1, 1)


def log_support(dyr):
    e = dyr.float().pow(2).sum(1)
    mx = e.flatten(1).max(1).values.clamp_min(1e-30).view(-1, 1, 1)
    return ((torch.log10(e.clamp_min(1e-30) / mx) + 8.0).clamp_min(0) / 8.0)


def sim_impulse(x, cache, L, k, M, rs):
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


def tf_log(e):
    mx = e.flatten(1).max(1).values.clamp_min(1e-30).view(-1, 1, 1)
    return (torch.log10(e.clamp_min(1e-30) / mx) + 8.0).clamp_min(0)


def match(a, b):
    a = a.flatten(1).to(HI); b = b.flatten(1).to(HI)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-30); b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-30)
    return a @ b.T


# ------------------------------------------------------------------ fault prototypes (same as bitflip.py: 80 train slices, rng 11)
FT = {L: torch.zeros(79, 79, device=DEV) for L in LAYERS}
FC = {L: torch.zeros(79, 79, device=DEV) for L in LAYERS}
rng_c = np.random.RandomState(11)
for ci, nm in enumerate(pick_slices("train", 80, seed=3)):
    x, _ = load_slice(nm, "train")
    cache = forward(net, x[None]); lab0 = cache["logits"].argmax(1)
    ks_all = torch.nonzero(brain_mask_patches(x)).flatten().cpu().numpy().tolist()
    for li, L in enumerate(LAYERS):
        dl, meta = gen_faults(x, cache, lab0, L, ks_all, 6, 23, rng_c)
        if not dl:
            continue
        g = tf_log(e40(rob(torch.stack(dl)), 1.0)); g = g / g.flatten(1).norm(dim=1).clamp_min(1e-30).view(-1, 1, 1)
        for bi, (k, _) in enumerate(meta):
            r, q = divmod(k, NP)
            FT[L][39 - 4 * r:79 - 4 * r, 39 - 4 * q:79 - 4 * q] += g[bi]
            FC[L][39 - 4 * r:79 - 4 * r, 39 - 4 * q:79 - 4 * q] += 1
for L in LAYERS:
    FT[L] = FT[L] / FC[L].clamp_min(1)
log("fault prototypes calibrated")


def recheck(x, cache, base, obs_l, ks_all, rs):
    """base (9, nK) candidate scores -> verified scores over (9*nK) (top-3 layers, best patch each, VM flips)."""
    nK = len(ks_all)
    out = torch.full((9 * nK,), -1e9, device=DEV, dtype=torch.float64)
    lv, kv = base.max(1)
    for lc in torch.topk(lv, VK).indices.tolist():
        sims = sim_impulse(x, cache, LAYERS[lc], ks_all[int(kv[lc])], VM, rs)
        if sims is None:
            continue
        c_ = cosm(obs_l, log_support(rob(sims)))[0]
        out[lc * nK + int(kv[lc])] = float(c_.topk(min(3, len(c_))).values.mean())
    return out


# ------------------------------------------------------------------ test faults (rng 7, as in bitflip.py) and all methods
# Sharding: with H2_SHARD=i/n every process still generates the faults of all patients in order (keeps the rng 7
# stream identical to bitflip.py) but evaluates the methods only for patients with si % n == i. Simulation draws
# are seeded per patient, so the result does not depend on sharding.
SH_I, SH_N = (int(v) for v in os.environ.get("H2_SHARD", "0/1").split("/"))
rng = np.random.RandomState(7)
# Resume: h2_partial_{i}of{n}.json is rewritten after every patient; a restart skips the finished patients.
PART = os.path.join(OUT, f"h2_partial_{SH_I}of{SH_N}.json")
if os.path.exists(PART):
    _p = json.load(open(PART)); rows, done = _p["rows"], set(_p["done"])
    log(f"resume: {len(done)} patients, {len(rows)} rows")
else:
    rows, done = [], set()
for si, nm in enumerate(test_names):
    x, _ = load_slice(nm, "test")
    cache = forward(net, x[None]); lab0 = cache["logits"].argmax(1)
    ks_all = torch.nonzero(brain_mask_patches(x)).flatten().cpu().numpy().tolist(); nK = len(ks_all)
    faults = []
    for li, L in enumerate(LAYERS):
        dl, meta = gen_faults(x, cache, lab0, L, ks_all, PER_LAYER, 0, rng)
        faults += [(li, k, d_) for d_, (k, _) in zip(dl, meta)]
    if not faults or si % SH_N != SH_I or si in done:
        continue
    rs_v, rs_a, rs_d = (np.random.RandomState(base + 1000 * si) for base in (31, 37, 41))
    D = torch.cat([place(FT[L], ks_all) for L in LAYERS])
    dy = torch.stack([f[2] for f in faults]); dyr = rob(dy)
    obs_l = log_support(dyr)                                            # (N,160,160) for re-simulation matching
    pm = match(tf_log(e40(dyr, 1.0)), D)                                # (N, 9*nK)
    G = torch.cat([grad_attributions(net, x, cache, dyr[c0:c0 + 12], ig_steps=0)["gsig"]
                   for c0 in range(0, len(faults), 12)])                    # chunked; per-query maps do not depend on the chunking
    atp = normalise(G, norm_choice["gsig"], calib["gsig"])[:, :, ks_all]   # (N,9,nK)
    sc = {"BIA prototype matching (fault-calibrated)": pm,
          "BIA prototype matching + forward re-check": torch.stack([recheck(x, cache, pm[b].view(9, nK), obs_l[b:b + 1], ks_all, rs_v)
                                          for b in range(len(faults))]),
          "AtP candidates + forward re-check": torch.stack([recheck(x, cache, atp[b].to(HI), obs_l[b:b + 1], ks_all, rs_a)
                                        for b in range(len(faults))])}
    # fault dictionary: every brain patch of every layer, M_d simulated flips each
    dic = {md: torch.full((len(faults), 9 * nK), -1e9, device=DEV, dtype=torch.float64) for md in MDS}
    for li, L in enumerate(LAYERS):
        for kj, k in enumerate(ks_all):
            sims = sim_impulse(x, cache, L, k, max(MDS), rs_d)
            if sims is None:
                continue
            cl = cosm(obs_l, log_support(rob(sims)))                     # (N, m)
            for md in MDS:
                c_ = cl[:, :md]
                dic[md][:, li * nK + kj] = c_.topk(min(3, c_.shape[1]), dim=1).values.mean(1).double()
    for md in MDS:
        sc[f"fault dictionary (M_d={md})"] = dic[md]
    for b, (li, k, _) in enumerate(faults):
        row = {"si": si, "L": li, "k": k}
        for key, v in sc.items():
            j = int(v[b].argmax()); row[key] = (j // nK, ks_all[j % nK])
        rows.append(row)
    done.add(si); save_json({"done": sorted(done), "rows": rows}, PART)
    log(f"{si} {nm} faults={len(faults)} rows={len(rows)}")


def boot(c, si, n=2000, seed=0):
    r = np.random.RandomState(seed); pats = np.unique(si); by = {p: c[si == p] for p in pats}
    v = [np.concatenate([by[p] for p in r.choice(pats, len(pats))]).mean() for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


si_arr = np.array([r["si"] for r in rows])
summ = {"n_visible": len(rows), "forwards_per_query": {"BIA prototype matching + forward re-check": VK * VM, "AtP candidates + forward re-check": VK * VM,
                                                         **{f"fault dictionary (M_d={md})": f"{md} x 9 x brain patches" for md in MDS}},
        "methods": {}}
for key in [k for k in rows[0] if k not in ("si", "L", "k")]:
    J = np.array([r[key][0] == r["L"] and r[key][1] == r["k"] for r in rows], float)
    Ly = np.array([r[key][0] == r["L"] for r in rows], float)
    summ["methods"][key] = {"joint": float(J.mean()), "joint_ci": boot(J, si_arr), "layer": float(Ly.mean())}
    print(f"{key:20s} joint {J.mean():.3f} {summ['methods'][key]['joint_ci']} layer {Ly.mean():.3f}", flush=True)
save_json({"summary": summ, "rows": rows, "shard": [SH_I, SH_N]},
          os.path.join(OUT, "h2_summary.json" if SH_N == 1 else f"h2_shard{SH_I}of{SH_N}.json"))
log("done")
