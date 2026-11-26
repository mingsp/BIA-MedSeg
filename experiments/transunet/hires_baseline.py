# -*- coding: utf-8 -*-
"""Seg-HiRes-Grad CAM baseline on TransUNet (Table 3), on exactly the test queries of run.py.

Rebuilds only the tune / test query sets of run.py (identical RNG use), computes ReLU(sum_c g_c * h_c)
per layer and patch for the observed output change (objective J = <dy, logits>), chooses the layer
normalisation on the tune queries (same rule as in the main localization experiment) and scores the test
queries. LRP is not run on TransUNet: its standard rules do not cover self-attention.
Writes results/transunet/t1b_hires.json.
Run: python experiments/transunet/hires_baseline.py   (GPU_FRAC limits GPU memory)
"""
import bia.transunet  # noqa: F401  (must come first: rebinds the layer list and forward pass)
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.core import TEST_TYPES, brain_mask_patches, make_delta, run_delta, patch_sum, normalise, fwd_grad

if DEV.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.3")))
OUT = set_out("transunet")
T0 = time.time()
net = load_model("transunet", 123)
for p in net.parameters():
    p.requires_grad_(False)
e2_test = pick_slices("test", 40, seed=2)
val = pick_slices("val", 124, seed=1)
tune_names = [("val", n) for n in val[100:124]]
test_names = [("test", n) for n in pick_slices("test", 243, seed=2) if n not in set(e2_test)][:60]


def hires(x, dy):
    hs = {}

    def keep(L):
        def fn(h):
            h.retain_grad(); hs[L] = h
            return h
        return fn
    with torch.enable_grad():
        f = fwd_grad(net, x[None].expand(dy.shape[0], *x.shape).clone().requires_grad_(True),
                     edits={L: keep(L) for L in LAYERS})
        (f["logits"] * dy).sum().backward()
    return torch.stack([patch_sum(F.relu((hs[L].grad * hs[L]).sum(1)), L) for L in LAYERS], 1).to(HI).detach()


@torch.no_grad()
def queries(names, seed0):
    """Same RNG consumption as build() in run.py (types_mode="each"); yields per-slice query batches."""
    rng = np.random.RandomState(seed0)
    gen = torch.Generator(device=DEV)
    for si, (split, nm) in enumerate(names):
        x, _ = load_slice(nm, split)
        cache = forward(net, x[None])
        bm = brain_mask_patches(x)
        ks_all = torch.nonzero(bm).flatten().cpu().numpy()
        for t in TEST_TYPES:
            for li in range(9):
                L = LAYERS[li]
                ks = rng.choice(ks_all, 4)
                gen.manual_seed(int(rng.randint(1 << 30)))
                d = make_delta(cache[L], L, ks, [t] * 4, gen)
                dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
                keep = (dy.pow(2).mean((1, 2, 3)).sqrt() > 1e-6).cpu().numpy()
                yield si, x, bm, ks_all, t, li, ks, dy, keep


# tune: choose the layer normalisation
calib_max, tune = [[] for _ in range(9)], []
for si, x, bm, ks_all, t, li, ks, dy, keep in queries(tune_names, 123):
    S = hires(x, dy)
    for l in range(9):
        calib_max[l] += S[:, l].max(1).values.cpu().tolist()
    tune.append((S.cpu(), bm.cpu(), li, keep))
calib = [float(np.median(v)) for v in calib_max]
acc = {}
for mode in ("raw", "sum", "calib"):
    a = []
    for S, bm, li, keep in tune:
        j = normalise(S, mode, calib).masked_fill(~bm[None, None], -1e30).flatten(1).argmax(1).numpy()
        a += [jj // 100 == li for jj, k in zip(j, keep) if k]
    acc[mode] = float(np.mean(a))
mode = max(acc, key=acc.get)
print("tune layer acc", acc, f"t={time.time()-T0:.0f}s", flush=True)

# test
rows = []
for si, x, bm, ks_all, t, li, ks, dy, keep in queries(test_names, 2026):
    ks_l = ks_all.tolist(); nK = len(ks_l)
    v = normalise(hires(x, dy), mode, calib)[:, :, ks_l].flatten(1)
    for b in range(4):
        if keep[b]:
            j = int(v[b].argmax())
            rows.append((si, TEST_TYPES.index(t), li, int(ks[b]), j // nK, ks_l[j % nK]))
r = np.array(rows)
J = (r[:, 2] == r[:, 4]) & (r[:, 3] == r[:, 5])


def boot(c, si, n=1000, seed=0):
    rs = np.random.RandomState(seed); pats = np.unique(si); by = {p: c[si == p] for p in pats}
    v = [np.concatenate([by[p] for p in rs.choice(pats, len(pats))]).mean() for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


out = {"method": "Seg-HiRes-Grad CAM", "norm": mode, "tune_layer_acc": acc, "n_rows": len(rows),
       "joint": float(J.mean()), "joint_ci": boot(J.astype(float), r[:, 0]), "layer": float((r[:, 2] == r[:, 4]).mean()),
       **{t: float(J[r[:, 1] == i].mean()) for i, t in enumerate(TEST_TYPES)},
       "per_layer_joint": [float(J[r[:, 2] == l].mean()) for l in range(9)]}
save_json(out, os.path.join(OUT, "t1b_hires.json"))
print(json.dumps(out), f"t={time.time()-T0:.0f}s", flush=True)
