# -*- coding: utf-8 -*-
"""Attribution-patching (AtP, noise reference) on the off-grid queries of any_position.py.

The queries are identical, one by one, to the offset16 / rand8 test sets of any_position.py (same
generator functions and random seeds). The layer-wise normalisation of AtP is the one selected on the
tune set in the main localization experiment. Metrics as in any_position.py: localization accuracy (grid
patch containing the box centre), layer accuracy, and overlap with the intervened box.
Reads results/base/e6_rows.pkl; writes results/any_position/atp_offgrid.json.
Run: python experiments/ablation/any_position_atp.py   (GPU_FRAC limits GPU memory)
"""
import os, pickle, time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.engine import P, NP
from bia.core import TEST_TYPES, brain_mask_patches, run_delta, grad_attributions, normalise

if DEV.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.15")))
OUT = set_out("any_position")
BASE_DIR = os.path.join(RESULTS, "base")
T0 = time.time()
net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)
leg = pickle.load(open(f"{BASE_DIR}/e6_rows.pkl", "rb"))
norm, calib = leg["norm"]["gsig"], leg["calib"]["gsig"]
used = set(pick_slices("test", 40, seed=2))
test_names = [("test", n) for n in pick_slices("test", 243, seed=2) if n not in used][:60]
CONDS = [("aligned16", 16, False), ("offset16", 16, True), ("rand8", 8, True)]


def frac_masks(L, boxes):
    s = STRIDE[L]
    m = torch.zeros(len(boxes), 1, GRID, GRID, device=DEV)
    for b, (y0, x0, p) in enumerate(boxes):
        m[b, 0, y0:y0 + p, x0:x0 + p] = 1
    return F.avg_pool2d(m, s) if s > 1 else m


def s2_delta(hc, L, boxes, t, gen):
    B = len(boxes); m = frac_masks(L, boxes)
    sig = hc.std(dim=(2, 3), keepdim=True); mu = hc.mean(dim=(2, 3), keepdim=True)
    h = hc.expand(B, *hc.shape[1:]); C = hc.shape[1]
    eps = torch.randn(h.shape, generator=gen, device=DEV)
    if t == "noise":
        d = sig * eps
    elif t == "ablate":
        d = mu - h
    elif t == "gain":
        d = h.clone()
    else:
        cm = (torch.rand(B, C, 1, 1, generator=gen, device=DEV) < 0.25).float()
        d = 2.0 * sig * eps * cm
    return d * m


def sample_boxes(rng, p, offset, ks_all, n):
    out = []
    while len(out) < n:
        if not offset:
            k = int(rng.choice(ks_all)); r, q = divmod(k, NP)
            out.append(((r * P, q * P, p), k)); continue
        y0, x0 = rng.randint(0, GRID - p + 1), rng.randint(0, GRID - p + 1)
        cy, cx = y0 + p / 2.0, x0 + p / 2.0
        k = int(min(cy // P, NP - 1)) * NP + int(min(cx // P, NP - 1))
        if k in ks_all:
            out.append(((y0, x0, p), k))
    return out


def touch_of(box):
    y0, x0, pp = box
    return {(yy // P) * NP + xx // P for yy in range(y0, y0 + pp) for xx in range(x0, x0 + pp)}


def boot(c, si, n=1000, seed=0):
    r = np.random.RandomState(seed); pats = np.unique(si); by = {p: c[si == p] for p in pats}
    v = [np.concatenate([by[p] for p in r.choice(pats, len(pats))]).mean() for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


res = {}
gen = torch.Generator(device=DEV)
for ci in (1, 2):
    cond, p, offset = CONDS[ci]
    J, Ly, Ov, SI = [], [], [], []
    for si, (split, nm) in enumerate(test_names):
        x, _ = load_slice(nm, split); cache = forward(net, x[None]); bm = brain_mask_patches(x)
        ks_all = torch.nonzero(bm).flatten().cpu().numpy().tolist(); nK = len(ks_all)
        rng = np.random.RandomState(1000 * si + ci)
        for t in TEST_TYPES:
            for li, L in enumerate(LAYERS):
                bx = sample_boxes(rng, p, offset, ks_all, 4)
                gen.manual_seed(int(rng.randint(1 << 30)))
                d = s2_delta(cache[L], L, [b for b, _ in bx], t, gen)
                dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
                keep = (dy.pow(2).mean((1, 2, 3)).sqrt() > 1e-6).cpu().numpy()
                S = grad_attributions(net, x, cache, dy, ig_steps=0)["gsig"]
                v = normalise(S, norm, calib)[:, :, ks_all].flatten(1)
                for b in range(4):
                    if not keep[b]:
                        continue
                    j = int(v[b].argmax()); pl, pk = j // nK, ks_all[j % nK]
                    J.append(pl == li and pk == bx[b][1]); Ly.append(pl == li)
                    Ov.append(pl == li and pk in touch_of(bx[b][0])); SI.append(si)
    J, Ly, Ov = (np.array(a, float) for a in (J, Ly, Ov)); SI = np.array(SI)
    res[cond] = {"joint": float(J.mean()), "joint_ci": boot(J, SI), "layer": float(Ly.mean()),
                 "overlap": float(Ov.mean()), "overlap_ci": boot(Ov, SI), "n": int(len(J))}
    print(cond, res[cond], f"t={time.time()-T0:.0f}s", flush=True)
save_json(res, os.path.join(OUT, "atp_offgrid.json"))
print("done", flush=True)
