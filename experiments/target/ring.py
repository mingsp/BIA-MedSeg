"""Target-driven layer choice: enlarge the segmentation outward by a ring (Fig. 13a).

Task: on 40 test slices, grow the predicted tumour by a ring of width w = 4 / 12 / 24 px just outside it. The edit
is the gradient direction of the in-ring tumour margin on all patches covering the ring, with norm ||h_patch||;
every candidate layer is executed on the real network. up4 is excluded (only a 1x1 conv follows, so its
selectivity is trivially ~1), leaving 8 candidate layers.
Real outcome: selectivity = share of tumour-margin change energy inside the ring; new_in / new_out pixels.
Selection rules (the layer is chosen before running the edit):
  VE         share of the response-model footprint (noise prototypes, summed over ring patches) inside the
             ring; 0 extra passes
  AtP-sel    first-order g_in.d / (|g_in.d| + |g_out.d|), with g_in / g_out the gradients of the in-ring /
             out-of-ring tumour margin; 2 backward passes
  GradMax    largest first-order in-ring gain g_in.d; 1 backward pass
  Random, and Oracle (best real selectivity, 8 forward passes)

Reads results/base/e6_templates.npz; writes results/ring/e5c_summary.json.
Run: python experiments/target/ring.py
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from bia.common import *
from bia.engine import P, NP, cell_mask
from bia.core import fwd_grad

OUT = set_out("ring")
T0 = time.time()
net = load_model("baseline", 123)
for p_ in net.parameters():
    p_.requires_grad_(False)
tm = np.load(base_path("e6_templates.npz"))
T = {L: torch.tensor(tm[f"{L}__noise"], device=DEV, dtype=torch.float32) for L in LAYERS}
CAND = LAYERS[:-1]


def dilate(m, w):
    return F.max_pool2d(m[None, None].float(), 2 * w + 1, 1, w)[0, 0] > 0


def score(logits):
    return torch.logsumexp(logits[:, 1:], 1) - logits[:, 0]


def footprint(L, ks):
    acc = torch.zeros(40, 40, device=DEV)
    for k in ks:
        r, q = divmod(k, NP)
        t = T[L][39 - 4 * r:79 - 4 * r, 39 - 4 * q:79 - 4 * q]
        acc += t / t.sum().clamp_min(1e-30)
    return acc


def grads(x, objs):
    out = []
    for fo in objs:
        hs = {}

        def keep(L):
            def fn(h):
                h.retain_grad(); hs[L] = h
                return h
            return fn
        with torch.enable_grad():
            xg = x[None].clone().requires_grad_(True)
            f = fwd_grad(net, xg, edits={L: keep(L) for L in LAYERS})
            fo(f["logits"]).backward()
        out.append({L: hs[L].grad.detach() for L in LAYERS})
    return out


names = pick_slices("test", 40, seed=2)
rows = []
for si, nm in enumerate(names):
    x, y = load_slice(nm, "test")
    cache = forward(net, x[None])
    pred0 = cache["logits"].argmax(1)[0] > 0
    s0 = score(cache["logits"])[0]
    brain = x.abs().sum(0) > 1e-6
    for w in (4, 12, 24):
        ring = dilate(pred0, w) & ~pred0 & brain
        if ring.sum() < 50:
            continue
        outside = brain & ~ring & ~pred0
        ks = [int(k) for k in torch.nonzero(ring.float().view(NP, P, NP, P).sum((1, 3)).flatten() > 0).flatten()]
        ring40 = F.avg_pool2d(ring[None, None].float(), 4)[0, 0]
        gin, gout = grads(x, [lambda lg: (score(lg)[0] * ring).sum(), lambda lg: (score(lg)[0] * outside).sum()])
        r = {"si": si, "w": w, "VE": {}, "AtP-sel": {}, "GradMax": {}, "real": {}}
        for L in CAND:
            m = cell_mask(L, ks)[0]
            gd = gin[L] * m
            d = (cache[L] * m).norm() * gd / gd.norm().clamp_min(1e-30)
            fp = footprint(L, ks)
            r["VE"][L] = float((fp * ring40).sum() / fp.sum().clamp_min(1e-30))
            a, b = float((gin[L] * d).sum()), float((gout[L] * d).sum())
            r["AtP-sel"][L] = a / max(abs(a) + abs(b), 1e-30)
            r["GradMax"][L] = a
            out = forward(net, x[None], edits={L: (lambda hh, d=d: hh + d)}, cache=cache)
            ds = score(out["logits"])[0] - s0
            p1 = out["logits"].argmax(1)[0] > 0
            e = ds ** 2
            r["real"][L] = {"sel": float(e[ring].sum() / e.sum().clamp_min(1e-30)),
                            "new_in": int((p1 & ~pred0 & ring).sum()), "new_out": int((p1 & ~pred0 & ~ring).sum()),
                            "lost": int((pred0 & ~p1).sum())}
        rows.append(r)
    if si % 10 == 0:
        print(f"{si}/40 t={time.time()-T0:.0f}s", flush=True)

rng = np.random.RandomState(3)
summ = {}
for w in (4, 12, 24):
    R = [r for r in rows if r["w"] == w]
    d = {"n": len(R)}
    for mth in ("VE", "AtP-sel", "GradMax"):
        pick = [max(CAND, key=lambda L: r[mth][L]) for r in R]
        rho = [spearmanr([r[mth][L] for L in CAND], [r["real"][L]["sel"] for L in CAND])[0] for r in R]
        d[mth] = {k: float(np.mean([r["real"][L][k] for r, L in zip(R, pick)])) for k in ("sel", "new_in", "new_out", "lost")}
        d[mth]["spearman_mean"] = float(np.nanmean(rho))
        d[mth]["layer_hist"] = [pick.count(L) for L in CAND]
        d[mth]["_sel_vec"] = [r["real"][L]["sel"] for r, L in zip(R, pick)]
    d["Random"] = {k: float(np.mean([np.mean([r["real"][L][k] for L in CAND]) for r in R])) for k in ("sel", "new_in", "new_out", "lost")}
    d["Random"]["_sel_vec"] = [float(np.mean([r["real"][L]["sel"] for L in CAND])) for r in R]
    po = [max(CAND, key=lambda L: r["real"][L]["sel"]) for r in R]
    d["Oracle"] = {k: float(np.mean([r["real"][L][k] for r, L in zip(R, po)])) for k in ("sel", "new_in", "new_out", "lost")}
    a = np.array(d["VE"]["_sel_vec"])
    for other in ("AtP-sel", "GradMax", "Random"):
        b = np.array(d[other]["_sel_vec"])
        bs = [np.mean((a - b)[i]) for i in [rng.randint(0, len(a), len(a)) for _ in range(2000)]]
        d[f"VE_minus_{other}"] = [float(np.mean(a - b)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    summ[f"w{w}"] = d
save_json({"summary": summ, "rows": rows}, f"{OUT}/e5c_summary.json")
for k, d in summ.items():
    print(k, "n", d["n"], " | ".join(f"{m} sel {d[m]['sel']:.3f} newIn {d[m]['new_in']:.1f} newOut {d[m]['new_out']:.1f}"
                                   for m in ("Random", "GradMax", "AtP-sel", "VE", "Oracle")))
    print("   VE-AtP", [round(v, 3) for v in d["VE_minus_AtP-sel"]], " VE-Random", [round(v, 3) for v in d["VE_minus_Random"]],
          " rho VE/AtP", round(d["VE"]["spearman_mean"], 2), round(d["AtP-sel"]["spearman_mean"], 2))
print(f"done t={time.time()-T0:.0f}s")
