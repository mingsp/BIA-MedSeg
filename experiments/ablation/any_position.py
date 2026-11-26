# -*- coding: utf-8 -*-
"""Source localization for interventions at any position and size, and seed stability of the localization network.

Same protocol as the fairness / ablation study (300 train + 100 val slices for training, 24 further val
slices for model selection, the same 60 test patients; training interventions are noise, mean ablation
and zeroing with log-uniform magnitude in [0.25, 2], 24 per layer and slice, 86,400 in total). Only the
position of the training interventions changes:
  grid     16 px patches aligned with the 10x10 grid (main setting); trained with seeds 1, 2, 3 for
           mean and standard deviation over seeds
  anypos   each intervention is one of three forms with equal probability: an aligned 16 px patch, a 16 px
           patch at a random pixel position, an 8 px patch at a random pixel position. The edit is applied
           through Eq. (3) (every position of the source layer is edited in proportion to its overlap with
           the intervened box); the label is the source layer and the grid patch containing the box centre.
Test sets (8,640 queries each):
  aligned            the main-experiment queries (same random seed as the main localization evaluation)
  offset16 / rand8   off-grid queries, generated with the same procedure and seeds as the attribution
                     comparison on off-grid queries (any_position_atp.py)
The main localization network is also re-evaluated on offset16 / rand8 as a reference. Metrics:
localization accuracy (grid patch containing the centre), layer accuracy, and the fraction of selected
patches that intersect the intervened box; 95% CIs by resampling patients.
Run: python experiments/ablation/any_position.py   (SMOKE=1 for a smoke test; GPU_FRAC limits GPU memory)
Writes results/any_position/.
"""
import os, time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.engine import P, NP
from bia.core import TRAIN_TYPES, TEST_TYPES, brain_mask_patches, make_delta, run_delta, InverseDecoder
from bia.variants import dec_in

if DEV.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("GPU_FRAC", "0.5")))
SMOKE = os.environ.get("SMOKE") == "1"
OUT = set_out("any_position" + ("_smoke" if SMOKE else ""))
BASE_DIR = os.path.join(RESULTS, "base")
T0 = time.time()
STEPS = 40 if SMOKE else 24 * (86400 // 192)
BS = 32 if SMOKE else 192
SEL = list(range(14))


def log(*a):
    print(*a, f"t={time.time()-T0:.0f}s", flush=True)


net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)
val = pick_slices("val", 124, seed=1)
train_names = [("train", n) for n in pick_slices("train", 300, seed=3)] + [("val", n) for n in val[:100]]
tune_names = [("val", n) for n in val[100:124]]
used = set(pick_slices("test", 40, seed=2))
test_names = [("test", n) for n in pick_slices("test", 243, seed=2) if n not in used][:60]
if SMOKE:
    train_names, tune_names, test_names = train_names[:3], tune_names[:2], test_names[:2]


# box interventions (same as in any_position_atp.py)
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


def box_delta_train(hc, L, boxes, types, gen, kappa):
    """Training interventions (noise with kappa / mean-ablation / zero) on arbitrary boxes."""
    B = len(boxes); m = frac_masks(L, boxes)
    sig = hc.std(dim=(2, 3), keepdim=True); mu = hc.mean(dim=(2, 3), keepdim=True)
    h = hc.expand(B, *hc.shape[1:])
    eps = torch.randn(h.shape, generator=gen, device=DEV)
    out = torch.zeros_like(h)
    for b, t in enumerate(types):
        if t == "noise":
            out[b] = float(kappa[b]) * sig[0] * eps[b]
        elif t == "ablate":
            out[b] = mu[0] - h[b]
        else:
            out[b] = -h[b]
    return out * m


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


CONDS = [("aligned16", 16, False), ("offset16", 16, True), ("rand8", 8, True)]


def touch_of(box):
    y0, x0, pp = box
    return sorted({(yy // P) * NP + xx // P for yy in range(y0, y0 + pp) for xx in range(x0, x0 + pp)})


# query sets (kept in RAM)
class Buf:
    def __init__(self, cap):
        self.dy = np.empty((cap, 4, GRID, GRID), np.float16); self.n = 0
        self.meta, self.touch = [], []
        self.X, self.PROB, self.BM = [], [], []

    def add_slice(self, x, cache, bm):
        self.X.append(x.half()); self.PROB.append(cache["logits"].softmax(1)[0].half()); self.BM.append(bm)

    def add(self, dy, keep, meta, touch):
        a = dy[torch.from_numpy(keep).to(DEV)].half().cpu().numpy()
        self.dy[self.n:self.n + len(a)] = a; self.n += len(a)
        self.meta += [m for m, k in zip(meta, keep) if k]; self.touch += [t for t, k in zip(touch, keep) if k]

    def done(self):
        self.dy = self.dy[:self.n]; self.meta = np.array(self.meta, np.int32)
        self.X = torch.stack(self.X); self.PROB = torch.stack(self.PROB); self.BM = torch.stack(self.BM)
        return self


@torch.no_grad()
def build_grid(names, per, mode, seed0, type_list):
    """Same RNG use as build_buffer.py (std realisation)."""
    rng = np.random.RandomState(seed0); gen = torch.Generator(device=DEV)
    B = Buf(len(names) * 9 * per * (len(type_list) if mode == "each" else 1))
    for si, (split, nm) in enumerate(names):
        x, _ = load_slice(nm, split); cache = forward(net, x[None]); bm = brain_mask_patches(x)
        ks_all = torch.nonzero(bm).flatten().cpu().numpy(); B.add_slice(x, cache, bm)
        loops = [(t, li) for t in type_list for li in range(9)] if mode == "each" else [(None, li) for li in range(9)]
        for t, li in loops:
            L = LAYERS[li]; ks = rng.choice(ks_all, per)
            if t is None:
                types = [type_list[i] for i in rng.randint(len(type_list), size=per)]
                kap = np.exp(rng.uniform(np.log(0.25), np.log(2.0), per))
            else:
                types, kap = [t] * per, None
            gen.manual_seed(int(rng.randint(1 << 30)))
            d = make_delta(cache[L], L, ks, types, gen, kappa=kap)
            dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
            keep = (dy.pow(2).mean((1, 2, 3)).sqrt() > 1e-6).cpu().numpy()
            tix = [TEST_TYPES.index(tt) if tt in TEST_TYPES else 10 + TRAIN_TYPES.index(tt) for tt in types]
            B.add(dy, keep, [[si, li, int(ks[b]), tix[b]] for b in range(per)], [[int(ks[b])] for b in range(per)])
        if si % 100 == 0:
            log(f"[grid {seed0}] {si + 1}/{len(names)} n={B.n}")
    return B.done()


@torch.no_grad()
def build_anypos(names, per, seed0):
    rng = np.random.RandomState(seed0); gen = torch.Generator(device=DEV)
    B = Buf(len(names) * 9 * per)
    for si, (split, nm) in enumerate(names):
        x, _ = load_slice(nm, split); cache = forward(net, x[None]); bm = brain_mask_patches(x)
        ks_all = torch.nonzero(bm).flatten().cpu().numpy().tolist(); B.add_slice(x, cache, bm)
        for li in range(9):
            L = LAYERS[li]
            conds = rng.randint(3, size=per)
            bx = [sample_boxes(rng, CONDS[c][1], CONDS[c][2], ks_all, 1)[0] for c in conds]
            types = [TRAIN_TYPES[i] for i in rng.randint(3, size=per)]
            kap = np.exp(rng.uniform(np.log(0.25), np.log(2.0), per))
            gen.manual_seed(int(rng.randint(1 << 30)))
            d = box_delta_train(cache[L], L, [b for b, _ in bx], types, gen, kap)
            dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
            keep = (dy.pow(2).mean((1, 2, 3)).sqrt() > 1e-6).cpu().numpy()
            B.add(dy, keep, [[si, li, int(bx[b][1]), 10 + TRAIN_TYPES.index(types[b])] for b in range(per)],
                  [touch_of(bx[b][0]) for b in range(per)])
        if si % 100 == 0:
            log(f"[anypos] {si + 1}/{len(names)} n={B.n}")
    return B.done()


@torch.no_grad()
def build_s2(names, ci, seed_base=0):
    """offset16 / rand8 queries, as in any_position_atp.py (rng = RandomState(1000 * si + ci + seed_base))."""
    cond, p, offset = CONDS[ci]
    gen = torch.Generator(device=DEV)
    B = Buf(len(names) * 4 * 9 * 4)
    for si, (split, nm) in enumerate(names):
        x, _ = load_slice(nm, split); cache = forward(net, x[None]); bm = brain_mask_patches(x)
        ks_all = torch.nonzero(bm).flatten().cpu().numpy().tolist(); B.add_slice(x, cache, bm)
        rng = np.random.RandomState(1000 * si + ci + seed_base)
        for ti, t in enumerate(TEST_TYPES):
            for li, L in enumerate(LAYERS):
                bx = sample_boxes(rng, p, offset, ks_all, 4)
                gen.manual_seed(int(rng.randint(1 << 30)))
                d = s2_delta(cache[L], L, [b for b, _ in bx], t, gen)
                dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
                keep = (dy.pow(2).mean((1, 2, 3)).sqrt() > 1e-6).cpu().numpy()
                B.add(dy, keep, [[si, li, int(bx[b][1]), ti] for b in range(4)], [touch_of(bx[b][0]) for b in range(4)])
    log(f"[s2 {cond}{'-tune' if seed_base else ''}] n={B.n}")
    return B.done()


TESTS = {"aligned": build_grid(test_names, 4, "each", 2026, TEST_TYPES),
         "offset16": build_s2(test_names, 1), "rand8": build_s2(test_names, 2)}
TUNE_GRID = build_grid(tune_names, 4, "each", 123, TEST_TYPES)
TUNE_ANY = {"aligned": TUNE_GRID, "offset16": build_s2(tune_names, 1, 777), "rand8": build_s2(tune_names, 2, 777)}


# train / evaluate
def inputs(S, idx):
    m = S.meta[idx]
    si = torch.from_numpy(m[:, 0].astype(np.int64)).to(DEV)
    dy = torch.from_numpy(S.dy[idx]).to(DEV, non_blocking=True)
    return dec_in(dy, S.PROB[si], S.X[si])[:, SEL], m, si


@torch.no_grad()
def predict(model, S):
    model.eval(); out = []
    for i in range(0, S.n, 256):
        idx = np.arange(i, min(i + 256, S.n))
        z, m, si = inputs(S, idx)
        with amp():
            s = model(z).float()
        out.append(s.masked_fill(~S.BM[si][:, None, :], -1e9).flatten(1).argmax(1).cpu().numpy())
    model.train()
    j = np.concatenate(out)
    return np.stack([j // 100, j % 100], 1)


def boot(c, si, n=1000, seed=0):
    r = np.random.RandomState(seed); pats = np.unique(si); by = {p: c[si == p] for p in pats}
    v = [np.concatenate([by[p] for p in r.choice(pats, len(pats))]).mean() for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def score(S, pred, ci=False):
    m = S.meta
    J = (pred[:, 0] == m[:, 1]) & (pred[:, 1] == m[:, 2])
    Ly = pred[:, 0] == m[:, 1]
    Ov = np.array([Ly[i] and pred[i, 1] in S.touch[i] for i in range(len(m))])
    out = {"joint": float(J.mean()), "layer": float(Ly.mean()), "overlap": float(Ov.mean()),
           "by_type": {t: float(J[m[:, 3] == i].mean()) for i, t in enumerate(TEST_TYPES)},
           "by_layer": [float(J[m[:, 1] == l].mean()) for l in range(9)]}
    if ci:
        out["joint_ci"] = boot(J.astype(float), m[:, 0]); out["layer_ci"] = boot(Ly.astype(float), m[:, 0])
        out["overlap_ci"] = boot(Ov.astype(float), m[:, 0])
    return out


def tune_score(model, tune):
    return float(np.mean([score(S, predict(model, S))["joint"] for S in tune.values()]))


def train(name, TR, tune, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = InverseDecoder(cin=14, w=32).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.1)
    idx = np.arange(TR.n); per_ep = len(idx) // BS; n_ep = int(np.ceil(STEPS / per_ep))
    best, best_state, tlog, step = -1, None, [], 0
    for ep in range(n_ep):
        order = idx[np.random.permutation(len(idx))]
        for b in range(per_ep):
            if step >= STEPS:
                break
            z, mb, si = inputs(TR, np.sort(order[b * BS:(b + 1) * BS]))
            with amp():
                s = model(z).float()
            s = s.masked_fill(~TR.BM[si][:, None, :], -1e9).flatten(1)
            lab = torch.from_numpy(mb[:, 1].astype(np.int64)).to(DEV)
            kk = torch.from_numpy(mb[:, 2].astype(np.int64)).to(DEV)
            loss = F.cross_entropy(s, lab * 100 + kk) + 0.5 * F.cross_entropy(torch.logsumexp(s.view(-1, 9, 100), 2), lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
        sc = tune_score(model, tune)
        tlog.append({"epoch": ep, "loss": float(loss), "tune": sc})
        log(f"[{name}] ep {ep}/{n_ep} loss {float(loss):.3f} tune {sc:.3f}")
        if sc > best:
            best, best_state = sc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step >= STEPS:
            break
    model.load_state_dict(best_state); torch.save(best_state, os.path.join(OUT, f"{name}.pt"))
    return model, {"best_tune": best, "log": tlog}


def evaluate(name, model, extra=None):
    res = {k: score(S, predict(model, S), ci=True) for k, S in TESTS.items()}
    save_json({"variant": name, "test": res, **(extra or {})}, os.path.join(OUT, f"{name}.json"))
    log(f"[{name}] " + " | ".join(f"{k} joint {v['joint']:.3f} layer {v['layer']:.3f} overlap {v['overlap']:.3f}"
                                    for k, v in res.items()))


# reference: the main localization network on the off-grid test sets
e6 = InverseDecoder(cin=14, w=32).to(DEV)
e6.load_state_dict(torch.load(f"{BASE_DIR}/e6_invnet.pt", map_location=DEV)); e6.eval()
evaluate("e6_reference", e6)

TR = build_grid(train_names, 24, "mixed", 0, TRAIN_TYPES)
log(f"grid buffer {TR.n} ({TR.dy.nbytes/1e9:.1f} GB)")
for seed in (1, 2, 3):
    name = f"grid_s{seed}"
    if os.path.exists(os.path.join(OUT, f"{name}.json")) and not SMOKE:
        continue
    model, info = train(name, TR, {"aligned": TUNE_GRID}, seed)
    evaluate(name, model, info)
del TR
import gc; gc.collect()
TR = build_anypos(train_names, 24, 0)
log(f"anypos buffer {TR.n} ({TR.dy.nbytes/1e9:.1f} GB)")
for seed in (1,) if SMOKE else (1, 2):
    name = f"anypos_s{seed}"
    if os.path.exists(os.path.join(OUT, f"{name}.json")) and not SMOKE:
        continue
    model, info = train(name, TR, TUNE_ANY, seed)
    evaluate(name, model, info)
log("done")
