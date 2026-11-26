"""Output-change footprints of interventions (Fig. 6, Sec. 4.3.1).

A. Direct footprints: 16-px patch Gaussian noise (kappa = 1 x channel std) at every layer and every
   brain patch; change-energy maps at every downstream layer and at the output (40x40, 4-px cells).
   Validation split: accumulate translation-aligned prototypes. Test split: store output maps
   (draw 1 = queries, draw 2 = independent re-run for the exhaustive dictionary, ablate = a second
   fault type) and the full intermediate maps for the first 10 test slices.
B. Edge kernels (validation): unit-block noise on one source path, read at the immediate consumer.
C. Linearity (validation subset): the same noise at kappa in {0.25, 0.5, 1, 2}.
D. Skip removal (all slices): replace one whole skip copy by its channel mean; and path split, i.e.
   patch noise on an encoder layer through the skip path only or the deep path only.

Writes results/base/e2_*.npz / .npy / .json (read by summarize_footprints.py).
Run: python experiments/response/footprints.py
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.engine import P, NP, brain_patches, make_edit, cell_mask
OUT = set_out("base")

net = load_model("baseline", 123)
G40 = 40
DST = LAYERS + ["logits"]
EDGES = [("inc", "deep", "down1"), ("down1", "deep", "down2"), ("down2", "deep", "down3"),
         ("down3", "deep", "down4"), ("down4", None, "up1"), ("down3", "skip", "up1"),
         ("up1", None, "up2"), ("down2", "skip", "up2"), ("up2", None, "up3"),
         ("down1", "skip", "up3"), ("up3", None, "up4"), ("inc", "skip", "up4"),
         ("up4", None, "logits")]
C = 2 * G40 - 1  # canvas size for offsets -39..39


def refs(cache):
    r = {L: float((cache[L] ** 2).sum(1).mean()) for L in LAYERS}
    r["logits"] = float((cache["logits"] ** 2).sum(1).mean())
    return r


def e40(d, ref):
    e = (d.float() ** 2).sum(1) / ref
    n = e.shape[-1]
    if n > G40:
        e = F.avg_pool2d(e[:, None], n // G40)[:, 0]
    elif n < G40:
        e = e.repeat_interleave(G40 // n, 1).repeat_interleave(G40 // n, 2)
    return e


def downstream(L):
    i = LAYERS.index(L)
    out = LAYERS[i:] + ["logits"]
    if L.startswith("up"):
        return out
    return out


def accumulate(canvas, count, maps, tl, norm):
    """Add maps (B,40,40) into the canvas, aligned so that the cell tl (B,2) sits at the canvas centre."""
    for b in range(maps.shape[0]):
        r0, c0 = int(tl[b][0]), int(tl[b][1])
        y0, x0 = G40 - 1 - r0, G40 - 1 - c0
        canvas[y0:y0 + G40, x0:x0 + G40] += (maps[b] / norm[b]).double().cpu().numpy()
        count[y0:y0 + G40, x0:x0 + G40] += 1


def radius_out(maps, centers):
    yy, xx = np.meshgrid(np.arange(G40) * 4 + 1.5, np.arange(G40) * 4 + 1.5, indexing="ij")
    m = maps.double().cpu().numpy()
    tot = m.sum((1, 2)) + 1e-30
    rad = np.sqrt(((m * ((yy[None] - centers[:, 0, None, None]) ** 2 +
                         (xx[None] - centers[:, 1, None, None]) ** 2)).sum((1, 2))) / tot)
    return rad, tot


calib = pick_slices("val", 16, seed=1)
test = pick_slices("test", 40, seed=2)
gen = torch.Generator(device=DEV)
T0 = time.time()

tmpl = {(L, j): [np.zeros((C, C)), np.zeros((C, C))] for L in LAYERS for j in downstream(L)}
stats = []          # rows: split, slice, layer, patch, dst, total, radius
test_out = {"q": [], "b": [], "a": [], "src": []}  # per-sample output maps
test_full = []      # intermediate maps of the first 10 test slices
meta = []

# A. direct footprints
for split, names in (("val", calib), ("test", test)):
    for si, nm in enumerate(names):
        x, y = load_slice(nm, split)
        cache = forward(net, x[None])
        R = refs(cache)
        ks = brain_patches(x)
        meta.append({"split": split, "slice": nm, "n_patches": len(ks)})
        tl = np.array([(divmod(k, NP)[0] * 4, divmod(k, NP)[1] * 4) for k in ks])
        cen = tl * 4 + P / 2 - 0.5
        draws = [("q", "noise", 1000 * si + 1)]
        if split == "test":
            draws += [("b", "noise", 1000 * si + 2), ("a", "ablate", 0)]
        for L in LAYERS:
            for tag, kind, seed in draws:
                gen.manual_seed(seed * 17 + LAYERS.index(L))
                fn = make_edit(L, ks, kind, cache[L], gen)
                out = forward(net, x[None].expand(len(ks), *x.shape), edits={L: fn}, cache=cache)
                maps = {j: e40(out[j] - cache[j], R[j]) for j in downstream(L)}
                src_tot = maps[L].sum((1, 2))
                if tag == "q":
                    for j in downstream(L):
                        rad, tot = radius_out(maps[j], cen)
                        for b, k in enumerate(ks):
                            stats.append((split, si, LAYERS.index(L), k, DST.index(j), float(tot[b]),
                                          float(rad[b]), float(src_tot[b])))
                    if split == "val":
                        for j in downstream(L):
                            accumulate(*tmpl[(L, j)], maps[j], tl, src_tot)
                if split == "test":
                    test_out[tag].append(maps["logits"].half().cpu().numpy())
                    if tag == "q":
                        test_out["src"].append(np.array([[si, LAYERS.index(L), k] for k in ks]))
                        if si < 10:
                            test_full.append({j: maps[j].half().cpu().numpy() for j in downstream(L)})
        print(f"A {split} {si} {nm} patches={len(ks)} t={time.time()-T0:.0f}s", flush=True)

np.savez_compressed(f"{OUT}/e2_test_maps.npz", q=np.concatenate(test_out["q"]),
                    b=np.concatenate(test_out["b"]), a=np.concatenate(test_out["a"]),
                    src=np.concatenate(test_out["src"]))
full = {}
idx = 0
for si in range(10):
    for L in LAYERS:
        d = test_full[idx]; idx += 1
        for j, v in d.items():
            full[f"s{si}_{L}_{j}"] = v
np.savez_compressed(f"{OUT}/e2_test_full.npz", **full)
np.save(f"{OUT}/e2_stats.npy", np.array([(0 if s[0] == "val" else 1,) + s[1:] for s in stats]))
np.savez_compressed(f"{OUT}/e2_templates.npz",
                    **{f"{L}__{j}": v[0] / np.maximum(v[1], 1) for (L, j), v in tmpl.items()},
                    **{f"cnt_{L}__{j}": v[1] for (L, j), v in tmpl.items()})
del test_full, full

# B. edge kernels
def block_mask(layer, blocks, b):
    s = STRIDE[layer]; n = GRID // s; c = b // s
    m = torch.zeros(len(blocks), 1, n, n, device=DEV)
    for i, (r, q) in enumerate(blocks):
        m[i, 0, r * c:(r + 1) * c, q * c:(q + 1) * c] = 1
    return m


ek = {e: [np.zeros((C, C)), np.zeros((C, C))] for e in EDGES}
rng = np.random.RandomState(5)
for si, nm in enumerate(calib):
    x, _ = load_slice(nm, "val")
    cache = forward(net, x[None]); R = refs(cache)
    fg = (x.abs().sum(0) > 1e-6).float()
    for (src, path, dst) in EDGES:
        b = max(4, STRIDE[src]); nb = GRID // b
        occ = fg.view(nb, b, nb, b).mean((1, 3))
        cand = torch.nonzero(occ >= 0.5).cpu().numpy()
        sel = cand[rng.choice(len(cand), min(64, len(cand)), replace=False)]
        m = block_mask(src, [tuple(v) for v in sel], b)
        sig = cache[src].std(dim=(2, 3), keepdim=True)
        gen.manual_seed(77 + si * 31 + EDGES.index((src, path, dst)))
        eps = torch.randn((len(sel),) + tuple(cache[src].shape[1:]), generator=gen, device=DEV)
        delta = sig * eps * m
        key = src if path is None else f"{src}@{path}"
        out = forward(net, x[None].expand(len(sel), *x.shape), edits={key: (lambda h: h + delta)},
                      cache=cache)
        srcmap = e40(delta, R[src])
        dmap = e40(out[dst] - cache[dst], R[dst])
        tl = sel * (b // 4)
        accumulate(*ek[(src, path, dst)], dmap, tl, srcmap.sum((1, 2)))
    print(f"B {si} t={time.time()-T0:.0f}s", flush=True)
np.savez_compressed(f"{OUT}/e2_edges.npz",
                    **{f"{s}__{p}__{d}": v[0] / np.maximum(v[1], 1) for (s, p, d), v in ek.items()})

# C. linearity
lin = []
for si, nm in enumerate(calib[:4]):
    x, _ = load_slice(nm, "val")
    cache = forward(net, x[None]); R = refs(cache)
    ks = brain_patches(x)[::5][:10]
    for L in LAYERS:
        m = cell_mask(L, ks)
        sig = cache[L].std(dim=(2, 3), keepdim=True)
        gen.manual_seed(999 + si)
        eps = torch.randn((len(ks),) + tuple(cache[L].shape[1:]), generator=gen, device=DEV)
        res = {}
        for kap in (0.25, 0.5, 1.0, 2.0):
            d = kap * sig * eps * m
            out = forward(net, x[None].expand(len(ks), *x.shape), edits={L: (lambda h, d=d: h + d)},
                          cache=cache)
            res[kap] = e40(out["logits"] - cache["logits"], R["logits"]).cpu().numpy()
        ref = res[1.0]
        for kap, v in res.items():
            for b in range(len(ks)):
                a1, a2 = np.sqrt(v[b]).ravel(), np.sqrt(ref[b]).ravel()
                cos = float(a1 @ a2 / (np.linalg.norm(a1) * np.linalg.norm(a2) + 1e-30))
                lin.append((si, LAYERS.index(L), kap, float(v[b].sum()), float(ref[b].sum()), cos))
np.save(f"{OUT}/e2_linearity.npy", np.array(lin))
print(f"C t={time.time()-T0:.0f}s", flush=True)

# D. skip removal and path split
def dice(pred, gt, cls_set):
    p = torch.isin(pred, torch.tensor(cls_set, device=DEV)); g = torch.isin(gt, torch.tensor(cls_set, device=DEV))
    return float(2 * (p & g).sum() / (p.sum() + g.sum()).clamp_min(1))


skip_rows, skip_maps, path_rows = [], {}, []
for split, names in (("val", calib), ("test", test)):
    for si, nm in enumerate(names):
        x, y = load_slice(nm, split)
        cache = forward(net, x[None]); R = refs(cache)
        pred0 = cache["logits"].argmax(1)[0]
        for e in ("inc", "down1", "down2", "down3"):
            mu = cache[e].mean(dim=(2, 3), keepdim=True)
            out = forward(net, x[None], edits={f"{e}@skip": (lambda h, mu=mu: mu.expand_as(h))},
                          cache=cache)
            pred = out["logits"].argmax(1)[0]
            row = [0 if split == "val" else 1, si, ["inc", "down1", "down2", "down3"].index(e)]
            for j in ("up1", "up2", "up3", "up4", "logits"):
                mp = e40(out[j] - cache[j], R[j])[0].cpu().numpy()
                row.append(float(mp.mean()))
                if split == "test" and si < 10:
                    skip_maps[f"s{si}_{e}_{j}"] = mp.astype(np.float16)
            row += [dice(pred, y, [1, 2, 3]), dice(pred0, y, [1, 2, 3]), float((pred != pred0).float().mean())]
            skip_rows.append(row)
        if split == "val":
            ks = brain_patches(x)
            tl = np.array([(divmod(k, NP)[0] * 4, divmod(k, NP)[1] * 4) for k in ks])
            cen = tl * 4 + P / 2 - 0.5
            for L in ("inc", "down1", "down2", "down3"):
                for path in ("skip", "deep"):
                    gen.manual_seed(5000 + si)
                    fn = make_edit(L, ks, "noise", cache[L], gen)
                    out = forward(net, x[None].expand(len(ks), *x.shape), edits={f"{L}@{path}": fn},
                                  cache=cache)
                    mp = e40(out["logits"] - cache["logits"], R["logits"])
                    rad, tot = radius_out(mp, cen)
                    for b in range(len(ks)):
                        path_rows.append((si, LAYERS.index(L), 0 if path == "skip" else 1,
                                          float(tot[b]), float(rad[b])))
np.save(f"{OUT}/e2_skip.npy", np.array(skip_rows))
np.save(f"{OUT}/e2_path.npy", np.array(path_rows))
np.savez_compressed(f"{OUT}/e2_skip_maps.npz", **skip_maps)
save_json({"calib": calib, "test": test, "meta": meta, "edges": EDGES, "time_s": time.time() - T0},
          f"{OUT}/e2_meta.json")
print(f"done t={time.time()-T0:.0f}s", flush=True)
