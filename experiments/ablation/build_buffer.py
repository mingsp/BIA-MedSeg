"""Training buffers for the fairness / ablation study (Table 2).

Protocol of the main localization experiment: 300 train + 100 val slices for training, 24 further val
slices for checkpoint selection, and the same 60 fresh test patients for testing. Every intervention is
realised three ways:
  std          exact star edit (block star, cell-aligned 16 px patch)
  interp       ordinary bilinear alignment: edit on the grid, bilinear map back (includes the round-trip
               error of the non-invertible alignment)
  interp_delta only the bilinear-downsampled edit is added, h + down(M*D)
For every train / tune / test query the 5 gradient attribution maps are stored as well; they are the
inputs of the gradient-map decoder.

The decoder inputs are full-resolution output changes (4x160x160 fp16, 200 KB each), so all buffers are
disk memmaps under results/ablation/buffer/ (~45 GB in total).
Run: python experiments/ablation/build_buffer.py   (SMOKE=1 for a 3-slice smoke test)
"""
import os, time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.engine import NP
from bia.core import TRAIN_TYPES, TEST_TYPES, brain_mask_patches, make_delta, run_delta, grad_attributions
from bia.variants import make_delta_interp, grad_feats, grid_masks

SMOKE = os.environ.get("SMOKE") == "1"
OUT = set_out("ablation" + ("_smoke" if SMOKE else ""))
BUF = os.path.join(OUT, "buffer"); os.makedirs(BUF, exist_ok=True)
T0 = time.time()
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
REAL = ("std", "interp", "interp_delta")


def make_delta_interp_delta(hc, L, ks, types, gen, kappa=None):
    """h + down(M*D): only the edit goes through the (non-invertible) bilinear map."""
    n = hc.shape[-1]
    up = lambda h: F.interpolate(h, size=(GRID, GRID), mode="bilinear", align_corners=False)
    dn = lambda g: F.interpolate(g, size=(n, n), mode="bilinear", align_corners=False)
    g = up(hc)[0]
    sig = hc.std(dim=(2, 3), keepdim=True)[0]
    mu = hc.mean(dim=(2, 3), keepdim=True)[0]
    M = grid_masks(ks)
    C = hc.shape[1]
    out = []
    for b, t in enumerate(types):
        e = torch.randn(g.shape, generator=gen, device=DEV)
        if t == "noise":
            D = (1.0 if kappa is None else float(kappa[b])) * sig * e
        elif t == "ablate":
            D = mu - g
        elif t == "zero":
            D = -g
        elif t == "gain":
            D = g
        else:
            cm = (torch.rand(C, 1, 1, generator=gen, device=DEV) < 0.25).float()
            D = 2.0 * sig * e * cm
        out.append(dn((M[b] * D)[None])[0])
    return torch.stack(out)


class Store:
    """Append-only fp16 memmaps for one query set."""

    def __init__(self, name, cap):
        self.name, self.cap, self.n = name, cap, 0
        mk = lambda k, shp, dt: np.lib.format.open_memmap(os.path.join(BUF, f"{name}_{k}.npy"), mode="w+",
                                                          dtype=dt, shape=(cap,) + shp)
        self.dy = {r: mk(f"dy_{r}", (4, GRID, GRID), np.float16) for r in REAL}
        self.gf = mk("gf", (45, NP, NP), np.float16)
        self.meta = mk("meta", (5,), np.int32)        # slice, layer, patch, type index, realisation-ok flag

    def add(self, dys, gf, meta):
        b = len(meta)
        for r in REAL:
            self.dy[r][self.n:self.n + b] = dys[r]
        self.gf[self.n:self.n + b] = gf
        self.meta[self.n:self.n + b] = meta
        self.n += b

    def close(self, extra):
        for r in REAL:
            self.dy[r].flush()
        self.gf.flush(); self.meta.flush()
        save_json({"n": self.n, "cap": self.cap, **extra}, os.path.join(BUF, f"{self.name}_info.json"))


def realise_all(x, cache, L, ks, types, kap, seed, gen):
    """Returns dict realisation -> dy (B,4,160,160) on device."""
    out = {}
    gen.manual_seed(seed)
    d = make_delta(cache[L], L, ks, types, gen, kappa=kap)
    out["std"] = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
    gen.manual_seed(seed + 1)
    d = make_delta_interp(cache[L], L, ks, types, gen, kappa=kap)
    out["interp"] = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
    gen.manual_seed(seed + 2)
    d = make_delta_interp_delta(cache[L], L, ks, types, gen, kappa=kap)
    out["interp_delta"] = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
    return out


def build(name, names, per, types_mode, seed0, type_list):
    cap = len(names) * 9 * per * (len(type_list) if types_mode == "each" else 1)
    st = Store(name, cap)
    rng = np.random.RandomState(seed0)
    gen = torch.Generator(device=DEV)
    X, PROB, BM = [], [], []
    for si, (split, nm) in enumerate(names):
        x, _ = load_slice(nm, split)
        cache = forward(net, x[None])
        bm = brain_mask_patches(x)
        ks_all = torch.nonzero(bm).flatten().cpu().numpy()
        X.append(x.half().cpu().numpy()); PROB.append(cache["logits"].softmax(1)[0].half().cpu().numpy())
        BM.append(bm.cpu().numpy())
        loops = [(t, li) for t in type_list for li in range(9)] if types_mode == "each" else [(None, li) for li in range(9)]
        for t, li in loops:
            L = LAYERS[li]
            ks = rng.choice(ks_all, per)
            if t is None:      # training: mixed types and magnitudes
                types = [type_list[i] for i in rng.randint(len(type_list), size=per)]
                kap = np.exp(rng.uniform(np.log(0.25), np.log(2.0), per))
            else:
                types, kap = [t] * per, None
            seed = int(rng.randint(1 << 30))
            dys = realise_all(x, cache, L, ks, types, kap, seed, gen)
            ok = (dys["std"].pow(2).mean((1, 2, 3)).sqrt() > 1e-6)
            for r in ("interp", "interp_delta"):
                ok &= dys[r].pow(2).mean((1, 2, 3)).sqrt() > 1e-6
            gf = grad_feats(grad_attributions(net, x, cache, dys["std"], ig_steps=0))
            keep = ok.cpu().numpy()
            tix = [TEST_TYPES.index(tt) if tt in TEST_TYPES else 10 + TRAIN_TYPES.index(tt) for tt in types]
            meta = np.array([[si, li, int(ks[b]), tix[b], 1] for b in range(per)], np.int32)[keep]
            st.add({r: dys[r][torch.from_numpy(keep).to(DEV)].half().cpu().numpy() for r in REAL},
                   gf[torch.from_numpy(keep).to(DEV)].half().cpu().numpy(), meta)
        if si % 25 == 0 or si == len(names) - 1:
            print(f"[{name}] {si + 1}/{len(names)} n={st.n} t={time.time()-T0:.0f}s", flush=True)
    np.save(os.path.join(BUF, f"{name}_X.npy"), np.stack(X))
    np.save(os.path.join(BUF, f"{name}_PROB.npy"), np.stack(PROB))
    np.save(os.path.join(BUF, f"{name}_BM.npy"), np.stack(BM))
    st.close({"names": names, "per": per, "seed": seed0, "types": type_list,
              "type_index": "0-3 = TEST_TYPES (noise, ablate, gain, chansub); 10+i = TRAIN_TYPES[i]"})


build("test", test_names, 4, "each", 2026, TEST_TYPES)          # same 60 patients and RNG seed as the main localization evaluation
build("tune", tune_names, 4, "each", 123, TEST_TYPES)
build("train", train_names, 24, "mixed", 0, TRAIN_TYPES)
print(f"done t={time.time()-T0:.0f}s", flush=True)
