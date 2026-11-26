# -*- coding: utf-8 -*-
"""Input-channel ablation of the localization network.

Same protocol as the fairness / ablation study (300 train + 100 val slices for training, 24 further val
slices for checkpoint selection, the same 60 fresh test patients; training on noise / mean-ablation / zero
interventions with log-uniform magnitude in [0.25, 2], testing on 4 intervention types x 4 patches per
layer), except that only the exact (std) realisation is generated, the buffers are kept in memory, and
each variant changes only the input channels of the network. Optimiser, steps, batch size and model
selection are unchanged.

Input channels (Eq. 10): 0-3 dy/r | 4 log(1+energy) | 5-8 clean output probability p | 9-12 input image x | 13 log r
Variants:
  full       all 14 channels (reference)
  no_logr    without the magnitude channel log r
  no_energy  without the energy channel
  no_prob    without the clean output probability p
  no_img     without the input image x
  out_only   output change only (dy/r, energy, log r), no p and no x
Run: python experiments/ablation/input_channels.py [variant,variant,...]   (SMOKE=1 for a smoke test)
Writes results/input_channels/<variant>.json, <variant>.pt, <variant>_test_pred.npy.
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.core import TRAIN_TYPES, TEST_TYPES, brain_mask_patches, make_delta, run_delta, InverseDecoder
from bia.variants import dec_in

SMOKE = os.environ.get("SMOKE") == "1"
OUT = set_out("input_channels" + ("_smoke" if SMOKE else ""))
T0 = time.time()
STEPS = 40 if SMOKE else 24 * (86400 // 192)
BS = 32 if SMOKE else 192
VARIANTS = {
    "full":      list(range(14)),
    "no_logr":   list(range(13)),
    "no_energy": [0, 1, 2, 3] + list(range(5, 14)),
    "no_prob":   list(range(5)) + list(range(9, 14)),
    "no_img":    list(range(9)) + [13],
    "out_only":  [0, 1, 2, 3, 4, 13],
}
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
for v in ONLY:
    assert v in VARIANTS, v

net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)

# same patients and random seeds as build_buffer.py
val = pick_slices("val", 124, seed=1)
train_names = [("train", n) for n in pick_slices("train", 300, seed=3)] + [("val", n) for n in val[:100]]
tune_names = [("val", n) for n in val[100:124]]
used = set(pick_slices("test", 40, seed=2))
test_names = [("test", n) for n in pick_slices("test", 243, seed=2) if n not in used][:60]
if SMOKE:
    train_names, tune_names, test_names = train_names[:3], tune_names[:2], test_names[:2]


@torch.no_grad()
def build(name, names, per, types_mode, seed0, type_list):
    """Same as the std part of build_buffer.build (same RNG consumption order); buffers stay in memory."""
    rng = np.random.RandomState(seed0)
    gen = torch.Generator(device=DEV)
    cap = len(names) * 9 * per * (len(type_list) if types_mode == "each" else 1)
    DYA = np.empty((cap, 4, GRID, GRID), np.float16)      # preallocated to avoid doubling memory when concatenating
    n = 0
    META, X, PROB, BM = [], [], [], []
    for si, (split, nm) in enumerate(names):
        x, _ = load_slice(nm, split)
        cache = forward(net, x[None])
        bm = brain_mask_patches(x)
        ks_all = torch.nonzero(bm).flatten().cpu().numpy()
        X.append(x.half()); PROB.append(cache["logits"].softmax(1)[0].half()); BM.append(bm)
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
            a = dy[torch.from_numpy(keep).to(DEV)].half().cpu().numpy()
            DYA[n:n + len(a)] = a; n += len(a); META.append(meta)
        if si % 50 == 0 or si == len(names) - 1:
            print(f"[build {name}] {si + 1}/{len(names)} t={time.time()-T0:.0f}s", flush=True)
    meta = np.concatenate(META)
    return {"n": len(meta), "dy": DYA[:n], "meta": meta, "X": torch.stack(X), "PROB": torch.stack(PROB),
            "BM": torch.stack(BM)}


TE = build("test", test_names, 4, "each", 2026, TEST_TYPES)
TU = build("tune", tune_names, 4, "each", 123, TEST_TYPES)
TR = build("train", train_names, 24, "mixed", 0, TRAIN_TYPES)
print(f"buffers: train {TR['n']} tune {TU['n']} test {TE['n']} ({TR['dy'].nbytes/1e9:.1f} GB) t={time.time()-T0:.0f}s",
      flush=True)


def inputs(S, idx, sel):
    m = S["meta"][idx]
    si = torch.from_numpy(m[:, 0].astype(np.int64)).to(DEV)
    dy = torch.from_numpy(S["dy"][idx]).to(DEV, non_blocking=True)
    return dec_in(dy, S["PROB"][si], S["X"][si])[:, sel], m, si


@torch.no_grad()
def predict(model, S, sel):
    model.eval()
    preds = []
    for i in range(0, S["n"], 256):
        idx = np.arange(i, min(i + 256, S["n"]))
        z, m, si = inputs(S, idx, sel)
        with amp():
            s = model(z).float()
        s = s.masked_fill(~S["BM"][si][:, None, :], -1e9)
        preds.append(s.flatten(1).argmax(1).cpu().numpy())
    model.train()
    j = np.concatenate(preds)
    return np.stack([j // 100, j % 100], 1)


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


def train(name):
    sel = VARIANTS[name]
    torch.manual_seed(1); np.random.seed(1)
    model = InverseDecoder(cin=len(sel), w=32).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.1)
    idx = np.arange(TR["n"])
    per_ep = len(idx) // BS
    n_ep = int(np.ceil(STEPS / per_ep))
    ev_every = max(1, n_ep // 24)
    best, best_state, log, step = -1, None, [], 0
    for ep in range(n_ep):
        order = idx[np.random.permutation(len(idx))]
        for b in range(per_ep):
            if step >= STEPS:
                break
            z, mb, si = inputs(TR, np.sort(order[b * BS:(b + 1) * BS]), sel)
            with amp():
                s = model(z).float()
            s = s.masked_fill(~TR["BM"][si][:, None, :], -1e9).flatten(1)
            lab = torch.from_numpy(mb[:, 1].astype(np.int64)).to(DEV)
            kk = torch.from_numpy(mb[:, 2].astype(np.int64)).to(DEV)
            loss = F.cross_entropy(s, lab * 100 + kk) + 0.5 * F.cross_entropy(torch.logsumexp(s.view(-1, 9, 100), 2), lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
        if ep % ev_every == 0 or step >= STEPS:
            tu = score(TU, predict(model, TU, sel))
            sc = float(np.mean([tu[t]["joint"] for t in TEST_TYPES]))
            log.append({"epoch": ep, "step": step, "loss": float(loss), "tune": tu})
            print(f"[{name}] ep {ep}/{n_ep} step {step} loss {float(loss):.3f} tune {sc:.3f} t={time.time()-T0:.0f}s", flush=True)
            if sc > best:
                best, best_state = sc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if step >= STEPS:
            break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT, f"{name}.pt"))
    pred = predict(model, TE, sel)
    te = score(TE, pred)
    np.save(os.path.join(OUT, f"{name}_test_pred.npy"), pred)
    if name == ONLY[0]:
        np.save(os.path.join(OUT, "test_meta.npy"), TE["meta"])
    save_json({"variant": name, "config": {"channels": sel, "steps": step, "batch": BS, "n_train_queries": int(TR["n"]),
                                           "device": str(DEV)},
               "best_tune_joint": best, "test": te, "log": log}, os.path.join(OUT, f"{name}.json"))
    print(f"[{name}] TEST " + json.dumps({k: round(v["joint"], 3) for k, v in te.items()}) +
          f" t={time.time()-T0:.0f}s", flush=True)


for name in ONLY:
    if os.path.exists(os.path.join(OUT, f"{name}.json")) and not SMOKE:
        print(f"[{name}] already done, skip"); continue
    train(name)
print(f"done t={time.time()-T0:.0f}s", flush=True)
