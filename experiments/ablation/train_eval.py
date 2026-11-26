"""Trains and evaluates every variant of the fairness / ablation study (Table 2).

Each variant is trained on the buffers written by build_buffer.py with the same optimiser, number of
steps and batch size as the main localization network; the checkpoint is selected on the tune set and
evaluated once on the 60 fresh test patients.

Variants (name: realisation, input channels, #train slices, train types, seed)
  full          exact star alignment, reference for every ablation
  full_s2       same, second seed (seed stability)
  gradlearned   same budget, but the decoder sees the 5 gradient attribution maps (+ pooled image and
                clean probability) instead of the output change
  interp        naive bilinear alignment pipeline (with round-trip error)
  interp_delta  careful bilinear pipeline (only the edit is mapped back)
  dyonly, nomag input-channel ablations
  noiseonly     trained on noise interventions only
  n133, n50     fewer training slices
Run: python experiments/ablation/train_eval.py [variant,variant,...]
Writes results/ablation/<variant>.json, <variant>.pt, <variant>_test_pred.npy.
"""
import json, os, sys, threading, queue, time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.core import InverseDecoder, TEST_TYPES, TRAIN_TYPES
from bia.variants import GradDecoder, grad_in, dec_in

SMOKE = os.environ.get("SMOKE") == "1"
OUT = set_out("ablation" + ("_smoke" if SMOKE else ""))
BUF = os.path.join(OUT, "buffer")
T0 = time.time()
STEPS = 40 if SMOKE else 24 * (86400 // 192)
BS = 32 if SMOKE else 192
ALL14 = list(range(14))
VARIANTS = {
    "full":         ("std", ALL14, 400, None, 1),
    "gradlearned":  ("grad", None, 400, None, 1),
    "interp":       ("interp", ALL14, 400, None, 1),
    "interp_delta": ("interp_delta", ALL14, 400, None, 1),
    "full_s2":      ("std", ALL14, 400, None, 2),
    "dyonly":       ("std", list(range(5)), 400, None, 1),
    "nomag":        ("std", list(range(13)), 400, None, 1),
    "noiseonly":    ("std", ALL14, 400, ["noise"], 1),
    "n133":         ("std", ALL14, 133, None, 1),
    "n50":          ("std", ALL14, 50, None, 1),
}
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)


def load_set(name):
    info = json.load(open(os.path.join(BUF, f"{name}_info.json")))
    n = info["n"]
    mm = lambda k: np.load(os.path.join(BUF, f"{name}_{k}.npy"), mmap_mode="r")
    d = {"n": n, "meta": np.array(mm("meta")[:n]), "gf": mm("gf"),
         "dy": {r: mm(f"dy_{r}") for r in ("std", "interp", "interp_delta")},
         "X": torch.from_numpy(np.load(os.path.join(BUF, f"{name}_X.npy"))).to(DEV),
         "PROB": torch.from_numpy(np.load(os.path.join(BUF, f"{name}_PROB.npy"))).to(DEV),
         "BM": torch.from_numpy(np.load(os.path.join(BUF, f"{name}_BM.npy"))).to(DEV)}
    return d


TR, TU, TE = load_set("train"), load_set("tune"), load_set("test")
print(f"buffers: train {TR['n']} tune {TU['n']} test {TE['n']}  t={time.time()-T0:.0f}s", flush=True)


def batch(S, idx, real, sel):
    """Fetch one batch (sorted indices give fast memmap reads); returns model input, metadata, slice index."""
    idx = np.sort(idx)
    m = S["meta"][idx]
    si = torch.from_numpy(m[:, 0].astype(np.int64)).to(DEV)
    if real == "grad":
        gf = torch.from_numpy(np.asarray(S["gf"][idx])).to(DEV)
        z = grad_in(gf, S["PROB"][si], S["X"][si])
    else:
        dy = torch.from_numpy(np.asarray(S["dy"][real][idx])).to(DEV)
        z = dec_in(dy, S["PROB"][si], S["X"][si])[:, sel]
    return z, m, si


def prefetch(S, order, real, sel, bs):
    q = queue.Queue(maxsize=6)

    def run():
        for i in range(0, len(order) - bs + 1, bs):
            idx = order[i:i + bs]
            mm = np.sort(idx)
            m = S["meta"][mm]
            if real == "grad":
                a = np.asarray(S["gf"][mm])
            else:
                a = np.asarray(S["dy"][real][mm])
            q.put((a, m))
        q.put(None)
    threading.Thread(target=run, daemon=True).start()
    while True:
        it = q.get()
        if it is None:
            return
        a, m = it
        si = torch.from_numpy(m[:, 0].astype(np.int64)).to(DEV)
        a = torch.from_numpy(a).to(DEV)
        z = grad_in(a, S["PROB"][si], S["X"][si]) if real == "grad" else dec_in(a, S["PROB"][si], S["X"][si])[:, sel]
        yield z, m, si


@torch.no_grad()
def predict(model, S, real, sel):
    model.eval()
    order = np.arange(S["n"])
    preds = []
    for z, m, si in prefetch(S, order, real, sel, 256):
        with amp():
            s = model(z).float()
        s = s.masked_fill(~S["BM"][si][:, None, :], -1e9)
        preds.append(s.flatten(1).argmax(1).cpu().numpy())
    rest = S["n"] % 256
    if rest:
        z, m, si = batch(S, order[-rest:], real, sel)
        with amp():
            s = model(z).float()
        s = s.masked_fill(~S["BM"][si][:, None, :], -1e9)
        preds.append(s.flatten(1).argmax(1).cpu().numpy())
    model.train()
    j = np.concatenate(preds)
    return np.stack([j // 100, j % 100], 1)            # (n, 2) predicted layer and patch, in buffer order


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
    real, sel, nlim, types, seed = VARIANTS[name]
    torch.manual_seed(seed); np.random.seed(seed)
    model = (GradDecoder() if real == "grad" else InverseDecoder(cin=len(sel), w=32)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.1)
    m = TR["meta"]
    keep = m[:, 0] < nlim
    if types is not None:
        tix = [TEST_TYPES.index(t) if t in TEST_TYPES else 10 + TRAIN_TYPES.index(t) for t in types]
        keep &= np.isin(m[:, 3], tix)
    idx = np.nonzero(keep)[0]
    per_ep = len(idx) // BS
    n_ep = int(np.ceil(STEPS / per_ep))
    ev_every = max(1, n_ep // 24)
    best, best_state, log, step, ep0 = -1, None, [], 0, 0
    ck = os.path.join(OUT, f"_resume_{name}.pt")                 # resume checkpoint for interrupted runs
    if os.path.exists(ck) and not SMOKE:
        c = torch.load(ck, map_location=DEV, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
        best, best_state, log, step, ep0 = c["best"], c["best_state"], c["log"], c["step"], c["ep"] + 1
        np.random.set_state(c["np_rng"]); torch.set_rng_state(c["torch_rng"].cpu())
        print(f"[{name}] resumed at epoch {ep0} step {step} best {best:.3f}", flush=True)
    for ep in range(ep0, n_ep):
        order = idx[np.random.permutation(len(idx))]
        for z, mb, si in prefetch(TR, order, real, sel, BS):
            if step >= STEPS:
                break
            with amp():
                s = model(z).float()
            s = s.masked_fill(~TR["BM"][si][:, None, :], -1e9).flatten(1)
            lab = torch.from_numpy(mb[:, 1].astype(np.int64)).to(DEV)
            kk = torch.from_numpy(mb[:, 2].astype(np.int64)).to(DEV)
            loss = F.cross_entropy(s, lab * 100 + kk) + 0.5 * F.cross_entropy(torch.logsumexp(s.view(-1, 9, 100), 2), lab)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
        if ep % ev_every == 0 or step >= STEPS:
            tu = score(TU, predict(model, TU, real, sel))
            sc = float(np.mean([tu[t]["joint"] for t in TEST_TYPES]))
            log.append({"epoch": ep, "step": step, "loss": float(loss), "tune": tu})
            print(f"[{name}] ep {ep}/{n_ep} step {step} loss {float(loss):.3f} tune {sc:.3f} t={time.time()-T0:.0f}s", flush=True)
            if sc > best:
                best, best_state = sc, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if not SMOKE:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "best": best, "best_state": best_state, "log": log, "step": step, "ep": ep,
                        "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state()}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
        if step >= STEPS:
            break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT, f"{name}.pt"))
    pred = predict(model, TE, real, sel)
    te = score(TE, pred)
    np.save(os.path.join(OUT, f"{name}_test_pred.npy"), pred)
    save_json({"variant": name, "config": {"realisation": real, "channels": sel, "n_train_slices": nlim,
                                           "train_types": types, "seed": seed, "steps": step, "batch": BS,
                                           "n_train_queries": int(len(idx))},
               "best_tune_joint": best, "test": te, "log": log}, os.path.join(OUT, f"{name}.json"))
    print(f"[{name}] TEST " + json.dumps({k: round(v["joint"], 3) for k, v in te.items()}) +
          f" t={time.time()-T0:.0f}s", flush=True)
    if os.path.exists(ck):
        os.remove(ck)


for name in ONLY:
    if os.path.exists(os.path.join(OUT, f"{name}.json")) and not SMOKE:
        print(f"[{name}] already done, skip"); continue
    train(name)
print(f"done t={time.time()-T0:.0f}s", flush=True)
