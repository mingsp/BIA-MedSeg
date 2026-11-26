"""Training of the localization network and construction of the prototype library (Table 1).

Builds an intervention buffer on non-test patients (300 train-split + 100 val slices; 24 other val
slices serve as the tune set) and learns
  (a) the response prototypes per (layer, intervention type), saved as e6_templates.npz;
  (b) the localization network (InverseDecoder, g_phi): output change -> (layer, patch), saved as e6_invnet.pt.
The checkpoint with the best joint (layer, patch) accuracy on the tune set is kept.

Writes results/base/e6_templates.npz, e6_invnet.pt, e6_train.json, e6_tuneQ.pt, e6_tuneQ_meta.pt.
Run: python experiments/localization/train.py
"""
import time, json
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.engine import NP
from bia.core import *
OUT = set_out("base")

net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)
T0 = time.time()
val = pick_slices("val", 124, seed=1)
train_names = [("train", n) for n in pick_slices("train", 300, seed=3)] + [("val", n) for n in val[:100]]
tune_names = [("val", n) for n in val[100:124]]
PER = 24                          # interventions per (slice, layer)
C79 = 79

dy_buf, lab_L, lab_k, lab_s, lab_t = [], [], [], [], []
X, PROB, BM = [], [], []
tmpl = {(L, t): [np.zeros((C79, C79)), np.zeros((C79, C79))] for L in LAYERS for t in TRAIN_TYPES}
gen = torch.Generator(device=DEV)
rng = np.random.RandomState(0)

for si, (split, nm) in enumerate(train_names):
    x, _ = load_slice(nm, split)
    cache = forward(net, x[None])
    bm = brain_mask_patches(x)
    ks_all = torch.nonzero(bm).flatten().cpu().numpy()
    X.append(x.half().cpu()); PROB.append(cache["logits"].softmax(1)[0].half().cpu()); BM.append(bm.cpu())
    for li, L in enumerate(LAYERS):
        ks = rng.choice(ks_all, PER)
        types = [TRAIN_TYPES[i] for i in rng.randint(len(TRAIN_TYPES), size=PER)]
        kap = np.exp(rng.uniform(np.log(0.25), np.log(2.0), PER))
        gen.manual_seed(int(rng.randint(1 << 30)))
        delta = make_delta(cache[L], L, ks, types, gen, kappa=kap)
        delta = torch.cat([delta, torch.zeros_like(delta[:1])])           # clean reference in the same batch
        out = run_delta(net, x, cache, L, delta)["logits"]
        dy = out[:-1] - out[-1:]
        rms = dy.pow(2).mean((1, 2, 3)).sqrt()
        ok = (rms > 1e-6).cpu().numpy()
        ref = float((out[-1] ** 2).sum(0).mean())
        em = e40(dy, ref).cpu().numpy()
        for b in range(PER):
            if not ok[b]:
                continue
            dy_buf.append(dy[b].half().cpu()); lab_L.append(li); lab_k.append(int(ks[b])); lab_s.append(si)
            lab_t.append(TRAIN_TYPES.index(types[b]))
            r, q = divmod(int(ks[b]), NP)
            y0, x0 = 39 - 4 * r, 39 - 4 * q
            tot = em[b].sum() + 1e-30
            tmpl[(L, types[b])][0][y0:y0 + 40, x0:x0 + 40] += em[b] / tot
            tmpl[(L, types[b])][1][y0:y0 + 40, x0:x0 + 40] += 1
    if si % 50 == 0:
        print(f"buffer {si}/{len(train_names)} n={len(dy_buf)} t={time.time()-T0:.0f}s", flush=True)

np.savez_compressed(f"{OUT}/e6_templates.npz",
                    **{f"{L}__{t}": v[0] / np.maximum(v[1], 1) for (L, t), v in tmpl.items()})
DY = torch.stack(dy_buf)                       # N,4,160,160 fp16 (CPU)
LL = torch.tensor(lab_L); KK = torch.tensor(lab_k); SS = torch.tensor(lab_s)
X = torch.stack(X).to(DEV); PROB = torch.stack(PROB).to(DEV); BM = torch.stack(BM).to(DEV)
N = len(DY)
print(f"buffer done N={N} ({DY.numel()*2/1e9:.1f} GB) t={time.time()-T0:.0f}s", flush=True)

# tune set: fixed queries of all 4 intervention types
def make_queries(names, per_type_layer, seed):
    rng = np.random.RandomState(seed)
    Q = []
    for si, (split, nm) in enumerate(names):
        x, _ = load_slice(nm, split)
        cache = forward(net, x[None])
        bm = brain_mask_patches(x)
        ks_all = torch.nonzero(bm).flatten().cpu().numpy()
        for t in TEST_TYPES:
            for li, L in enumerate(LAYERS):
                ks = rng.choice(ks_all, per_type_layer)
                gen.manual_seed(int(rng.randint(1 << 30)))
                delta = make_delta(cache[L], L, ks, [t] * len(ks), gen)
                delta = torch.cat([delta, torch.zeros_like(delta[:1])])
                out = run_delta(net, x, cache, L, delta)["logits"]
                dy = out[:-1] - out[-1:]
                for b in range(len(ks)):
                    if dy[b].pow(2).mean().sqrt() > 1e-6:
                        Q.append((si, t, li, int(ks[b]), dy[b].half().cpu()))
    return Q


tuneQ = make_queries(tune_names, 4, seed=123)
tX = torch.stack([load_slice(n, s)[0] for s, n in tune_names]).half()
tP = torch.stack([forward(net, load_slice(n, s)[0][None])["logits"].softmax(1)[0] for s, n in tune_names]).half()
tB = torch.stack([brain_mask_patches(load_slice(n, s)[0]) for s, n in tune_names])
print(f"tune queries {len(tuneQ)} t={time.time()-T0:.0f}s", flush=True)


def dec_in(dy, prob, x):
    dy = dy.float(); prob = prob.float(); x = x.float()
    rms = dy.pow(2).mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
    e = dy.pow(2).sum(1, keepdim=True) / rms.pow(2)
    return torch.cat([dy / rms, torch.log1p(e), prob, x, torch.log(rms).expand_as(e) / 5.0], 1)


def evaluate(model, Q, Xs, Ps, Bs):
    model.eval()
    res = []
    with torch.no_grad(), amp():
        for i in range(0, len(Q), 256):
            chunk = Q[i:i + 256]
            si = torch.tensor([q[0] for q in chunk])
            dy = torch.stack([q[4] for q in chunk]).to(DEV)
            z = dec_in(dy, Ps[si].to(DEV), Xs[si].to(DEV))
            s = model(z).float()
            s = s.masked_fill(~Bs[si].to(DEV)[:, None, :], -1e9)
            j = s.flatten(1).argmax(1).cpu().numpy()
            for q, jj in zip(chunk, j):
                res.append((q[1], q[2], q[3], jj // 100, jj % 100))
    model.train()
    out = {}
    for t in TEST_TYPES:
        r = [v for v in res if v[0] == t]
        out[t] = {"layer": float(np.mean([a[1] == a[3] for a in r])),
                  "joint": float(np.mean([a[1] == a[3] and a[2] == a[4] for a in r]))}
    return out


# train the localization network
model = InverseDecoder(cin=14, w=32).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
EPOCHS, BS = 24, 192
steps = EPOCHS * (N // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
best, best_state, log = -1, None, []
step = 0
for ep in range(EPOCHS):
    perm = torch.randperm(N)
    for i in range(0, N - BS + 1, BS):
        idx = perm[i:i + BS]
        si = SS[idx].to(DEV)
        z = dec_in(DY[idx].to(DEV, non_blocking=True), PROB[si], X[si])
        tgt = (LL[idx] * 100 + KK[idx]).to(DEV)
        with amp():
            s = model(z).float()
        s = s.masked_fill(~BM[si][:, None, :], -1e9).flatten(1)
        loss = F.cross_entropy(s, tgt) + 0.5 * F.cross_entropy(torch.logsumexp(s.view(-1, 9, 100), 2), LL[idx].to(DEV))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); step += 1
    ev = evaluate(model, tuneQ, tX, tP, tB)
    score = np.mean([v["joint"] for v in ev.values()])
    log.append({"epoch": ep, "loss": float(loss), "tune": ev})
    print(f"ep {ep} loss {float(loss):.3f} tune {json.dumps({k: round(v['layer'],3) for k, v in ev.items()})} "
          f"joint {score:.3f} t={time.time()-T0:.0f}s", flush=True)
    if score > best:
        best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
torch.save(best_state, f"{OUT}/e6_invnet.pt")
save_json({"log": log, "best_tune_joint": best, "N": N, "train": train_names, "tune": tune_names},
          f"{OUT}/e6_train.json")
torch.save({"tuneQ": [(a, b, c, d) for a, b, c, d, _ in tuneQ]}, f"{OUT}/e6_tuneQ_meta.pt")
torch.save({"dy": torch.stack([q[4] for q in tuneQ]), "meta": [(a, b, c, d) for a, b, c, d, _ in tuneQ],
            "names": tune_names}, f"{OUT}/e6_tuneQ.pt")
print(f"done t={time.time()-T0:.0f}s")
