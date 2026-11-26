# -*- coding: utf-8 -*-
"""Per-query time of the added baselines (Table 1).

Same protocol as the timing of the other methods: 5 test patients x 9 layers x 2 noise interventions
(90 queries, batch size 1). Given the observed output change dy, the time is measured until (layer, patch)
is returned for
  gradient methods (one backward pass)  re-measured here for a same-session reference;
  LRP                  zennit EpsilonPlusFlat (BatchNorm merged): one forward and one modified backward;
  Seg-HiRes-Grad CAM   one forward and one backward;
  exhaustive matching  with three references (noise, mean ablation, zero): one network run per reference
                       for every brain candidate at every layer.

Reads results/base/e6_rows.pkl; writes results/sensitivity/s4b2_timing_new.json.
Run: python experiments/localization/timing.py
"""
import copy, os, pickle, time, statistics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bia.common import *
from bia.core import brain_mask_patches, make_delta, run_delta, grad_attributions, patch_sum, fwd_grad
from zennit.composites import EpsilonPlusFlat
from zennit.canonizers import SequentialMergeBatchNorm

OUT = set_out("sensitivity")
net = load_model("baseline", 123)
for p in net.parameters():
    p.requires_grad_(False)
net_lrp = copy.deepcopy(net)
for m in net_lrp.modules():
    if isinstance(m, nn.ReLU):
        m.inplace = False
leg = pickle.load(open(base_path("e6_rows.pkl"), "rb"))
names = leg["test"][:5]


def t_ms(fn, rep=1):
    sync(); t0 = time.perf_counter()
    for _ in range(rep):
        out = fn()
    sync()
    return (time.perf_counter() - t0) * 1000 / rep, out


def keep_into(hs):
    def mk(L):
        def fn(h):
            h.retain_grad(); hs[L] = h
            return h
        return fn
    return {L: mk(L) for L in LAYERS}


res = {k: [] for k in ("Gradient (one backward)", "LRP", "Seg-HiRes-Grad CAM", "Exhaustive matching (3 types)")}
gen = torch.Generator(device=DEV)
for si, nm in enumerate(names):
    x, _ = load_slice(nm, "test")
    cache = forward(net, x[None])
    ks_all = torch.nonzero(brain_mask_patches(x)).flatten()
    rng = np.random.RandomState(si)
    for li, L in enumerate(LAYERS):
        ks = rng.choice(ks_all.cpu().numpy(), 2).tolist()
        gen.manual_seed(int(rng.randint(1 << 30)))
        d = make_delta(cache[L], L, ks, ["noise"] * 2, gen)
        dys = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
        for b in range(2):
            dy = dys[b:b + 1]

            def f_grad():
                return grad_attributions(net, x, cache, dy, ig_steps=0)

            def f_lrp():
                hs = {}
                comp = EpsilonPlusFlat(canonizers=[SequentialMergeBatchNorm()])
                with comp.context(net_lrp) as mod, torch.enable_grad():
                    f = fwd_grad(mod, x[None].clone().requires_grad_(True), edits=keep_into(hs))
                    (f["logits"] * dy).sum().backward()
                S = torch.stack([patch_sum(hs[L2].grad.abs().sum(1), L2) for L2 in LAYERS], 1)
                return int(S[:, :, ks_all].flatten(1).argmax())

            def f_hires():
                hs = {}
                with torch.enable_grad():
                    f = fwd_grad(net, x[None].clone().requires_grad_(True), edits=keep_into(hs))
                    (f["logits"] * dy).sum().backward()
                S = torch.stack([patch_sum(F.relu((hs[L2].grad * hs[L2]).sum(1)), L2) for L2 in LAYERS], 1)
                return int(S[:, :, ks_all].flatten(1).argmax())

            def f_brute3():
                for t in ("noise", "ablate", "zero"):
                    for L2 in LAYERS:
                        kk = ks_all.cpu().numpy().tolist()
                        dd = make_delta(cache[L2], L2, kk, [t] * len(kk), gen)
                        run_delta(net, x, cache, L2, dd)

            if si == 0 and li == 0 and b == 0:            # warm-up
                for f in (f_grad, f_lrp, f_hires, f_brute3):
                    t_ms(f)
            res["Gradient (one backward)"].append(t_ms(f_grad, 3)[0])
            res["LRP"].append(t_ms(f_lrp, 3)[0])
            res["Seg-HiRes-Grad CAM"].append(t_ms(f_hires, 3)[0])
            if b == 0:
                res["Exhaustive matching (3 types)"].append(t_ms(f_brute3)[0])
    print(si, nm, {k: round(statistics.median(v), 1) for k, v in res.items()}, flush=True)
summ = {k: {"median_ms": statistics.median(v), "n": len(v)} for k, v in res.items()}
summ["device"] = str(DEV)
save_json(summ, os.path.join(OUT, "s4b2_timing_new.json"))
print(summ)
