"""Dumps validation-slice candidate data used to design the selection rule of fix_missed.py (Fig. 13b,c).

Only validation slices are used here; the test slices are never touched when designing the rule. For every
validation slice with a missed-tumour region it stores the clean margin s, the ground truth, the candidate patches
and, per (alpha, layer), the real margin change of every candidate edit (fp16, 160x160) together with the cheap
predictors (sketch energy, first-order AtP terms).

Writes results/fix_missed/val_dump/slice_XXX.pt (about 20 MB each).
Run: python experiments/target/dump_val.py
"""
import os, time, torch
import bia.target as D
from bia.common import *
from bia.core import run_delta

OUT = os.path.join(set_out("fix_missed"), "val_dump"); os.makedirs(OUT, exist_ok=True)
T0 = time.time(); n = 0
for si, nm in enumerate(pick_slices("val", 124, seed=1)):
    if os.path.exists(f"{OUT}/slice_{si:03d}.pt"):
        n += 1; continue
    x, gt = load_slice(nm, "val")
    st = D.prepare(x, gt, 10_000 + si)
    if st is None:
        continue
    rec = {"name": nm, "s": st["s"].half().cpu(), "gt": st["gtt"].cpu(), "ks": st["ks"], "x": x.half().cpu(), "c": {}}
    for a in D.ALPHAS:
        for L in LAYERS:
            delta, q = D.candidates(st, L, a)
            ds = D.margin(run_delta(D.net, x, st["cache"], L, delta)["logits"]) - st["s"]
            rec["c"][(L, a)] = {"ds": ds.half().cpu(), "E": D.energy(st, L, delta).cpu(),
                                "atpT": (st["gT"][L] * delta).flatten(1).sum(1).cpu(),
                                "atpO": (st["gO"][L] * delta).flatten(1).sum(1).cpu(),
                                "atpD": (st["gD"][L] * delta).flatten(1).sum(1).cpu()}
    torch.save(rec, f"{OUT}/slice_{si:03d}.pt"); n += 1
    if DEV.type == "mps":
        torch.mps.empty_cache()
    print(f"{si} n={n} t={time.time()-T0:.0f}s", flush=True)
print("done", n)
