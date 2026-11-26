# -*- coding: utf-8 -*-
"""Summary of the input-channel ablation: localization accuracy of each variant and its difference to "full".

Patient-level paired bootstrap (1,000 resamples), 95% confidence intervals.
Reads results/input_channels/<variant>_test_pred.npy and test_meta.npy; writes
results/input_channels/e8_summary.json.
Run: python experiments/ablation/summarize_input_channels.py
"""
import os
import numpy as np
from bia.common import RESULTS, save_json

R = os.path.join(RESULTS, "input_channels")
meta = np.load(os.path.join(R, "test_meta.npy"))
names = ["full", "no_logr", "no_energy", "no_prob", "no_img", "out_only"]
ok = {}
for n in names:
    p = os.path.join(R, f"{n}_test_pred.npy")
    if os.path.exists(p):
        pr = np.load(p)
        ok[n] = ((pr[:, 0] == meta[:, 1]) & (pr[:, 1] == meta[:, 2])).astype(float)
pats = np.unique(meta[:, 0])
rows = [np.nonzero(meta[:, 0] == s)[0] for s in pats]
rng = np.random.RandomState(0)
B = [rng.randint(len(pats), size=len(pats)) for _ in range(1000)]


def boot(v):
    per = np.array([v[r].sum() for r in rows]); cnt = np.array([len(r) for r in rows])
    bs = [per[b].sum() / cnt[b].sum() for b in B]
    return [float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


out = {}
for n, v in ok.items():
    d = {"acc": boot(v)}
    if n != "full" and "full" in ok:
        d["diff_vs_full"] = boot(v - ok["full"])
    d["per_type"] = {t: float(v[meta[:, 3] == i].mean()) for i, t in enumerate(["noise", "ablate", "gain", "chansub"])}
    out[n] = d
    print(n, {k: np.round(x, 3).tolist() if isinstance(x, list) else {kk: round(vv, 3) for kk, vv in x.items()} for k, x in d.items()})
save_json(out, os.path.join(R, "e8_summary.json"))
