"""Summary table of all trained ablation variants (Table 2).

Layer+patch joint accuracy on the 60 fresh test patients, patient-level bootstrap 95% CI, and paired
differences to the "full" model.
Reads results/ablation/*_test_pred.npy; writes results/ablation/e7_summary.json
and prints the table.
Run: python experiments/ablation/summarize.py
"""
import glob, json, os
import numpy as np
from bia.common import *
from bia.core import TEST_TYPES

OUT = set_out("ablation")
meta = np.load(os.path.join(OUT, "buffer", "test_meta.npy"), mmap_mode="r")
n = json.load(open(os.path.join(OUT, "buffer", "test_info.json")))["n"]
meta = np.array(meta[:n])
pat = meta[:, 0]
B = 2000
rng = np.random.RandomState(0)
boots = [rng.choice(np.unique(pat), len(np.unique(pat))) for _ in range(B)]
idx_of = {p: np.nonzero(pat == p)[0] for p in np.unique(pat)}


def ci(hit):
    m = [np.mean(np.concatenate([hit[idx_of[p]] for p in bs])) for bs in boots]
    return [float(np.mean(hit)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


hits = {}
for f in sorted(glob.glob(os.path.join(OUT, "*_test_pred.npy"))):
    name = os.path.basename(f)[:-len("_test_pred.npy")]
    pr = np.load(f)
    hits[name] = (pr[:, 0] == meta[:, 1]) & (pr[:, 1] == meta[:, 2])
summ = {}
for name, h in hits.items():
    summ[name] = {"all": ci(h.astype(float))}
    for ti, t in enumerate(TEST_TYPES):
        sel = meta[:, 3] == ti
        summ[name][t] = float(np.mean(h[sel]))
    if "full" in hits and name != "full":
        d = h.astype(float) - hits["full"].astype(float)
        summ[name]["minus_full"] = ci(d)
save_json(summ, os.path.join(OUT, "e7_summary.json"))
print(f"{'variant':14s} {'all [95% CI]':26s} " + " ".join(f"{t:>8s}" for t in TEST_TYPES) + "   minus full")
for name, s in summ.items():
    if name.startswith("_"):
        continue
    a = s["all"]
    mf = s.get("minus_full")
    print(f"{name:14s} {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]     " + " ".join(f"{s[t]:8.3f}" for t in TEST_TYPES) +
          (f"   {mf[0]:+.3f} [{mf[1]:+.3f}, {mf[2]:+.3f}]" if mf else ""))
