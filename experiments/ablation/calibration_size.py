# -*- coding: utf-8 -*-
"""Sensitivity to the number of slices used to build the response prototypes (Fig. 10).

Training-free prototype matching. All data come from the intervention buffers of the fairness / ablation
study (400 training slices x 9 layers x 24 mixed-type interventions; 60 test patients x 4 test types x 9
layers x 4 queries = 8,640 queries), so the analysed network is not run again. For a random subset of N
slices the response prototypes are rebuilt as in Eq. (7) (pooled energy aligned on the patch centre and
averaged, 79x79), and prototype matching (Eq. (9)) is evaluated on the same test set. Localization and
layer accuracy are reported; for N < 400 the subset is drawn 3 times.
Reads results/ablation/buffer/ (written by build_buffer.py); writes
results/sensitivity/s4a_proto_count.json.
Run: python experiments/ablation/calibration_size.py
"""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.engine import NP
from bia.core import TRAIN_TYPES, TEST_TYPES

OUT = set_out("sensitivity")
BUF = os.path.join(RESULTS, "ablation", "buffer")
T0 = time.time()
C79 = 79
# type indices in the buffer: 0-3 = TEST_TYPES; 10+i = TRAIN_TYPES[i]
TIX = {0: "noise", 1: "ablate", 12: "zero"}


def pooled_energy(name):
    """Pooled 40x40 energy of every query, normalised to unit sum (as when building the prototypes); cached in OUT."""
    fn = os.path.join(OUT, f"em_{name}.npy")
    if os.path.exists(fn):
        return np.load(fn)
    dy = np.load(os.path.join(BUF, f"{name}_dy_std.npy"), mmap_mode="r")
    n = json.load(open(os.path.join(BUF, f"{name}_info.json")))["n"]
    em = np.zeros((n, 40, 40), np.float32)
    for i in range(0, n, 512):
        d = torch.from_numpy(np.asarray(dy[i:i + 512])).to(DEV).float()
        e = F.avg_pool2d(d.pow(2).sum(1, keepdim=True), 4)[:, 0]
        e = e / e.flatten(1).sum(1).clamp_min(1e-30)[:, None, None]
        em[i:i + 512] = e.cpu().numpy()
        if i % 20480 == 0:
            print(f"  energy {name} {i}/{n} t={time.time()-T0:.0f}s", flush=True)
    np.save(fn, em)
    return em


em_tr = pooled_energy("train"); em_te = pooled_energy("test")
m_tr = np.load(os.path.join(BUF, "train_meta.npy"))[:len(em_tr)]
m_te = np.load(os.path.join(BUF, "test_meta.npy"))[:len(em_te)]
BM = np.load(os.path.join(BUF, "test_BM.npy"))
n_slices = int(m_tr[:, 0].max()) + 1
print(f"train {len(em_tr)} queries / {n_slices} slices, test {len(em_te)}  t={time.time()-T0:.0f}s", flush=True)


def build_templates(slices):
    sel = np.isin(m_tr[:, 0], slices)
    T = {}
    for li, L in enumerate(LAYERS):
        for tix, t in TIX.items():
            idx = np.nonzero(sel & (m_tr[:, 1] == li) & (m_tr[:, 3] == tix))[0]
            acc = np.zeros((C79, C79)); cnt = np.zeros((C79, C79))
            for i in idx:
                r, q = divmod(int(m_tr[i, 2]), NP)
                y0, x0 = 39 - 4 * r, 39 - 4 * q
                acc[y0:y0 + 40, x0:x0 + 40] += em_tr[i]; cnt[y0:y0 + 40, x0:x0 + 40] += 1
            T[(L, t)] = torch.tensor(acc / np.maximum(cnt, 1), device=DEV, dtype=torch.float32)
    return T


def place(T, ks):
    return torch.stack([T[39 - 4 * (k // NP):79 - 4 * (k // NP), 39 - 4 * (k % NP):79 - 4 * (k % NP)] for k in ks])


def unit_sqrt(a):
    a = a.flatten(1).to(HI).clamp_min(0).sqrt()
    return a / a.norm(dim=1, keepdim=True).clamp_min(1e-30)


OBS = torch.from_numpy(em_te).to(DEV)


def evaluate(T):
    joint = np.zeros(len(em_te), bool); layer = np.zeros(len(em_te), bool)
    for si in np.unique(m_te[:, 0]):
        ks = np.nonzero(BM[si])[0].tolist(); nK = len(ks)
        idx = np.nonzero(m_te[:, 0] == si)[0]
        obs = unit_sqrt(OBS[idx])
        sc = torch.stack([obs @ unit_sqrt(torch.cat([place(T[(L, t)], ks) for L in LAYERS])).T
                          for t in TRAIN_TYPES]).max(0).values
        j = sc.argmax(1).cpu().numpy()
        pl, pk = j // nK, np.array(ks)[j % nK]
        layer[idx] = pl == m_te[idx, 1]; joint[idx] = layer[idx] & (pk == m_te[idx, 2])
    by_type = {t: float(joint[m_te[:, 3] == i].mean()) for i, t in enumerate(TEST_TYPES)}
    return float(joint.mean()), float(layer.mean()), by_type


res = {}
for N in (5, 10, 25, 50, 100, 200, 400):
    reps = 1 if N == n_slices else 3
    runs = []
    for r in range(reps):
        sl = np.arange(n_slices) if N == n_slices else np.random.RandomState(100 * N + r).choice(n_slices, N, replace=False)
        acc, lacc, bt = evaluate(build_templates(sl))
        runs.append({"joint": acc, "layer": lacc, "by_type": bt})
        print(f"N={N} rep {r}: joint {acc:.3f} layer {lacc:.3f} {bt}  t={time.time()-T0:.0f}s", flush=True)
    res[N] = {"runs": runs, "joint_mean": float(np.mean([x["joint"] for x in runs])),
              "joint_min": float(np.min([x["joint"] for x in runs])), "joint_max": float(np.max([x["joint"] for x in runs])),
              "layer_mean": float(np.mean([x["layer"] for x in runs])),
              "queries_per_template": float(N * 24 / 3)}
json.dump(res, open(os.path.join(OUT, "s4a_proto_count.json"), "w"), indent=1)
print(f"done t={time.time()-T0:.0f}s")
