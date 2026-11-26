"""Model specificity of the response model (Fig. 11).

Networks: the U-Net explained everywhere else (seed 123), a U-Net trained with another seed (3407),
a U-Net without skip connections (seed 123), and a U-Net with random weights (parameter-randomisation check).
For every network M:
  1. calibrate prototypes on M (100 training slices x 9 layers x 8 mixed interventions);
  2. describe M: output footprint radius and gain per layer;
  3. localize on 60 fresh test patients (noise and ablate, 4 interventions per layer) with
     PM-own (prototypes calibrated on M), PM-transfer (prototypes of the original U-Net applied to M),
     InvNet-transfer (localization network trained on the original U-Net applied to M) and AtP-noise;
  4. per-layer cosine between the prototypes of M and those of the original U-Net.

Reads results/base/e6_templates.npz-style inputs (e6_invnet.pt, e6_rows.pkl); writes results/model_specificity/x1_summary.json.
Run: python experiments/analysis/model_specificity.py
"""
import json, time
import numpy as np
import torch
import torch.nn.functional as F
from bia.common import *
from bia.engine import P, NP
from bia.core import (TRAIN_TYPES, brain_mask_patches, make_delta, run_delta, e40, cos_amp, InverseDecoder,
                    grad_attributions, normalise)

OUT = set_out("model_specificity")
T0 = time.time()
C79 = 79


def get_net(tag):
    if tag == "random":
        torch.manual_seed(0)
        from bia.models.unet import UNetBaseline
        return UNetBaseline(4, 4, bilinear=False).eval().to(DEV)
    kind, seed = tag.split("_")
    return load_model(kind, int(seed))


def place(T, k):
    r, q = divmod(int(k), NP)
    return T[39 - 4 * r:79 - 4 * r, 39 - 4 * q:79 - 4 * q]


def calibrate(net, names):
    acc = {(L, t): [np.zeros((C79, C79)), np.zeros((C79, C79))] for L in LAYERS for t in TRAIN_TYPES}
    stats = {L: {"radius": [], "gain": []} for L in LAYERS}
    rng = np.random.RandomState(0); gen = torch.Generator(device=DEV)
    yy, xx = torch.meshgrid(torch.arange(GRID, device=DEV, dtype=HI), torch.arange(GRID, device=DEV, dtype=HI), indexing="ij")
    for split, nm in names:
        x, _ = load_slice(nm, split)
        cache = forward(net, x[None])
        ref = float((cache["logits"] ** 2).sum(1).mean())
        ks_all = torch.nonzero(brain_mask_patches(x)).flatten().cpu().numpy()
        for L in LAYERS:
            ks = rng.choice(ks_all, 8)
            types = [TRAIN_TYPES[i] for i in rng.randint(3, size=8)]
            gen.manual_seed(int(rng.randint(1 << 30)))
            d = make_delta(cache[L], L, ks, types, gen)
            dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
            em = e40(dy, ref).cpu().numpy()
            e = (dy.to(HI) ** 2).sum(1)
            src = (d.to(HI) ** 2).sum((1, 2, 3)) / (cache[L].to(HI) ** 2).sum(1).mean()
            for b in range(8):
                tot = em[b].sum()
                if tot <= 1e-12:
                    continue
                r, q = divmod(int(ks[b]), NP)
                y0, x0 = 39 - 4 * r, 39 - 4 * q
                acc[(L, types[b])][0][y0:y0 + 40, x0:x0 + 40] += em[b] / tot
                acc[(L, types[b])][1][y0:y0 + 40, x0:x0 + 40] += 1
                if types[b] == "noise":
                    cy, cx = r * P + P / 2 - 0.5, q * P + P / 2 - 0.5
                    eb = e[b]
                    stats[L]["radius"].append(float(torch.sqrt((eb * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum() / eb.sum())))
                    stats[L]["gain"].append(float(eb.sum() / ref / src[b]))
    T = {k: torch.tensor(v[0] / np.maximum(v[1], 1), device=DEV, dtype=torch.float32) for k, v in acc.items()}
    desc = {L: {"radius_median": float(np.median(v["radius"])), "gain_median": float(np.median(v["gain"]))}
            for L, v in stats.items()}
    return T, desc


src_net = get_net("baseline_123")
inv = InverseDecoder(cin=14, w=32).to(DEV)
inv.load_state_dict(torch.load(base_path("e6_invnet.pt"), map_location=DEV)); inv.eval()
leg = __import__("pickle").load(open(base_path("e6_rows.pkl"), "rb"))
calib_names = [("train", n) for n in pick_slices("train", 400, seed=3)[300:400]]    # disjoint from the training slices of the localization network
used = set(pick_slices("test", 40, seed=2))
test_names = [n for n in pick_slices("test", 243, seed=2) if n not in used][:60]


def dec_in(dy, prob, x):
    rms = dy.pow(2).mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
    e = dy.pow(2).sum(1, keepdim=True) / rms.pow(2)
    return torch.cat([dy / rms, torch.log1p(e), prob.expand(dy.shape[0], -1, -1, -1),
                      x[None].expand(dy.shape[0], -1, -1, -1), torch.log(rms).expand_as(e) / 5.0], 1)


res = {}
T_src = None
for tag in ["baseline_123", "baseline_3407", "noskip_123", "random"]:
    net = src_net if tag == "baseline_123" else get_net(tag)
    for p_ in net.parameters():
        p_.requires_grad_(False)
    T_own, desc = calibrate(net, calib_names)
    if tag == "baseline_123":
        T_src = T_own
    print(f"[{tag}] calibrated t={time.time()-T0:.0f}s  radius={[round(desc[L]['radius_median'],1) for L in LAYERS]}", flush=True)
    sim = {L: float(F.cosine_similarity(T_own[(L, 'noise')].flatten()[None].sqrt(), T_src[(L, 'noise')].flatten()[None].sqrt()))
           for L in LAYERS}
    rng = np.random.RandomState(2026); gen = torch.Generator(device=DEV)
    hits = {m: [] for m in ("PM-own", "PM-transfer", "InvNet-transfer", "AtP-noise")}
    layer_hits = {m: [] for m in hits}
    for si, nm in enumerate(test_names):
        x, _ = load_slice(nm, "test")
        cache = forward(net, x[None])
        ref = float((cache["logits"] ** 2).sum(1).mean())
        prob = cache["logits"].softmax(1)
        ks_all = torch.nonzero(brain_mask_patches(x)).flatten().cpu().numpy().tolist(); nK = len(ks_all)
        D = {which: {t: torch.cat([torch.stack([place(TT[(L, t)], k) for k in ks_all]) for L in LAYERS]) for t in TRAIN_TYPES}
             for which, TT in (("own", T_own), ("src", T_src))}
        for t in ("noise", "ablate"):
            for li, L in enumerate(LAYERS):
                ks = rng.choice(ks_all, 4)
                gen.manual_seed(int(rng.randint(1 << 30)))
                d = make_delta(cache[L], L, ks, [t] * 4, gen)
                dy = run_delta(net, x, cache, L, d)["logits"] - cache["logits"]
                keep = (dy.pow(2).mean((1, 2, 3)).sqrt() > 1e-6).cpu().numpy()
                obs = e40(dy, ref)
                sc = {"PM-own": torch.stack([cos_amp(obs, D["own"][tt]) for tt in TRAIN_TYPES]).max(0).values,
                      "PM-transfer": torch.stack([cos_amp(obs, D["src"][tt]) for tt in TRAIN_TYPES]).max(0).values}
                with torch.no_grad(), amp():
                    s = inv(dec_in(dy, prob, x)).float()
                sc["InvNet-transfer"] = s[:, :, ks_all].flatten(1).to(HI)
                S = grad_attributions(net, x, cache, dy, ig_steps=0)
                sc["AtP-noise"] = normalise(S["gsig"], "sum")[:, :, ks_all].flatten(1)
                for b in range(4):
                    if not keep[b]:
                        continue
                    tj = li * nK + ks_all.index(int(ks[b]))
                    for m, v in sc.items():
                        j = int(v[b].argmax())
                        hits[m].append(j == tj); layer_hits[m].append(j // nK == li)
        if DEV.type == "mps":
            torch.mps.empty_cache()
    res[tag] = {"describe": desc, "template_cos_to_unet123": sim,
                "joint": {m: float(np.mean(v)) for m, v in hits.items()},
                "layer": {m: float(np.mean(v)) for m, v in layer_hits.items()}, "n": len(hits["PM-own"])}
    print(f"[{tag}] joint " + json.dumps({m: round(v, 3) for m, v in res[tag]['joint'].items()}) +
          f"  template-cos {[round(v, 2) for v in sim.values()]} t={time.time()-T0:.0f}s", flush=True)
    save_json(res, f"{OUT}/x1_summary.json")
print(f"done t={time.time()-T0:.0f}s")
