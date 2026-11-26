"""Shared utilities: paths, device, data loading, U-Net forward with edits, and the invertible alignment.

Invertible alignment Q_l (class Star):
    native feature h_l in R^{C x H_l x W_l}, H_l = 160 / s_l
    common-grid feature z_l = Q_l h_l in R^{C x 160 x 160}
    Q_l has orthonormal columns, so its transpose is an exact left inverse (Q_l^T Q_l = I) and the
    round trip preserves the norm.
Two variants are implemented:
    'block'    : block replication scaled by 1/s (exact cell locality)
    'bilinear' : bilinear interpolation (align_corners=False), orthonormalised as B (B^T B)^{-1/2}
"""
import contextlib, json, os
import numpy as np
import torch
import torch.nn.functional as F

from bia.models.unet import UNetBaseline, UNetNoSkip

# ----------------------------------------------------------------------------- paths
# Locations can be overridden with environment variables; defaults are relative to the repository root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BIA_DATA", os.path.join(ROOT, "data", "brats2d"))
CKPT = os.environ.get("BIA_CKPT", os.path.join(ROOT, "checkpoints"))
RESULTS = os.environ.get("BIA_RESULTS", os.path.join(ROOT, "results"))
MANIFEST = os.environ.get("BIA_SPLITS", os.path.join(ROOT, "splits", "manifest.json"))
BASE = os.path.join(RESULTS, "base")          # prototype library, localization network and other shared outputs
OUT = BASE                                    # each script sets its own output folder with set_out()


def set_out(exp):
    """Create results/<exp>/ and make it this script's OUT (returned)."""
    global OUT
    OUT = os.path.join(RESULTS, exp)
    os.makedirs(OUT, exist_ok=True)
    return OUT


def base_path(name):
    return os.path.join(BASE, name)


# ----------------------------------------------------------------------------- device
if os.environ.get("BIA_DEVICE"):
    DEV = torch.device(os.environ["BIA_DEVICE"])
elif torch.cuda.is_available():
    DEV = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEV = torch.device("mps")
else:
    DEV = torch.device("cpu")


# Exact float32 on CUDA: TF32 convolutions add a ~1e-3 (rms) noise floor to the logits, which swamps the
# output change of small interventions.
if DEV.type == "cuda":
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

# highest precision available on the device (MPS has no float64)
HI = torch.float32 if DEV.type == "mps" else torch.float64


def amp():
    """Mixed-precision context for the small decoders (bf16 on CUDA, fp16 on Apple MPS)."""
    if DEV.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if DEV.type == "mps":
        return torch.autocast("mps", dtype=torch.float16)
    return contextlib.nullcontext()


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    elif DEV.type == "mps":
        torch.mps.synchronize()


LAYERS = ["inc", "down1", "down2", "down3", "down4", "up1", "up2", "up3", "up4"]
STRIDE = {"inc": 1, "down1": 2, "down2": 4, "down3": 8, "down4": 16,
          "up1": 8, "up2": 4, "up3": 2, "up4": 1}
SKIP_OF = {"inc": "up4", "down1": "up3", "down2": "up2", "down3": "up1"}  # encoder -> decoder consumer
GRID = 160


def load_model(kind="baseline", seed=123):
    net = (UNetBaseline if kind == "baseline" else UNetNoSkip)(4, 4, bilinear=False)
    sd = torch.load(f"{CKPT}/unet_{kind}/seed_{seed}/best_model.pth", map_location="cpu")
    net.load_state_dict(sd)
    return net.eval().to(DEV)


def load_slice(name, split="test"):
    x = np.load(f"{DATA}/{split}_Image/{name}.npy").astype(np.float32)  # H W 4
    m = np.load(f"{DATA}/{split}_Mask/{name}.npy").astype(np.int64)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous().to(DEV), torch.from_numpy(m).to(DEV)


def pick_slices(split, n, seed, min_tumor=600):
    """One tumour-bearing slice per patient, in a seeded random patient order.
    The ordered lists are stored in the splits manifest (key "<split>_seed<seed>"); the list for a
    given seed is prefix-stable, so the first n entries are returned."""
    key = f"{split}_seed{seed}"
    man = json.load(open(MANIFEST))
    if key not in man or min_tumor != 600:
        raise KeyError(f"{key} (min_tumor={min_tumor}) not in manifest")
    lst = man[key]
    if n > len(lst):
        raise ValueError(f"only {len(lst)} slices available for {key}")
    return lst[:n]


# ----------------------------------------------------------------------------- forward
@torch.no_grad()
def forward(net, x, edits=None, cache=None):
    """Run the U-Net with optional interventions.

    edits: dict  key -> fn(h)->h.  key 'down2' edits the module output (both consumers),
           'down2@skip' edits only the copy sent through the skip connection,
           'down2@deep' edits only the copy sent to the next encoder block.
    cache: clean features of the same input (batch 1) to skip recomputation upstream of edits.
    Returns dict of features (post-edit) and 'logits'.
    """
    edits = edits or {}
    B = x.shape[0]
    first = min([LAYERS.index(k.split("@")[0]) for k in edits] or [len(LAYERS)])
    f = {}

    def get(name, compute):
        i = LAYERS.index(name)
        if cache is not None and i < first:
            h = cache[name].expand(B, *cache[name].shape[1:])
        else:
            h = compute()
        if name in edits:
            h = edits[name](h)
        f[name] = h
        return h

    def br(name, h, path):
        k = f"{name}@{path}"
        return edits[k](h) if k in edits else h

    x1 = get("inc", lambda: net.inc(x))
    x2 = get("down1", lambda: net.down1(br("inc", x1, "deep")))
    x3 = get("down2", lambda: net.down2(br("down1", x2, "deep")))
    x4 = get("down3", lambda: net.down3(br("down2", x3, "deep")))
    x5 = get("down4", lambda: net.down4(br("down3", x4, "deep")))
    u1 = get("up1", lambda: net.up1(x5, br("down3", x4, "skip")))
    u2 = get("up2", lambda: net.up2(u1, br("down2", x3, "skip")))
    u3 = get("up3", lambda: net.up3(u2, br("down1", x2, "skip")))
    u4 = get("up4", lambda: net.up4(u3, br("inc", x1, "skip")))
    f["logits"] = net.outc(u4)
    return f


# ----------------------------------------------------------------------------- invertible alignment
class Star:
    """Invertible alignment Q_l: native grid -> common 160x160 grid (to_common) and its transpose back (to_native)."""

    def __init__(self, mode="block", dtype=torch.float64):
        self.mode, self.dtype = mode, dtype
        self._mats = {}

    def _axis(self, n):
        """Orthonormal 1-D upsampling matrix Q (GRID x n); its transpose Q^T is the exact inverse on range(Q)."""
        if n in self._mats:
            return self._mats[n]
        s = GRID // n
        if self.mode == "block":
            Q = torch.zeros(GRID, n, dtype=self.dtype)
            for i in range(n):
                Q[i * s:(i + 1) * s, i] = 1.0 / np.sqrt(s)
        else:
            eye = torch.eye(n, dtype=self.dtype)[None]  # 1 x n x n  (channels=n basis vectors)
            Bm = F.interpolate(eye, size=GRID, mode="linear", align_corners=False)[0].T  # GRID x n
            G = Bm.T @ Bm
            w, V = torch.linalg.eigh(G)
            Q = Bm @ (V @ torch.diag(w.rsqrt()) @ V.T)  # B (B^T B)^{-1/2}
        self._mats[n] = Q
        return Q

    def cond(self, n):
        if self.mode == "block":
            return 1.0
        eye = torch.eye(n, dtype=torch.float64)[None]
        Bm = F.interpolate(eye, size=GRID, mode="linear", align_corners=False)[0].T
        sv = torch.linalg.svdvals(Bm)
        return float(sv.max() / sv.min())

    def to_common(self, h):  # (..., C, n, n) -> (..., C, GRID, GRID)
        Q = self._axis(h.shape[-1]).to(h.device, h.dtype)
        return Q @ h @ Q.T

    def to_native(self, z, n):  # (..., C, GRID, GRID) -> (..., C, n, n)
        Q = self._axis(n).to(z.device, z.dtype)
        return Q.T @ z @ Q

    def project(self, z, n):  # orthogonal projection onto range(Q_l)
        return self.to_common(self.to_native(z, n))


def patch_mask(p, row, col):
    """Boolean GRID x GRID mask of patch (row, col) for patch size p."""
    m = torch.zeros(GRID, GRID, dtype=torch.bool, device=DEV)
    m[row * p:(row + 1) * p, col * p:(col + 1) * p] = True
    return m


def native_cells(mask, s):
    """Native-cell mask covered by a common-grid mask (cell-aligned patches)."""
    n = GRID // s
    return mask.view(n, s, n, s).any(3).any(1)


def energy_map(d):
    """Per-pixel squared change summed over channels: (B,C,H,W)->(B,H,W)."""
    return (d.to(HI) ** 2).sum(1)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
