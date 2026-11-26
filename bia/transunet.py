"""TransUNet (R50-ViT-B/16) adapter: a second explained network for BIA (Table 3; same BraTS data and splits).

Import this module before bia.engine and bia.core: it rebinds common.LAYERS / STRIDE / SKIP_OF / forward /
load_model in place, so every helper that reads them (make_delta, run_delta, grad_attributions, cell_masks,
patch_sum, ...) works unchanged.

Analysed layers (9, as for the U-Net); s = downsampling factor with respect to the 160 x 160 input:
  R1  ResNet root conv            (64 ch,  80 x 80, s=2)   -> skip to D3
  R2  ResNet stage 1              (256 ch, 40 x 40, s=4)   -> skip to D2  (zero-padded 39 -> 40 as in the model)
  R3  ResNet stage 2              (512 ch, 20 x 20, s=8)   -> skip to D1
  T6  output of Transformer block 6 of 12   (768 ch, 10 x 10 tokens, s=16)
  T12 output of Transformer block 12 (after the final LayerNorm)  (768 ch, 10 x 10, s=16)
  D1  decoder block 1             (256 ch, 20 x 20, s=8)
  D2  decoder block 2             (128 ch, 40 x 40, s=4)
  D3  decoder block 3             (64 ch,  80 x 80, s=2)
  D4  decoder block 4             (16 ch, 160 x 160, s=1)   -> segmentation head (3x3 conv)
Every stride divides the 16-px patch, so each patch maps to whole native cells (one token at s=16).
The checkpoint is read from $TU_CKPT_DIR (default checkpoints/transunet_r50_vit_b16/seed_<seed>/best_model.pth).
It was trained on the same 876/125/250 patient split with 4 classes.
"""
import os
import torch
import torch.nn.functional as F
from bia import common
from bia.models.transunet.vit_seg_configs import get_r50_b16_config
from bia.models.transunet.vit_seg_modeling import VisionTransformer

TU_LAYERS = ["R1", "R2", "R3", "T6", "T12", "D1", "D2", "D3", "D4"]
TU_STRIDE = {"R1": 2, "R2": 4, "R3": 8, "T6": 16, "T12": 16, "D1": 8, "D2": 4, "D3": 2, "D4": 1}
TU_SKIP_OF = {"R1": "D3", "R2": "D2", "R3": "D1"}
TU_CKPT = os.environ.get("TU_CKPT_DIR", os.path.join(common.CKPT, "transunet_r50_vit_b16"))


def build_transunet():
    cfg = get_r50_b16_config()
    cfg.n_classes = 4
    cfg.n_skip = 3
    cfg.patches.grid = (common.GRID // 16, common.GRID // 16)
    return VisionTransformer(cfg, img_size=common.GRID, num_classes=4, in_channels=4, vis=False)


def load_transunet(kind="transunet", seed=123):
    net = build_transunet()
    sd = torch.load(f"{TU_CKPT}/seed_{seed}/best_model.pth", map_location="cpu")
    net.load_state_dict(sd, strict=True)
    return net.eval().to(common.DEV)


def _tok2map(t):
    B, n, C = t.shape
    s = int(round(n ** 0.5))
    return t.transpose(1, 2).reshape(B, C, s, s)


def _map2tok(m):
    return m.flatten(2).transpose(1, 2)


def forward_tu_raw(net, x, edits=None, cache=None):
    """Same contract as common.forward (edits: key -> fn(h)->h, 'L@skip' / 'L@deep' for R1-R3)."""
    edits = edits or {}
    B = x.shape[0]
    first = min([TU_LAYERS.index(k.split("@")[0]) for k in edits] or [len(TU_LAYERS)])
    f = {}

    def get(name, compute):
        i = TU_LAYERS.index(name)
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

    emb = net.transformer.embeddings
    hyb = emb.hybrid_model
    enc = net.transformer.encoder
    dec = net.decoder

    r1 = get("R1", lambda: hyb.root(x))                                        # B,64,80,80

    def c_r2():
        y = F.max_pool2d(br("R1", r1, "deep"), kernel_size=3, stride=2, padding=0)   # 39 x 39
        y = hyb.body[0](y)
        return F.pad(y, (0, 1, 0, 1))                                          # zero-pad to 40, as in the model
    r2 = get("R2", c_r2)
    r3 = get("R3", lambda: hyb.body[1](br("R2", r2, "deep")[:, :, :39, :39]))  # deep path uses the 39 x 39 map

    def c_t6():
        y = hyb.body[2](br("R3", r3, "deep"))                                  # B,1024,10,10
        t = emb.patch_embeddings(y).flatten(2).transpose(1, 2) + emb.position_embeddings
        for blk in enc.layer[:6]:
            t, _ = blk(t)
        return _tok2map(t)
    t6 = get("T6", c_t6)

    def c_t12():
        t = _map2tok(t6)
        for blk in enc.layer[6:]:
            t, _ = blk(t)
        return _tok2map(enc.encoder_norm(t))
    t12 = get("T12", c_t12)
    d1 = get("D1", lambda: dec.blocks[0](dec.conv_more(t12), skip=br("R3", r3, "skip")))
    d2 = get("D2", lambda: dec.blocks[1](d1, skip=br("R2", r2, "skip")))
    d3 = get("D3", lambda: dec.blocks[2](d2, skip=br("R1", r1, "skip")))
    d4 = get("D4", lambda: dec.blocks[3](d3, skip=None))
    f["logits"] = net.segmentation_head(d4)
    return f


@torch.no_grad()
def forward_tu(net, x, edits=None, cache=None):
    return forward_tu_raw(net, x, edits, cache)


forward_tu.__wrapped__ = forward_tu_raw          # bia.core: fwd_grad = forward.__wrapped__

# ------------------------------------------------------------------ rebind the shared names in bia.common
common.LAYERS[:] = TU_LAYERS
common.STRIDE.clear(); common.STRIDE.update(TU_STRIDE)
common.SKIP_OF.clear(); common.SKIP_OF.update(TU_SKIP_OF)
common.forward = forward_tu
common.load_model = load_transunet
