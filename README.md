**English** | [中文](README_zh.md)

# BIA: Bidirectional Intervention Analysis for Medical Image Segmentation Networks

BIA explains *how* a frozen segmentation network arrives at its output by intervening on its intermediate layers. It answers two questions:

- **Forward:** if a given layer is changed at a given location, how will the segmentation output change?
- **Backward:** given only an observed change of the output, which layer and which location did it come from?

The analysis sits entirely outside the network: it reads intermediate features, injects interventions and reads the output change, without modifying any parameter of the explained model.

## Method

| Component | What it does | Code |
|---|---|---|
| Invertible multi-layer alignment | Maps the features of every analysed layer onto a common 160 x 160 grid with an injective, norm-preserving operator Q_l. An intervention defined on a 16-px patch of this grid is mapped back to the original layer exactly, so interventions are comparable across layers of different resolution. | `bia/common.py` (`Star`, `forward`), `bia/engine.py` |
| Intervention response model | Builds response prototypes per layer and intervention type by centring and averaging output-change energy maps over calibration images. Shifting a prototype to a patch predicts the output change of a new intervention without running the network. | `experiments/localization/train.py`, `bia/core.py` |
| Intervention source localization | Infers the source layer and patch of an intervention from the output change alone, with three solvers: prototype matching (training-free), a learned localization network, and a forward re-check that simulates the top candidates using forward passes only. | `bia/core.py` (`InverseDecoder`), `experiments/localization/`, `experiments/bitflip/` |

## Repository structure

```
BIA-MedSeg/
├── bia/                        core library
│   ├── common.py               paths, data loading, models, forward pass with interventions, alignment operator
│   ├── engine.py               patch interventions on the common grid
│   ├── core.py                 intervention types, localization network, attribution baselines, metrics
│   ├── variants.py             training variants used in the ablation study
│   ├── target.py               response model used to select target-driven interventions
│   ├── transunet.py            TransUNet adapter (layer list and forward pass)
│   └── models/                 U-Net and vendored TransUNet
├── experiments/
│   ├── alignment/              exactness and expressibility of the alignment
│   ├── response/               output footprints, prototypes, forward fidelity
│   ├── localization/           prototype library, localization network, main comparison
│   ├── ablation/               fairness control, ablations, sensitivity, any-position interventions
│   ├── transunet/              the same protocol on TransUNet
│   ├── analysis/               prototypes across different networks
│   ├── bitflip/                localization of hardware bit flips
│   └── target/                 target-driven choice of intervention layer and location
├── splits/                     patient split and fixed slice order
├── requirements.txt
└── pyproject.toml
```

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/mingsp/BIA-MedSeg.git
cd BIA-MedSeg
pip install -r requirements.txt
pip install -e .
```

The experiments were run with PyTorch 2.1 on a single NVIDIA RTX 4090 (TF32 disabled) and with PyTorch 2.x on Apple silicon (MPS). The device is selected automatically (CUDA, then MPS, then CPU) and can be set with `BIA_DEVICE`. `experiments/alignment/exactness.py` works in float64 and therefore needs CUDA or CPU.

## Data and models

**Data.** BraTS 2023 adult glioma (BraTS-GLI), four MRI sequences (T1, T1ce, T2, FLAIR). For each patient one axial slice with at least 600 tumour pixels is cropped to 160 x 160. Patients are split 876 / 125 / 250 into train / validation / test (`splits/brats2023_gli_seed42.json`); `splits/manifest.json` fixes the slice order used by every experiment. Expected layout:

```
data/brats2d/
├── train_Image/<case>.npy     float32, 160 x 160 x 4
├── train_Mask/<case>.npy      int, 160 x 160, labels 0-3
├── val_Image/   val_Mask/
└── test_Image/  test_Mask/
```

**Explained networks.** The segmentation networks are trained beforehand and kept frozen:

```
checkpoints/
├── unet_baseline/seed_123/best_model.pth      main U-Net
├── unet_baseline/seed_3407/best_model.pth     U-Net trained with another seed
├── unet_noskip/seed_123/best_model.pth        U-Net without skip connections
└── transunet_r50_vit_b16/seed_123/best_model.pth
```

All locations can be changed with environment variables:

| Variable | Default |
|---|---|
| `BIA_DATA` | `data/brats2d` |
| `BIA_CKPT` | `checkpoints` |
| `BIA_RESULTS` | `results` |
| `BIA_SPLITS` | `splits/manifest.json` |
| `TU_CKPT_DIR` | `checkpoints/transunet_r50_vit_b16` |
| `BIA_DEVICE` | automatic (`cuda`, then `mps`, then `cpu`) |

## Reproducing the results

Every script writes to `results/<name>/`. Shared outputs (prototype library, localization network, per-query results) are written to `results/base/` and read by the later steps, so run the steps in this order.

**1. Response model, localization network and main comparison**

```bash
python experiments/alignment/exactness.py
python experiments/response/footprints.py
python experiments/response/summarize_footprints.py
python experiments/localization/train.py
python experiments/localization/evaluate.py | tee results/base/e6_eval.log
python experiments/localization/summarize.py
```

**2. Experiments**

| Paper | Script | Output |
|---|---|---|
| Sec. 4.2.1, Fig. 5: alignment exactness and expressibility | `experiments/alignment/exactness.py`, `experiments/alignment/expressibility.py` | `base/`, `expressibility/` |
| Sec. 4.2.2, Fig. 6: response prototypes, spread and gain | `experiments/response/footprints.py`, `summarize_footprints.py` | `base/` |
| Sec. 4.2.2, Fig. 7: forward fidelity | `experiments/response/fidelity.py` | `fidelity/` |
| Sec. 4.2.3, Table 1, Fig. 8: localization on U-Net | `experiments/localization/evaluate.py`, `more_baselines.py`, `timing.py` | `base/`, `more_baselines/`, `sensitivity/` |
| Sec. 4.2.4, Table 2: fairness control and ablations | `experiments/ablation/build_buffer.py`, then `train_eval.py`, `summarize.py`; `input_channels.py`, `summarize_input_channels.py`; `any_position.py`, `any_position_atp.py` | `ablation/`, `input_channels/`, `any_position/` |
| Sec. 4.2.4, Fig. 10: sensitivity | `experiments/ablation/calibration_size.py`, `amplitude.py` (after `build_buffer.py`) | `sensitivity/` |
| Sec. 4.2.5, Table 3: TransUNet | `experiments/transunet/run.py`, `hires_baseline.py` | `transunet/` |
| Sec. 4.3.1, Fig. 11: skip connections and network correspondence | `experiments/response/footprints.py`, `experiments/analysis/model_specificity.py`, `experiments/transunet/run.py` | `base/`, `model_specificity/`, `transunet/` |
| Sec. 4.3.2, Table 4: bit-flip localization | `H1_VERIFY=1 H1_BUDGETS=2,5,10,20,40 python experiments/bitflip/bitflip.py` (after the ablation step), `CUDNN_OFF=1 python experiments/bitflip/forward_only.py` | `bitflip/`, `bitflip_forward/` |
| Sec. 4.3.3, Fig. 13: target-driven interventions | `experiments/target/ring.py`; `dump_val.py`, then `fix_missed.py` | `ring/`, `fix_missed/` |

`forward_only.py` can be split across processes with `H2_SHARD=i/n` (i = 0 ... n-1) and resumes from per-patient checkpoints.

## Evaluation protocol

All main results are reported on 60 test patients that are never used for building prototypes, training or choosing hyper-parameters. Prototypes and the localization network are built from 300 training and 100 validation patients; 24 further validation patients are used for hyper-parameter selection. Interventions act on 16-px patches of the common grid whose brain area exceeds one half; test interventions are noise, mean ablation, gain and partial-channel noise, of which the last two are never seen when building prototypes or training. A query is counted as correct only if both the layer and the patch are correct. Confidence intervals are 95% patient-level bootstrap intervals.

## Third-party code

`bia/models/transunet/` contains files from [TransUNet](https://github.com/Beckschen/TransUNet) (Apache License 2.0); see `bia/models/transunet/NOTICE.md` for the local changes. The LRP baseline uses [Zennit](https://github.com/chr5tphr/zennit).
