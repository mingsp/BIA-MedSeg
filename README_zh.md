[English](README.md) | **中文**

# BIA：面向医学图像分割网络的双向干预分析

BIA 对参数冻结的分割网络的中间层施加干预，以此说明网络内部各层如何影响分割结果。它回答两个方向的问题：

- **正向：** 在某一层的某个位置施加干预，分割输出会怎样变化？
- **逆向：** 只给出输出的变化，它来自哪一层、哪个位置？

分析系统完全挂接在网络外部，只读取中间特征、注入干预、读取输出变化，不修改被解释网络的任何参数。

## 方法

| 模块 | 作用 | 代码 |
|---|---|---|
| 可逆多层特征对齐 | 用单射、保范数的对齐映射 Q_l 把各层特征映射到统一的 160 x 160 网格上。在网格的 16 像素图像块上定义的干预可以精确映射回原层，不同分辨率的层上的干预因此可以直接比较。 | `bia/common.py`（`Star`、`forward`）、`bia/engine.py` |
| 干预响应模型 | 在校准图像上把输出变化能量图以图像块为中心对齐后求平均，按层和干预类型构建响应原型。把原型平移到某个图像块即可预测一次新干预的输出变化，不需要运行网络。 | `experiments/localization/train.py`、`bia/core.py` |
| 干预来源定位 | 只依据输出变化推断干预所在的层和图像块，提供三种解法：免训练的原型匹配、学习型的来源定位网络，以及只用前向计算模拟候选位置的前向复核。 | `bia/core.py`（`InverseDecoder`）、`experiments/localization/`、`experiments/bitflip/` |

## 目录结构

```
BIA-MedSeg/
├── bia/                        核心库
│   ├── common.py               路径、数据读取、模型、可施加干预的前向计算、对齐映射
│   ├── engine.py               统一网格上的图像块干预
│   ├── core.py                 干预类型、来源定位网络、归因对比方法、评价指标
│   ├── variants.py             消融实验使用的训练变体
│   ├── target.py               按目标选择干预时使用的响应模型
│   ├── transunet.py            TransUNet 适配（分析层列表与前向计算）
│   └── models/                 U-Net 与 TransUNet 的网络定义
├── experiments/
│   ├── alignment/              对齐的精确性与可表达性
│   ├── response/               输出变化的扩散、响应原型、正向保真度
│   ├── localization/           原型库、来源定位网络、主要对比实验
│   ├── ablation/               公平性对照、消融、敏感性分析、任意位置干预
│   ├── transunet/              在 TransUNet 上按同一协议的验证
│   ├── analysis/               响应原型与不同网络的对应关系
│   ├── bitflip/                硬件比特翻转的来源定位
│   └── target/                 按目标选择干预的层和位置
├── splits/                     病人划分与固定的切片顺序
├── requirements.txt
└── pyproject.toml
```

## 安装

需要 Python 3.10 及以上版本。

```bash
git clone https://github.com/mingsp/BIA-MedSeg.git
cd BIA-MedSeg
pip install -r requirements.txt
pip install -e .
```

实验在单张 NVIDIA RTX 4090（PyTorch 2.1，关闭 TF32）和 Apple 芯片（PyTorch 2.x，MPS）上运行。程序按 CUDA、MPS、CPU 的顺序自动选择设备，也可以用 `BIA_DEVICE` 指定。`experiments/alignment/exactness.py` 使用 float64 计算，需要在 CUDA 或 CPU 上运行。

## 数据与模型

**数据。** BraTS 2023 成人胶质瘤数据集（BraTS-GLI），包含 T1、T1ce、T2 和 FLAIR 四个 MRI 序列。每个病人取一张肿瘤面积不少于 600 像素的轴位切片，裁剪为 160 x 160。病人按 876、125、250 划分为训练集、验证集和测试集（`splits/brats2023_gli_seed42.json`），`splits/manifest.json` 固定了所有实验使用的切片顺序。数据目录格式如下：

```
data/brats2d/
├── train_Image/<case>.npy     float32，160 x 160 x 4
├── train_Mask/<case>.npy      整数，160 x 160，标签 0-3
├── val_Image/   val_Mask/
└── test_Image/  test_Mask/
```

**被解释的网络。** 分割网络事先训练好并保持冻结：

```
checkpoints/
├── unet_baseline/seed_123/best_model.pth      主要的 U-Net
├── unet_baseline/seed_3407/best_model.pth     更换随机种子训练的 U-Net
├── unet_noskip/seed_123/best_model.pth        无跳接的 U-Net
└── transunet_r50_vit_b16/seed_123/best_model.pth
```

各路径都可以用环境变量修改：

| 变量 | 默认值 |
|---|---|
| `BIA_DATA` | `data/brats2d` |
| `BIA_CKPT` | `checkpoints` |
| `BIA_RESULTS` | `results` |
| `BIA_SPLITS` | `splits/manifest.json` |
| `TU_CKPT_DIR` | `checkpoints/transunet_r50_vit_b16` |
| `BIA_DEVICE` | 自动选择（依次为 `cuda`、`mps`、`cpu`） |

## 复现实验结果

每个脚本把结果写到 `results/<名称>/`。原型库、来源定位网络和逐次查询的结果等公共输出写在 `results/base/`，供后续步骤读取，因此请按以下顺序运行。

**1. 响应模型、来源定位网络与主要对比实验**

```bash
python experiments/alignment/exactness.py
python experiments/response/footprints.py
python experiments/response/summarize_footprints.py
python experiments/localization/train.py
python experiments/localization/evaluate.py | tee results/base/e6_eval.log
python experiments/localization/summarize.py
```

**2. 各项实验**

| 论文 | 脚本 | 输出 |
|---|---|---|
| 4.2.1 节，图 5：对齐的精确性与可表达性 | `experiments/alignment/exactness.py`、`experiments/alignment/expressibility.py` | `base/`、`expressibility/` |
| 4.2.2 节，图 6：响应原型、扩散半径与增益 | `experiments/response/footprints.py`、`summarize_footprints.py` | `base/` |
| 4.2.2 节，图 7：正向保真度 | `experiments/response/fidelity.py` | `fidelity/` |
| 4.2.3 节，表 1、图 8：U-Net 上的来源定位 | `experiments/localization/evaluate.py`、`more_baselines.py`、`timing.py` | `base/`、`more_baselines/`、`sensitivity/` |
| 4.2.4 节，表 2：公平性对照与消融 | `experiments/ablation/build_buffer.py`，然后 `train_eval.py`、`summarize.py`；`input_channels.py`、`summarize_input_channels.py`；`any_position.py`、`any_position_atp.py` | `ablation/`、`input_channels/`、`any_position/` |
| 4.2.4 节，图 10：参数敏感性 | `experiments/ablation/calibration_size.py`、`amplitude.py`（在 `build_buffer.py` 之后运行） | `sensitivity/` |
| 4.2.5 节，表 3：TransUNet | `experiments/transunet/run.py`、`hires_baseline.py` | `transunet/` |
| 4.3.1 节，图 11：跳接的作用与网络对应关系 | `experiments/response/footprints.py`、`experiments/analysis/model_specificity.py`、`experiments/transunet/run.py` | `base/`、`model_specificity/`、`transunet/` |
| 4.3.2 节，表 4：比特翻转的来源定位 | `H1_VERIFY=1 H1_BUDGETS=2,5,10,20,40 python experiments/bitflip/bitflip.py`（在消融实验之后运行）、`CUDNN_OFF=1 python experiments/bitflip/forward_only.py` | `bitflip/`、`bitflip_forward/` |
| 4.3.3 节，图 13：按目标选择干预 | `experiments/target/ring.py`；`dump_val.py`，然后 `fix_missed.py` | `ring/`、`fix_missed/` |

`forward_only.py` 可以用 `H2_SHARD=i/n`（i = 0 ... n-1）分成多个进程运行，中断后从逐病人的断点继续。

## 评价协议

所有主要结果都在 60 个测试病人上报告，这些病人不参与任何原型构建、网络训练和超参数选择。响应原型和来源定位网络由 300 个训练病人和 100 个验证病人构建，另取 24 个验证病人选择超参数。干预作用在统一网格上脑内面积超过一半的 16 像素图像块上；测试使用噪声、均值消融、增益和部分通道噪声 4 类干预，其中后两类在构建原型和训练时从未出现。只有层和图像块都正确时才记为定位正确。置信区间为按病人自助抽样得到的 95% 置信区间。

## 第三方代码

`bia/models/transunet/` 中的文件来自 [TransUNet](https://github.com/Beckschen/TransUNet)（Apache License 2.0），本地改动见 `bia/models/transunet/NOTICE.md`。LRP 对比方法使用 [Zennit](https://github.com/chr5tphr/zennit)。
