# Third-Party Notices

## TransUNet

- Upstream: https://github.com/Beckschen/TransUNet
- Pinned commit: `02ef0010b36eb8328b5e689eadaf613602edf9b8`
- License: Apache License 2.0
- Vendored files: `vit_seg_modeling.py`, `vit_seg_modeling_resnet_skip.py`, and `vit_seg_configs.py`
- Local changes: four-channel BraTS input support, deterministic RGB-kernel expansion, explicit input-size validation, and a minimal `ConfigDict` shim in `vit_seg_configs.py` in place of `ml_collections`.

The Apache License 2.0 text is included in `LICENSE` in this folder.
