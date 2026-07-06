# Technical Audit Log: Pipeline Reconstruction

This document lists out all critical failures discovered within the BioHealth Diagnostics Global machine learning pipeline. This documents the root causes and applied mathematical/structural corrections to restore system integrity.

## Dataset Profile (Pre-Audit Observation)

Recovered datasets, profiled via `inspect_data.py` before any code modification:

| Dataset | Channels | Image size | Classes | Train samples | Test samples |
|---|---|---|---|---|---|
| cells   | 3 | 64×64 | 8  | 13,671 | 3,421 |
| chest   | 1 | 64×64 | 2  | 5,232  | 624   |
| lesions | 3 | 64×64 | 7  | 8,010  | 2,005 |
| orgs    | 1 | 64×64 | 11 | 15,367 | 8,216 |
| organs  | 1 | 64×64 | 11 | 500    | 200   |

All image tensors are `float32` in range `[0, 1]`. All label tensors are `int64` with shape `(N, 1)`.

## Bug Inventory

| Issue ID | File Name | Manifestation (The Error) | Mathematical / Logical Root Cause | Implemented Structural Correction | Git Commit Hash |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 001 | `Code/data.py` | Validation accuracy matches random chance (~19%) and the model fails to learn features. | Stray legacy lines at the end of partition assignments overwrote `train_labels` and `val_labels` with an unshuffled tail slice, breaking index parity with the shuffled image tensors. | Excised the redundant, unshuffled index overwrites to preserve the alignment produced by `torch.randperm`. | 54844ed |
| 002 | `Code/data.py` | `TensorDataset` throws a shape mismatch error during dataset instantiation. | Image arrays were sliced to isolate the validation boundary, but the corresponding labels were left at full length, causing a dimension mismatch. | Applied matching slicing `[:val_start]` to the label arrays to enforce dimensional parity. | f357376 |
| 003 | `Code/data.py` | Pipeline crashes with `FileNotFoundError` immediately upon execution. | The path-resolution logic appended an incorrect `_data` suffix to the filename constructor when searching for target `.pt` files. | Removed the invalid suffix from the `Path` f-string to match actual target filenames. | 6738177 |
| 004 | `Code/data.py` | Batch-normalization layers crash validation with `ValueError: Expected input batch_size > 1`. | Remaining samples were not divisible by the batch size, producing a dangling single-sample final batch that breaks batch-stats tracking. | Set `drop_last=True` on the training and validation `DataLoader` instances to guarantee homogeneous batch shapes. | 13b7991 |
| 005 | `Code/trainer.py` | Loss fails to converge, flatlines, or explodes into numerical instability across epochs. | The optimization loop lacked an explicit gradient reset between iterations, causing gradients to accumulate and corrupt weight updates. | Injected `self.optimizer.zero_grad()` at the start of the training batch loop. | b25b403 |
| 006 | `Code/trainer.py` | `RuntimeError: 0D or 1D target tensor expected` terminates training during the backward pass. | Labels were loaded as 2D column tensors `(N, 1)` in float format, violating the 1D integer format required by `nn.CrossEntropyLoss`. | Applied `.squeeze().long()` to targets across training and evaluation loops. | 1cd9058 |
| 007 | `Code/trainer.py` | Shape-mismatch crash when feeding varying diagnostic profiles into specific backbones. | Feeding 3-channel RGB tensors into 1-channel-initialized architectures (or vice-versa) broke the first-layer matrix multiplication. | Added dynamic channel alignment: `.mean(dim=1)` to collapse RGB to grayscale and `.repeat(1, 3, 1, 1)` to expand grayscale to RGB, keyed off the model's first-layer shape. | 5e523a6 |
| 008 | `Code/trainer.py` <br> `Code/transfer.py` | Validation/test metrics show a severe distribution gap and accuracy collapse. | Asymmetric preprocessing: training normalized inputs while evaluation passed raw unscaled pixels, creating a train/eval distribution mismatch. | Standardized `T.Normalize` constants across training, validation, and test paths. | f863350 |
| 009 | `Code/models.py` | ResNet18 backbone behaves as a static linear transformation and fails to learn. | The recovered custom ResNet18 hardcoded `Identity` activations and omitted the `return` statement in its forward graph. | Replaced the broken custom structure with a `torchvision`-backed ResNet18 wrapper exposing a configurable pretrained flag and a fresh Dropout+Linear head. | f357376 |
| 010 | `Code/models.py` | VGG16 initialization fails instantly with a `NameError`. | The constructor referenced an undefined helper class (`ConvBlock`) missing from module scope. | Rebuilt the feature loop using native inline `nn.Conv2d`, `nn.BatchNorm2d`, and activation layers. | 5422aec |
| 011 | `Code/models.py` | AlexNet crashes with a matrix-shape error when input resolution differs from 64×64. | The flattened feature-map size collided with the hardcoded input dimension of the first fully connected layer. | Inserted `nn.AdaptiveAvgPool2d((4, 4))` as a dimensional bridge to guarantee spatial invariance across input sizes. | cd4c5a6 |
| 012 | `Code/train.py` <br> `Code/transfer.py` | Scripts crash with `ModuleNotFoundError` when run from the root directory. | Modules used an absolute `Code.` package prefix that broke path resolution under direct execution. | Refactored to direct local sibling imports (e.g., `from data import get_loaders`). | d47bb4d |
| 013 | `Code/train.py` | Severe underfitting; the network mirrors random guessing. | The classifier dropout was sabotaged to a hardcoded rate of `0.99`, nullifying almost all feature signal. | Restored a standard, config-driven dropout baseline of `0.5`. | f357376 |
| 014 | `run_benchmarks.py` | Rigid pipeline requires manual hardcoding to change evaluation states. | Model selection and hyperparameters were baked into single-run scripts, preventing multi-dataset comparison. | Consolidated all variables into an external `config.json` and built `run_benchmarks.py` to automate the suite. | 789fc54 |