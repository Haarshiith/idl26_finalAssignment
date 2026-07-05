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
| 001 | `Code/data.py` | Validation accuracy matches random chance (~19%) and model fails to learn features. | Stray legacy lines at the end of partition assignments overwrote `train_labels` and `val_labels` with an unshuffled tail slice, breaking index parity with the shuffled image tensors. | Excised the redundant, unshuffled index overwrites to preserve the absolute alignment calculated by `torch.randperm`. | 54844ed |
| 002 | `Code/data.py` | `TensorDataset` throws a shape mismatch error during training initialization. | Image arrays were properly sliced to isolate the validation boundary, but corresponding labels were left at full length, causing an instantiation dimension mismatch. | Applied matching list slicing `[:val_start]` to label arrays to enforce dimensional parity. | f357376 |
| 003 | `Code/data.pt` | Pipeline crashes with `FileNotFoundError` immediately upon execution. | The path resolution logic dynamically appended an incorrect `_data` suffix to the filename constructor when searching for target `.pt` files. | Removed the invalid string suffix from the `Path` constructor f-string to match target filenames. | 6738177 |
| 004 | `Code/data.py` | Batch normalization layers crash validation runs with `ValueError: Expected input batch_size > 1`. | The remaining validation and training samples were not perfectly divisible by the batch size, resulting in a dangling single-sample final batch that breaks batch stats tracking. | Configured `drop_last=True` inside the training and validation `DataLoader` instantiations to ensure homogeneous tensor shapes. | 13b7991 |
| 005 | `Code/trainer.py` | Loss fails to converge, flatlines, or explodes into numerical instability across epochs. | Optimization loop lacked an explicit gradient reset between forward iterations, causing historical gradients to accumulate infinitely and corrupt weight updates. | Injected `self.optimizer.zero_grad()` at the initiation of the training batch iteration loop. | b25b403 |
| 006 | `Code/trainer.py` | `RuntimeError: 0D or 1D target tensor expected` terminates training during the backward pass. | Label arrays were loaded as 2D column matrices `(N, 1)` formatted as float tensors, violating the strict 1D integer format required by `nn.CrossEntropyLoss`. | Applied `.squeeze().long()` to target tensors across both training and evaluation iterations to flatten dimensions. | 1cd9058 |
| 007 | `Code/trainer.py` | Structural shape mismatch crashes when feeding varying diagnostic profiles into specific backbones. | Feeding 3-channel RGB image tensors directly into architectures initialized for 1-channel grayscale processing (or vice-versa) broke core weight matrix multiplications. | Implemented dynamic input channel checking logic using `.mean(dim=1)` to collapse RGB data and `.repeat(1, 3, 1, 1)` to scale grayscale inputs. | 5e523a6 |
| 008 | `Code/trainer.py` <br> `Code/transfer.py` | Validation and test metrics show a critical distribution gap, resulting in a severe accuracy collapse. | Asymmetric preprocessing was occurring; training loops processed images using standard ImageNet distribution scaling, while evaluation code parsed raw, unscaled pixel floats. | Standardized preprocessing across the entire environment by implementing matching `T.Normalize` constants across all data tracking points. | f863350 |
| 009 | `Code/models.py` | ResNet18 backbone operates as a static linear transformation, failing to extract complex spatial hierarchies. | The recovered legacy ResNet18 block definitions contained hardcoded identity mappings and completely omitted the required forward graph `return` statement. | Substituted the broken custom structure with a robust, native `torchvision`-backed ResNet18 wrapper containing modular pre-training flags. | f357376 |
| 010 | `Code/models.py` | VGG16 network initialization fails instantly with a structural `NameError`. | The network architecture constructor attempted to instantiate an undefined custom convolutional helper class (`ConvBlock`) missing from the module scope. | Refactored the architecture loop to build sequences using native, inline `nn.Conv2d`, `nn.BatchNorm2d`, and activation layers. | 5422aec |
| 011 | `Code/models.py` | AlexNet model crashes with dimension mismatch errors when input spatial resolution varies from 64x64. | The flattened convolutional feature map output size directly collided with the static input dimension bounds of the primary fully connected layer. | Integrated a responsive `nn.AdaptiveAvgPool2d((4, 4))` layer as a structural bridge to preserve spatial invariance across incoming resolutions. | cd4c5a6 |
| 012 | `Code/train.py` <br> `Code/transfer.py` | Core execution scripts crash with `ModuleNotFoundError` when run from the root directory. | Internal relative code modules incorrectly used an absolute `Code.` package path prefix, breaking path resolution when run natively. | Refactored script import pathways to use direct local sibling references (e.g., `from data import get_loaders`). | d47bb4d |
| 013 | `Code/train.py` | Severe underfitting; deep networks exhibit zero learning capacity and mirror random guessing. | The classifier layer regularizer parameters were sabotaged, hardcoding a dropout rate of `0.99`, which nullified almost all feature mapping signals. | Corrected the dropout hyperparameter baseline to a standard, dynamically handled `0.5` configuration. | f357376 |
| 014 | `run_benchmarks.py` | Rigid pipeline architecture requires exhaustive manual hardcoding to modify simple evaluation states. | Training execution, model selection, and hyperparameter structures were isolated inside single-run scripts, limiting multi-dataset comparison capabilities. | Consolidated all variables into an external `config.json` system and built `run_benchmarks.py` to automate the suite execution. | 789fc54 |