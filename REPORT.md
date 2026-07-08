# Consolidated Benchmark Report: Operation Cyber-Histology

**Course:** Introduction to Deep Learning (SS26) - Final Assignment 

**Authors:** Harshith Babu Prakash Babu, Muhammad Talha Khan

**Matriculation Number:** 10001198, 10013383

**Program:** Master's in Artificial Intelligence, THWS

**Phase:** Final Architecture Pairings & Performance Evaluation

## Executive Summary

After the pipeline was audited and repaired, every model was run through a single automated `benchmark.py` across all four datasets. Each model trained for `15 epochs` at `batch size 64`, using Adam at a learning rate of `0.001` with dropout `0.3`, and a fixed seed of `42` for consistency across runs. Every result below is measured on the held-out **test** split, and we report the **best-validation-epoch checkpoint** rather than the final epoch, since a few datasets dip on the last epoch and evaluating that would understate the model. 

The report covers three parts: 
* The Recovered-Model Benchmark, 
* The Green Initiative Efficiency Work, 
* The Data-Scarcity study on the `organs` dataset.

## Part 1: Consolidated Benchmark Report

The three recovered models (AlexNet, VGG16, ResNet18) were evaluated on every dataset, after the pipeline was fixed. The table reports accuracy, precision, recall, and macro F1 on the test set.

| Selected Architecture | Dataset | Accuracy | Precision | Recall | Macro F1-Score | Status |
|-------|---------|----------|-----------|--------|----------|----------|
| `AlexNet` | **cells** | 95.67% | 0.9592 | 0.9553 | 0.9570 | **PASS** (>90%) |
| `VGG16` | **cells** | 96.54% | 0.9671 | 0.9579 | 0.9622 | **PASS** (>90%) |
| `ResNet18` | **cells** | 96.95% | 0.9718 | 0.9677 | 0.9695 | **PASS** (>90%) |
| `AlexNet` | **chest** | 83.09% | 0.8750 | 0.7615 | 0.7807 | **FAIL** (>87%) |
| `VGG16` | **chest** | 82.39% | 0.8842 | 0.7594 | 0.7783 | **FAIL** (>87%) |
| `ResNet18` | **chest** | 86.62% | 0.8996 | 0.8197 | 0.8396 | **BORDER** (>87%) |
| `AlexNet` | **lesions** | 75.96% | 0.5806 | 0.4971 | 0.5150 | **PASS** (>67%) |
| `VGG16` | **lesions** | 71.77% | 0.3799 | 0.2798 | 0.2888 | **PASS** (>67%) |
| `ResNet18` | **lesions** | 73.47% | 0.5323 | 0.4135 | 0.4224 | **PASS** (>67%) |
| `AlexNet` | **orgs** | 89.47% | 0.8823 | 0.8809 | 0.8795 | **PASS** (>83%) |
| `VGG16` | **orgs** | 89.29% | 0.8762 | 0.8771 | 0.8751 | **PASS** (>83%) |
| `ResNet18` | **orgs** | 92.04% | 0.9164 | 0.9112 | 0.9116 | **PASS** (>83%) |

### Architectural Selection Rationale:

The recommended pairings follow from the observed results:

| Dataset | Benchmark | Recommended model | Best baseline | Result |
| :--- | :--- | :--- | :--- | :--- |
| `cells` | 90% | **ResNet18** | 96.95% | PASS |
| `chest` | 87% | **ResNet18** | 86.62% | BORDER |
| `lesions` | 67% | **AlexNet** | 75.96% | PASS |
| `orgs` | 83% | **ResNet18** | 92.04% | PASS |

**Cells, Lesions, and Orgs** clear their floors comfortably. **Chest is in borderline**, in this run the best model is `ResNet18` reached `86.62%`, which sits just under the 87% target and the reason is it turns out not to be a capacity problem. 

On **Chest**, every Model reaches roughly `97–98%` validation accuracy but drops into the low-to-mid `~80%` on test. That large validation-to-test gap points to a distribution shift between the training/validation data.

Lesions is worth a second look because accuracy and macro F1 disagree sharply. For instance, **AlexNet** hits `75.96%` accuracy but only `0.5150` macro F1. That gap is the signature of class imbalance. 

**ResNet18** has more parameters than **AlexNet** yet lands in the same range. Across repeated runs the **Chest** test score drifts `1-2%` either way, so it clears 87% on some runs and falls just short on others. The honest read is that chest is a genuinely borderline dataset for all of these architectures, and the ceiling comes from the data rather than the model.

## Part 2: Green Initiative Analysis: Efficiency Verification & Architectural Complexity

The goal here was to design a leaner architecture that keeps accuracy close to the baselines while cutting runtime and memory. We built **GreenNet**.

### The Architecture:

`GreenNet` is four small convolutional blocks followed by global average pooling and a single linear layer:

- Four blocks of `Conv2d(3x3, padding 1) → BatchNorm2d → ReLU → MaxPool2d(2)`, with channels growing 16 → 32 → 64 → 128.
- `AdaptiveAvgPool2d((1,1))` - global average pooling, the same idea ResNet18 uses to collapse the feature map.
- Dropout, then `Linear(128, num_classes)`.
- A single normalization line at the start of `forward` that z-scores the input batch, so `GreenNet` normalizes its own inputs without touching the shared data pipeline the baselines use.

Every layer type is one that already appears in the provided models, so nothing exotic was introduced. The design choice that matters most is using global average pooling instead of large fully-connected layers. `AlexNet` and VGG keep most of their weights in those final dense layers, and dropping them is what takes `GreenNet` down to roughly 100K parameters, against millions for the baselines.

### Efficiency Verification Matrix:

| Model | Dataset | Accuracy | Training time (s) | Training memory (MB) | Inference Latency (ms) | Peak GPU Memory (Training) (MB) |
|-------|---------|----------|----------------|----------------|--------------|----------------|
| `GreenNet` | **cells** | 96.67% | 26.25 | 139.4 | 0.0702 | 98.7 |
| `ResNet18` | **cells** | 96.95% | 583.16 | 1539.6 | 0.8662 | 607.5 |
| `GreenNet` | **chest** | 86.61% | 9.32 | 137.4 | 0.0582 | 96.7 |
| `ResNet18` | **chest** | 86.62% | 224.07 | 1535.4 | 0.8837 | 605.5 |
| `GreenNet` | **lesions** | 75.61% | 15.77 | 139.4 | 0.0721 | 98.7 |
| `ResNet18` | **lesions** | 73.47% | 342.03 | 1538.6 | 0.8674 | 607.5 |
| `GreenNet` | **orgs** | 91.30% | 26.52 | 137.4 | 0.0543 | 96.7 |
| `ResNet18` | **orgs** | 92.04% | 655.40 | 1535.5 | 0.8645 | 605.5 |

### Architectural Trade-off Evaluation:

Comparing `GreenNet` against `ResNet18`, which was the strongest baseline overall:

- **Accuracy:** `GreenNet` stays within about `0.75` points on **cells**, **chest**, and **orgs**, and actually beats `ResNet18` on **lesions** (`75.61 vs 73.47`).
- **Training time:** It trains roughly `20–25 times` faster (for example `26.5s vs 655.4s` on **orgs**).
- **Inference latency:** roughly `0.05–0.07 ms` per sample against about `0.87 ms`, a bit over `10 times` faster.
- **Training memory:** about `137 MB` against roughly `1.536 MB`, so around `11 times` lighter.
- **Inference memory:** about `97 MB` against roughly `606 MB`, around `6 times` lighter.

So the trade is heavily in `GreenNet`'s favour. Nearly identical accuracy at a small fraction of the compute and memory. It is also competitive with the other two baselines on every dataset, and on **Chest** it matches `ResNet18` and beats both **AlexNet** and **VGG16**.

### Averaged across all four datasets

`ResNet18` is the sharpest single contrast. Because it is the most expensive baseline, but it is fair to check `GreenNet` against all three original models at once. Averaging each metric over the four datasets:

| Model | Avg accuracy | Avg training time (s) | Avg training mem (MB) | Avg latency (ms) | Avg infererence memory (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `AlexNet` | 86.04% | 31.0 | ~201 | 0.076 | ~124 |
| `VGG16` | 84.99% | 184.6 | ~874 | 0.400 | ~484 |
| `ResNet18` | 87.27% | 451.2 | ~1537 | 0.865 | ~606 |
| `GreenNet` | 87.54% | 19.5 | ~138 | 0.062 | ~98 |

`GreenNet` has the highest average accuracy of the four models (`87.54`, just ahead of `ResNet18`'s `87.27`), and it is also the cheapest model on every axis. It trains faster than even `AlexNet`, the lightest baseline, and uses roughly a tenth of `ResNet18`'s memory. So `GreenNet` is not merely comparable to the original configurations. On average it is both the most accurate and the least expensive. `GreenNet` is more accurate while using a sixth of the training memory and running about `six times` faster at inference.

We also tested a wider variant (`channels 32 → 64 → 128 → 256`, about four times the parameters). It did not meaningfully improve accuracy, for **Chest** it stayed in the same range and **Cells** was slightly worse, which fits the earlier finding that chest is limited by its data, not its model size. Since the wider version cost more for no real gain, we kept the small `GreenNet`, which also fits the "keep it simple" brief better.


## Part 3: Data-Scarcity Post-Mortem

**From scratch** trains `GreenNet` on **organs** directly, starting from random weights. **Transfer** first trains `GreenNet` on the larger `orgs` dataset, saves those weights, then loads them and fine-tunes on `organs`. Since `orgs` and `organs` are the same 11-class grayscale task at different sizes, the features learned on the larger set carry over almost directly.

### Controlled Scarce-Data Benchmark Matrix

| Approach | Test accuracy | Macro F1 | Best val accuracy |
|----------|---------------|----------|-------------------|
| From scratch | 54.45% | 43.48 | ~66 |
| Transfer | 67.70% | 60.02 | 92.00 |

### Quantitative Impact Analysis:

Both approaches clear the `40%` floor, but transfer has more. Clearly **+13.25 points of accuracy and +16.54 points of macro F1.** The Macro F1 jump is the more telling one, because it means transfer helped across all 11 classes, not just the easy ones.

The training curves back this up. From scratch, validation accuracy lurched wildly from epoch to epoch, from the 10s up to the 80s and back down, exactly what you expect when a 50-image for validation slice is being pushed around by a handful of examples. With transfer, the model opened at 72% validation accuracy in the very first epoch, higher than the scratch model reached reliably at some point. Starting from knowledge learned on the larger dataset is doing real work here.

Transfer did not make the scarcity problem disappear. Even with pretrained weights there is still a gap between training accuracy (high 80s) and test accuracy (**67.70%**), and validation gets shaky again in the last few epochs. That residual overfitting is expected with only ~500 images and transfer reduces the problem but cannot remove it.

### Recommendations as more data arrives:

* **Now:** use transfer learning from `orgs`. It is a clear, cheap win over training from scratch and needs no extra data.
- **Short term:** the biggest remaining limit is the tiny validation set, which makes model selection noisy. Simple data augmentation (flips, small rotations) and a larger or cross-validated validation split would make the fine-tuning more stable and the reported numbers more trustworthy.
- **Longer term:** once `organs` grows to a few thousand images, it is worth re-checking whether fine-tuning the whole network still beats training from scratch, and whether freezing only the early layers gives a better balance. With enough data the advantage of transfer shrinks, but until then it is the right default.