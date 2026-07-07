# REPORT — Operation Cyber-Histology

**Authors:** [Your Name] ([Enrollment Number]), Harshith Babu ([Enrollment Number])
**Course:** MAI / IDL SS26 — Final Assignment

All numbers below come from a single benchmark run (`benchmark.py`), 15 epochs per model, batch size 64, Adam at lr 0.001, dropout 0.3, seed fixed at 42. Every model is evaluated on the held-out **test** split, and we report the **best validation epoch** (best-model checkpointing) rather than the final epoch, because a few datasets dip on the last epoch and testing that would undersell the model.

---

## Part 1 — Consolidated Benchmark Report

The three recovered models (AlexNet, VGG16, ResNet18) were run across all four datasets after the pipeline was fixed. The table reports accuracy, macro-averaged precision, recall, and macro F1 on the test set.

| Model | Dataset | Accuracy | Precision | Recall | Macro F1 |
|-------|---------|----------|-----------|--------|----------|
| AlexNet | cells | 95.97 | 95.92 | 95.53 | 95.70 |
| VGG16 | cells | 96.43 | 96.71 | 95.79 | 96.22 |
| ResNet18 | cells | 96.93 | 97.18 | 96.77 | 96.95 |
| AlexNet | chest | 82.05 | 88.50 | 76.15 | 78.07 |
| VGG16 | chest | 81.89 | 88.42 | 75.94 | 77.83 |
| ResNet18 | chest | 86.22 | 89.96 | 81.97 | 83.96 |
| AlexNet | lesions | 75.96 | 56.06 | 49.71 | 51.50 |
| VGG16 | lesions | 71.77 | 37.99 | 27.98 | 28.88 |
| ResNet18 | lesions | 73.17 | 53.23 | 41.35 | 42.24 |
| AlexNet | orgs | 89.47 | 88.23 | 88.09 | 87.95 |
| VGG16 | orgs | 89.09 | 87.62 | 87.71 | 87.51 |
| ResNet18 | orgs | 92.04 | 91.64 | 91.12 | 91.16 |

### Floor check and recommended pairing

The target floors were cells 90%, chest 87%, lesions 67%, orgs 83%.

| Dataset | Floor | Best baseline | Result | Recommended model |
|---------|-------|---------------|--------|-------------------|
| cells | 90% | ResNet18 (96.93) | clears | ResNet18 for peak accuracy; AlexNet if speed matters |
| chest | 87% | ResNet18 (86.22) | borderline — see note | ResNet18 |
| lesions | 67% | AlexNet (75.96) | clears | AlexNet |
| orgs | 83% | ResNet18 (92.04) | clears | ResNet18 |

cells, lesions, and orgs clear their floors comfortably. **chest is the awkward one.** In this run the best model (ResNet18) reached 86.22%, which sits just under the 87% target. We looked into why, and it is not a simple capacity problem. On chest every deep model reaches ~97–98% validation accuracy but drops to the low-to-mid 80s on test — a large validation-to-test gap that points to a distribution shift between the training/validation data and the test data. Making the model bigger does not close this gap (ResNet18 has far more parameters than AlexNet yet lands in the same range). Across repeated runs the chest test score moves by one to two points in either direction, so it does cross 87% on some runs and falls just short on others. Our honest read is that chest is a genuinely borderline dataset for all of these architectures, and the ceiling comes from the data, not the model.

### A note on lesions

lesions is worth a second look because accuracy and macro F1 disagree sharply. AlexNet, for example, hits 75.96% accuracy but only 51.50 macro F1. That gap is the signature of class imbalance: with seven classes and some of them rare, a model can score well on the common classes (which drives accuracy up) while doing poorly on the rare ones (which drags macro F1 down, since macro F1 weights every class equally). The floor is defined on accuracy, which is cleared, but the low macro F1 is the more honest description of per-class behaviour, and a real deployment would need to address the rare classes directly.

---

## Part 2 — Green Initiative Analysis

The task here was to design a leaner architecture that keeps accuracy close to the baselines while cutting runtime and memory. We built **GreenNet**.

### The architecture

GreenNet is four small convolutional blocks followed by global average pooling and a single linear layer:

- Four blocks of `Conv2d(3x3, padding 1) → BatchNorm2d → ReLU → MaxPool2d(2)`, with channels growing 16 → 32 → 64 → 128.
- `AdaptiveAvgPool2d((1,1))` — global average pooling, the same idea ResNet18 uses to collapse the feature map.
- Dropout, then `Linear(128, num_classes)`.
- A single normalization line at the start of `forward` that z-scores the input batch, so GreenNet normalizes its own inputs without touching the shared data pipeline the baselines use.

Every layer type is one that already appears in the provided models, so nothing exotic was introduced. The design choice that matters most is using global average pooling instead of large fully-connected layers. AlexNet and VGG keep most of their weights in those final dense layers, and dropping them is what takes GreenNet down to roughly 100K parameters, versus millions for the baselines.

### Cost and accuracy, side by side

| Model | Dataset | Accuracy | Train time (s) | Train mem (MB) | Latency (ms) | Infer mem (MB) |
|-------|---------|----------|----------------|----------------|--------------|----------------|
| GreenNet | cells | 96.67 | 26.25 | 139.4 | 0.0702 | 98.7 |
| ResNet18 | cells | 96.93 | 583.16 | 1539.6 | 0.8662 | 607.5 |
| GreenNet | chest | 86.06 | 9.32 | 137.4 | 0.0582 | 96.7 |
| ResNet18 | chest | 86.22 | 224.07 | 1535.4 | 0.8837 | 605.5 |
| GreenNet | lesions | 75.61 | 15.77 | 139.4 | 0.0721 | 98.7 |
| ResNet18 | lesions | 73.17 | 342.03 | 1538.6 | 0.8674 | 607.5 |
| GreenNet | orgs | 91.30 | 26.52 | 137.4 | 0.0543 | 96.7 |
| ResNet18 | orgs | 92.04 | 655.40 | 1535.5 | 0.8645 | 605.5 |

### What the numbers say

Comparing GreenNet against ResNet18, which was the strongest baseline overall:

- **Accuracy:** GreenNet stays within about 0.75 points on cells, chest, and orgs, and actually beats ResNet18 on lesions (75.61 vs 73.17).
- **Training time:** GreenNet trains roughly 20–25 times faster (for example 26.5s vs 655.4s on orgs).
- **Training memory:** about 137 MB against roughly 1,536 MB, so around 11 times lighter.
- **Inference latency:** roughly 0.05–0.07 ms per sample against about 0.87 ms, a bit over 10 times faster.
- **Inference memory:** about 97 MB against roughly 606 MB, around 6 times lighter.

So the trade is heavily in GreenNet's favour: near-identical accuracy at a small fraction of the compute and memory. On its own GreenNet is also competitive with the other two baselines on every dataset, and on chest it matches ResNet18 and beats both AlexNet and VGG16.

### Averaged across all four datasets

ResNet18 is the sharpest single contrast because it is the most expensive baseline, but it is fair to check GreenNet against all three original models at once. Averaging each metric over the four datasets gives:

| Model | Avg accuracy | Avg train time (s) | Avg train mem (MB) | Avg latency (ms) | Avg infer mem (MB) |
|-------|--------------|--------------------|--------------------|------------------|--------------------|
| AlexNet | 85.86 | 31.0 | ~201 | 0.076 | ~124 |
| VGG16 | 84.80 | 184.7 | ~874 | 0.400 | ~484 |
| ResNet18 | 87.09 | 451.2 | ~1537 | 0.865 | ~606 |
| GreenNet | 87.41 | 19.5 | ~138 | 0.062 | ~98 |

This is the clearest way to state the result. GreenNet has the highest average accuracy of the four (87.41, just ahead of ResNet18's 87.09), and it is also the cheapest model on every axis — it trains faster than even AlexNet, the lightest baseline, and uses roughly a tenth of ResNet18's memory. So GreenNet is not merely comparable to the original configurations; on average it is both the most accurate and the least expensive. The gap over VGG16 is especially one-sided: GreenNet is more accurate while using roughly a sixth of the training memory and running about six times faster at inference.

We also tested a wider variant (channels 32 → 64 → 128 → 256, about four times the parameters). It did not meaningfully improve accuracy — chest stayed in the same range and cells was slightly worse — which lined up with the earlier finding that chest is limited by its data rather than by model size. Since the wider version cost more for no real gain, we kept the small GreenNet, which is also the better fit for the "keep it simple" brief.

One honest caveat on the inference memory numbers: they are measured after freeing the optimizer and gradient state, so they reflect the forward pass only. This is why inference memory is noticeably lower than training memory for every model.

---

## Part 3 — Data-Scarcity Post-Mortem

The organs dataset is small (around 500 training images), and the task was to get useful accuracy out of it, with a floor of 40% on the test set. We compared two approaches using the same GreenNet architecture.

**From scratch** means training GreenNet on organs directly, starting from random weights.

**Transfer** means first training GreenNet on the larger orgs dataset (`pretrain.py` saves those weights), then loading them and fine-tuning on organs (`transfer.py`). orgs and organs are the same 11-class grayscale task, just different sizes, so the features learned on the larger set carry over almost directly.

| Approach | Test accuracy | Macro F1 | Best val accuracy |
|----------|---------------|----------|-------------------|
| From scratch | 54.00 | 43.85 | ~66 |
| Transfer | 67.00 | 59.54 | 92.00 |

Both clear the 40% floor, but transfer wins clearly: **+13 points of accuracy and +15.7 points of macro F1.** The macro F1 jump is the more telling one, because it means transfer helped across all eleven classes, not just the easy ones.

The training curves back this up. Training from scratch, validation accuracy lurched around wildly from epoch to epoch (from the teens up to the 80s and back down), which is what you expect when a 50-image validation slice is being pushed around by a handful of examples. With transfer, the model started at 72% validation accuracy in the very first epoch — higher than the scratch model reached reliably at any point — and climbed more steadily to a best of 92%. Starting from knowledge learned on the larger dataset is doing real work here.

Transfer did not make the scarcity problem disappear. Even with pretrained weights there is still a gap between training accuracy (high 80s) and test accuracy (67%), and validation gets shaky again in the last few epochs. That residual overfitting is expected with only ~500 images; transfer reduces the problem but cannot remove it.

### Recommendations as more data arrives

- **Now:** use transfer learning from orgs. It is a clear, cheap win over training from scratch and needs no extra data.
- **Short term:** the biggest remaining limit is the tiny validation set, which makes model selection noisy. Simple data augmentation (flips, small rotations) and a larger or cross-validated validation split would make the fine-tuning more stable and the reported numbers more trustworthy.
- **Longer term:** once organs grows to a few thousand images, it is worth re-checking whether fine-tuning the whole network still beats training from scratch, and whether freezing only the early layers gives a better balance. With enough data the advantage of transfer shrinks, but until then it is the right default.
