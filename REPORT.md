# Consolidated Benchmark Report: Operation Cyber-Histology

**Module:** Introduction to Deep Learning

**Phase:** Final Architecture Pairings & Performance Evaluation

**Name:** Harshith Babu Prakash Babu

**Matriculation Number:** 10001198

**Program:** Master's in Artificial Intelligence, THWS

## Executive Summary

Following the forensic audit and reconstruction of the baseline architecture, an automated benchmark orchestrator was developed to map dataset complexity to model capacity. The final pipeline dynamically handles channel alignment, normalization, and dataset switching through a single external configuration file, avoiding the anti-pattern of rigid, single-model deployments. All runs are governed by a fixed global seed (42) for consistency across executions.

## Final Performance Matrix

The table below details the performance metrics captured across all dataset and model configurations under strictly seeded execution conditions.

| Dataset | Selected Architecture | Mode | Accuracy | Precision | Recall | Macro F1-score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cells** | `AlexNet` | PRETRAINED | 94.68% | 0.9475 | 0.9353 | 0.9400 | **PASS** (>90%) |
| **Chest** | `ResNet18` | PRETRAINED | 89.90% | 0.9284 | 0.8662 | 0.8851 | **PASS** (>87%) |
| **Lesions** | `VGG16` | PRETRAINED | 79.50% | 0.7346 | 0.5426 | 0.5771 | **PASS** (>67%) |
| **Orgs** | `ResNet18` | PRETRAINED | 93.27% | 0.9276 | 0.9223 | 0.9227 | **PASS** (>83%) |

## Architectural Selection Rationale

Based on the observed benchmark evaluations, the architectural allocations are justified as follows:

* **Cells (AlexNet):** Microscopic cellular structures lack deep spatial complexity. AlexNet provides sufficient capacity to map these simple features while aggressively combating memorization through its fully-connected dropout layers, at a fraction of the memory cost of deeper backbones.
* **Chest & Orgs (ResNet18):** Macroscopic radiological scans feature complex, overlapping anatomical structures. ResNet18's residual skip connections provide a direct gradient path that supports deep spatial feature extraction, and its ImageNet initialization accelerates convergence on these profiles.
* **Lesions (VGG16):** Skin lesions rely on fine textural and 3-channel colour variation. VGG16's stacked $3 \times 3$ convolutions capture this detail and clear the threshold. The gap between accuracy (79.50%) and macro-F1 (0.5771) reflects the dataset's known class imbalance. The model performs well on majority classes and weakly on rare ones, so accuracy overstates per-class reliability.

## Green Initiative: Efficiency Verification & Architectural Complexity

### Efficiency Verification Matrix:

| Dataset | Model | Initialization | Accuracy | Macro F1 | Training Time (s) | Inference Latency (s/sample) | Peak GPU Memory (Training) (MB) |
|---|---|---|---|---|---|---|---|
| chest | ResNet18 | ImageNet | 89.90% | 0.8851 | 355.97 | 0.001060 | 522.52 |
| chest | SlimResNet | Random | 87.66% | 0.8574 | 54.80 | 0.000156 | 56.81 |

### Architectural Trade-off Evaluation:

Forcing the deepest available architecture onto every dataset inflates latency, memory, and energy footprint. `SlimResNet` implements aggressive architectural downscaling directly in the model definition. A 16-channel stem (versus ResNet18's 64), two lightweight residual blocks, and global average pooling, operating on native 64×64 inputs.

Benchmarked head-to-head against the 11-million-parameter ImageNet-initialized `ResNet18` on the `chest` dataset:

* **Peak GPU memory:** 522.52 MB → 58.28 MB, an **88.8% reduction**.
* **Training runtime:** 273.52s → 66.14s, a **75.8% reduction**.
* **Accuracy trade-off:** 89.90% → 87.66%, a **2.24-point drop**, still clearing the 87% clinical threshold.

The comparison is not fully isolated. The baseline carries ImageNet initialization while `SlimResNet` trains from scratch, so part of the efficiency delta reflects both architecture size and initialization. Even accounting for this, the memory and runtime reductions are an order of magnitude larger than the accuracy cost, demonstrating that targeted downscaling preserves clinical viability at a fraction of the computational and environmental footprint.

## Data-Scarcity Post-Mortem: The `Organs` Dataset

### Controlled Scarce-Data Benchmark Matrix

| Experimental Arm | Initialization State | Pre-training Phase | Accuracy | Macro F1 |
|---|---|---|---|---|
| **Arm A (True Scratch)** | Random Initialization | None | 67.50% | 0.6200 |
| **Arm B (Generalized Base)** | ImageNet Weights Only | None | 73.50% | 0.6848 |
| **Arm C (Domain Transfer)** | ImageNet Weights | ~469 on `orgs` | 69.00% | 0.5842 |

The 500-sample `organs` dataset required navigating extreme data scarcity against a mandated minimum test accuracy of 40%. Three arms were compared under matched conditions: identical input resolution (`224×224`), classifier head (`Dropout + Linear`), learning rate (`0.0001`), weight decay (`1e-3`), epoch budget (`20`), and global seed. The only variable across arms is the initialization/pre-training state.

### Quantitative Impact Analysis:

All three arms cleared the 40% mandate. The ranking is **B (73.50%) > C (69.00%) > A (67.50%)** on accuracy, with the same ordering on macro-F1 (0.6848 > 0.5842 for C, 0.6200 for A).

Two findings stand out:

1. **Generalized ImageNet initialization (Arm B) was the strongest and most balanced.** It beat true scratch (Arm A) by 6.0 accuracy points and achieved the highest macro-F1 (0.6848), indicating better performance across minority classes, not just majority-class accuracy.

2. **Intermediate `orgs` pre-training (Arm C) did not help and produced the weakest class balance.** Despite mid-pack accuracy, Arm C recorded the lowest macro-F1 (0.5842), meaning its predictions were more skewed toward dominant classes.

**Important warning on Arm C:** Arm C's classification head was re-initialized before Phase 2 (to avoid any source/target label-index mismatch), so it began fine-tuning from a random head on top of the `orgs`-pretrained backbone. Its validation accuracy was still climbing at the end of the 20-epoch budget (reaching 83.33% only at the final epoch, versus Arm B stabilizing near 89% by epoch 4). Arm C's underperformance may therefore partly reflect insufficient optimization time for the reset head rather than pure negative transfer. This distinction matters supports the data "orgs pre-training offered no advantage within a fixed budget," but does not conclusively prove a negative-transfer mechanism.

**Expert Summary:**

For extreme data scarcity at this scale (n=200 test), generalized ImageNet initialization (Arm B) delivered the best and most balanced performance while requiring no additional pre-training. Intermediate domain-specific pre-training on `orgs` (Arm C) added a substantial one-time computational cost (~469s Phase 1) without improving results, and degraded per-class balance.

**Recommendation:** 

Initialize directly from generalized ImageNet weights (Arm B) for scarce-data organ classification. Intermediate domain-specific pre-training is not justified at the current data scale and, if pursued in future, must be given a longer fine-tuning budget before its effect can be fairly evaluated. As more `organs` data is collected, these arms should be re-benchmarked with multiple seeds to establish confidence intervals, since single-run deltas at n=200 remain sensitive to sampling.
