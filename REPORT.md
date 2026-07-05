# Consolidated Benchmark Report: Operation Cyber-Histology

**Module:** Introduction to Deep Learning

**Phase:** Final Architecture Pairings & Performance Evaluation

**Name:** Harshith Babu Prakash Babu

**Matriculation Number:** 10001198

**Program:** Master's in Artificial Intelligence, THWS

## Executive Summary

Following the forensic audit and reconstruction of the baseline architecture, an automated benchmark orchestrator was developed to map dataset complexity to optimal model capacity. The final pipeline dynamically handles channel alignment and dataset switching, completely avoiding the anti-pattern of rigid, single-model deployments.

## Final Performance Matrix

The table below details the performance metrics captured across all dataset and model configurations under strictly seeded execution conditions.

| Dataset | Selected Architecture | Mode | Accuracy | Precision | Recall | Macro F1-score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cells** | `AlexNet` | PRETRAINED | 92.87% | 0.9170 | 0.9255 | 0.9197 | **PASS** (>90%) |
| **Chest** | `ResNet18` | PRETRAINED | 90.38% | 0.9275 | 0.8744 | 0.8916 | **PASS** (>87%) |
| **Lesions** | `VGG16` | PRETRAINED | 80.40% | 0.6498 | 0.5660 | 0.5936 | **PASS** (>67%) |
| **Orgs** | `ResNet18` | PRETRAINED | 94.86% | 0.9413 | 0.9409 | 0.9409 | **PASS** (>83%) |

## Architectural Selection Rationale

Based on the observed benchmark evaluations, the architectural allocations are justified as follows:

* **Cells (AlexNet):** Microscopic cellular structures lack deep spatial complexity. AlexNet provides sufficient capacity to map these simple features while aggressively combating memorization through its native fully-connected dropout layers.
* **Chest & Orgs (ResNet18):** Macroscopic radiological scans feature highly complex, overlapping anatomical structures. ResNet18 is the optimal pairing, as its residual skip connections provide a direct gradient path that prevents vanishing gradients while extracting deep spatial hierarchies.
* **Lesions (VGG16):** Skin lesions require intense textural extraction and rely heavily on 3-channel RGB color data. VGG16's strict reliance on continuous $3 \times 3$ convolutions ("simplicity through depth") provides the necessary capacity to capture fine dermatological variations, successfully clearing the baseline threshold despite inherent dataset class imbalances (reflected in the lower F1 score compared to raw accuracy).

## Green Initiative: Efficiency Verification & Architectural Complexity

### Efficiency Verification Matrix:

| Dataset | Model | Mode | Training Time (s) | Inference Latency (s/sample) | Peak GPU Memory (Training) (MB) |
|---|---|---|---|---|---|
| chest | ResNet18 | PRETRAINED | 355.97 | 0.001060 | 522.52 |
| chest | SlimResNet | PRETRAINED | 54.80 | 0.000156 | 56.81 |

### Architectural Trade-off Evaluation:

A common anti-pattern in deep learning is forcing the deepest available architecture onto every dataset, which unnecessarily inflates inference latency, peak GPU memory consumption, and overall energy footprint. To meet the Executive Board's sustainability targets, this pipeline implements Architectural Downscaling by actively mapping model capacity to the intrinsic complexity of the dataset.

A custom, lightweight `SlimResNet` architecture (utilizing restricted 16-channel convolutions and a shallow 2-block depth with random initialization) was benchmarked against the standard 11-million parameter `ResNet18` (utilizing ImageNet pre-trained weights) on the `chest` dataset.

### Conclusion:

By deploying the streamlined `SlimResNet` architecture, we achieved an **89.13% memory** and **84.61% runtime reduction**. This reflects the combined effect of reduced channel width, shallower depth, and native 64x64 resolution processing. The 2.88% accuracy trade-off (90.38% → 87.50%) quantitatively demonstrates that targeted architectural downscaling preserves robust clinical viability at a fraction of the computational and environmental cost.

## Data-Scarcity Post-Mortem: The `Organs` Dataset

### Controlled Scarce-Data Benchmark Matrix
| Experimental Arm | Initialization State | Pre-training Phase | Accuracy | Macro F1 |
|---|---|---|---|---|
| **Arm A (True Scratch)** | Random Initialization | None | 65.50% | 0.6057 |
| **Arm B (Generalized Base)** | ImageNet Weights Only | None | 70.00% | 0.6192 |
| **Arm C (Domain Transfer)** | ImageNet Weights | 470.73s on `orgs` | 61.50% | 0.5034 |

The integration of the new 500-sample `organs` dataset required navigating extreme data scarcity. The Chief of Medical Testing mandated a minimum test accuracy threshold of 40%. To evaluate the optimal solution, a strictly controlled A/B test was orchestrated.

To ensure scientific validity and remove confounders, both the random initialization and the Knowledge Transfer pipeline were forced to utilize the exact same input resolution (224x224), classification head (Dropout + Linear), learning rate (`0.0001`), and regularization (`1e-3` weight decay).

### Quantitative Impact Analysis:

Pipeline execution is governed by a global random seed to ensure consistent initialization and batch order across arms. For Phase 2 of Arm C, the full network was fine-tuned with a re-initialized classification head to allow feature adaptation while avoiding source-domain index misalignment.

**Expert Summary:**

All three experimental configurations successfully cleared the 40% diagnostic mandate. However, no statistically significant advantage was detected across the arms given the evaluation bounds ($n=200$ test samples), meaning no single approach demonstrated a definitive edge at this data scale.

**Recommendation:** 

Arm C introduces substantial operational complexity, requiring a hidden, one-time Phase 1 pre-training duration of **470.73 seconds** on the `orgs` base. Because the performance deltas sit within the single-run noise band, heavy multi-stage transfer pre-training frameworks are currently not justified. Until a much larger, semantically certain proxy dataset is curated, initializing directly from generalized ImageNet configurations (Arm B) preserves computational resources while maintaining equivalent clinical triage performance.