# Consolidated Benchmark Report

**Module:** Introduction to Deep Learning

**Phase:** Final Architecture Pairings & Performance Evaluation

**Name:** Harshith Babu Prakash Babu

**Matriculation Number:** 10001198

**Program:** Master's in Artificial Intelligence, THWS

## Executive Summary

Following the forensic audit and reconstruction of the baseline architecture, an automated benchmark orchestrator was developed to map dataset complexity to optimal model capacity. The final pipeline dynamically handles channel alignment and dataset switching, completely avoiding the anti-pattern of rigid, single-model deployments.

## Final Performance Matrix

| Dataset | Selected Architecture | Accuracy | Precision | Recall | Macro F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cells** | `AlexNet` | 95.44% | 0.9559 | 0.9501 | 0.9519 | **PASS** (>90%) |
| **Chest** | `ResNet18` | 90.71% | 0.9353 | 0.8761 | 0.8947 | **PASS** (>87%) |
| **Lesions** | `VGG16` | 77.71% | 0.5531 | 0.4559 | 0.4895 | **PASS** (>67%) |
| **Orgs** | `ResNet18` | 94.19% | 0.9369 | 0.9359 | 0.9354 | **PASS** (>83%) |

## Architectural Selection Rationale

* **Cells (AlexNet):** Microscopic cellular structures lack deep spatial complexity. AlexNet provides sufficient capacity to map these simple features while aggressively combating memorization through its native fully-connected dropout layers.
* **Chest & Orgs (ResNet18):** Macroscopic radiological scans feature highly complex, overlapping anatomical structures. ResNet18 is the optimal pairing, as its residual skip connections provide a direct gradient path that prevents vanishing gradients while extracting deep spatial hierarchies.
* **Lesions (VGG16):** Skin lesions require intense textural extraction and rely heavily on 3-channel RGB color data. VGG16's strict reliance on continuous 3x3 convolutions ("simplicity through depth") provides the necessary capacity to capture fine dermatological variations, successfully clearing the baseline threshold despite inherent dataset class imbalances (reflected in the lower F1 score compared to raw accuracy).

## Green Initiative: Efficiency Verification & Architectural Complexity

A common anti-pattern in deep learning is forcing the deepest available architecture onto every dataset, which unnecessarily inflates inference latency, peak GPU memory consumption, and overall energy footprint. To meet the Executive Board's sustainability targets, this pipeline implements Architectural Downscaling by actively mapping model capacity to the intrinsic complexity of the dataset.

A custom, lightweight `SlimResNet` architecture (utilizing restricted 16-channel convolutions and a shallow 2-block depth with random initialization) was benchmarked against the standard 11-million parameter `ResNet18` (utilizing ImageNet pre-trained weights) on the `chest` dataset.

**Efficiency Results:**
* `ResNet18`: 90.71% Accuracy | 522.52 MB Peak Memory | 276.43s Training Time
* `SlimResNet`: 89.42% Accuracy | 56.81 MB Peak Memory | 58.61s Training Time

**Conclusion:** By deploying the `SlimResNet` architecture, we achieved an 89% memory and 79% runtime reduction. This reflects the combined effect of reduced channel width, shallower depth, and native 64x64 resolution processing. The 1.29% accuracy trade-off (90.71% → 89.42%) quantitatively demonstrates that targeted architectural downscaling preserves robust clinical viability at a fraction of the computational and environmental cost.

## Efficiency Verification Matrix

| Dataset | Model | Mode | Training Time (s) | Inference Latency (s/sample) | Peak GPU Memory (Training) (MB) |
|---|---|---|---|---|---|
| cells | AlexNet | PRETRAINED | 143.42 | 0.000192 | 128.05 |
| chest | ResNet18 | PRETRAINED | 276.43 | 0.000782 | 522.52 |
| chest | SlimResNet | PRETRAINED | 58.61 | 0.000178 | 56.81 |
| lesions | VGG16 | PRETRAINED | 387.42 | 0.000492 | 694.52 |
| orgs | ResNet18 | PRETRAINED | 799.24 | 0.000767 | 522.59 |
| organs | ResNet18 | SCRATCH | 27.32 | 0.000767 | 522.59 |
| organs | ResNet18 | TRANSFER | 26.88 | 0.000719 | 522.22 |

*(Note: The 'Transfer' mode training time accurately reflects Phase 2 target-domain fine-tuning, leveraging MLOps caching for the Phase 1 source-domain pre-training).*

## Data-Scarcity Post-Mortem: The `Organs` Dataset

The integration of the new 500-sample `organs` dataset required navigating extreme data scarcity. The Chief of Medical Testing mandated a minimum test accuracy threshold of 40%. To evaluate the optimal solution, a strictly controlled A/B test was orchestrated.

To ensure scientific validity and remove confounders, both the random initialization and the Knowledge Transfer pipeline were forced to utilize the exact same input resolution (224x224), classification head (Dropout + Linear), learning rate (`0.0001`), and regularization (`1e-3` weight decay).

**The Results:**

Both pipelines successfully cleared the 40% mandate.

* `organs (scratch)`: 65.50% Accuracy | 0.6564 Precision | 0.5992 Recall | 0.5771 Macro F1
* `organs (transfer)`: 62.50% Accuracy | 0.5650 Precision | 0.5559 Recall | 0.5512 Macro F1

**Expert Summary:**

Both approaches successfully cleared the 40% mandate. The random-init Scratch model versus the ImageNet-and-orgs Transfer model showed no significant difference (a 3.0% delta, which equates to exactly 6 test samples out of 200, falls well within single-run variance). Therefore, no statistically significant advantage was demonstrated for either approach. This is itself a highly meaningful finding for extreme data scarcity at this scale, the heavy Transfer pipeline neither clearly helps nor clearly hurts the network's ability to adapt compared to a True Scratch (random initialization) approach.

**Recommendation:** 

For extreme data scarcity, intermediate domain-specific pre-training introduces unnecessary computational overhead unless the source dataset is a mathematically perfect proxy for the target. As more `organs` data is gathered, utilizing generalized ImageNet base weights will likely yield superior clinical reliability without the risk of over-specializing on narrow domain-transfer strategies.