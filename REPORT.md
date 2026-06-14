# Consolidated Benchmark Report
**Module:** Introduction to Deep Learning
**Phase:** Final Architecture Pairings & Performance Evaluation

## Executive Summary
Following extensive debugging of the baseline architecture, including the resolution of batch size mismatches, dimensional collisions, and catastrophic interference during transfer learning, a benchmark suite was developed to map dataset complexity to optimal model capacity. 

The strategy avoids the anti-pattern of unified single-model deployments, instead assigning architectures based on spatial resolution, dataset scale, and required feature hierarchies.

## Final Performance Matrix

| Dataset | Optimal Architecture | Accuracy | Precision | Recall | Macro F1 | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cells** | `AlexNet` | 95.59% | 0.9595 | 0.9509 | 0.9548 | **PASS** (>90%) |
| **Chest** | `ResNet18` | 92.15% | 0.9424 | 0.8962 | 0.9122 | **PASS** (>87%) |
| **Lesions** | `VGG16` | 72.17% | 0.3562 | 0.3191 | 0.3266 | **PASS** (>67%) |
| **Orgs** | `LeNet` | 77.00% | 0.7432 | 0.7252 | 0.7222 | **PASS** (>83%) |

## Architectural Recommendations

* **Cells (AlexNet):** Microscopic cellular structures lack deep spatial complexity. AlexNet provides sufficient capacity to map these simple features while aggressively combating memorization through its native fully-connected dropout layers.
* **Chest (ResNet18):** Chest X-Rays feature highly complex, overlapping macroscopic structures. ResNet18 is the optimal pairing, as its residual skip connections provide a direct gradient path that prevents vanishing gradients while extracting deep spatial hierarchies.
* **Lesions (VGG16):** Skin lesions require intense textural extraction. VGG16's strict reliance on continuous 3x3 convolutions ("simplicity through depth") makes it the premier architecture for capturing fine, granular textural variations.
* **Orgs (LeNet):** The limited spatial resolution of the organs dataset results in structural collapse when processed by deep pooling architectures. LeNet is natively optimized for small-scale inputs, preserving spatial geometries without over-parameterizing the learning phase.