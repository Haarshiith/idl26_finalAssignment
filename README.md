# MAI - IDL 2026 - Final Project Assignment

# BioHealth Histology Pipeline Reconstitution
**Course:** Introduction to Deep Learning (SS26) - Final Assignment  
**Author:** Harshith Babu Prakash Babu (Matricultion no: 10001198)

## 📋 Project Overview
This repository contains the successfully audited, repaired, and optimized deep learning pipeline for BioHealth Diagnostics Global. Following a critical system wipe, the pipeline has been reconstructed from legacy draft caches to train and evaluate core model registries (`AlexNet`, `VGG16`, `ResNet18`) across standardized diagnostic image profiles (`cells`, `chest`, `lesions`, and `orgs`).

The architecture features dynamic channel alignment, tailored data normalization, strict regularization to combat overfitting, and a pre-training/transfer learning framework that satisfies all corporate clinical triage constraints.

---

## 📁 Repository Structure
```text
├── Code/
│   ├── data.py             # Volatile tensor data loading & train/val splitting
│   ├── models.py           # Deep learning model architectures (LeNet, VGG, ResNet)
│   ├── fit.py              # Training loop routines, evaluation, and latency tracking
│   ├── train.py            # Standard training and metric execution entry point
│   └── transfer.py         # Domain-Aligned Knowledge Transfer framework
├── data/                   # Target diagnostic datasets (.pt files) [Git ignored]
├── config.json             # Execution configuration registry (automatically generated)
├── run_benchmarks.py       # Automated multi-dataset benchmark execution suite
├── AUDIT_LOG.md            # Technical inventory of code defects and engineering fixes
├── REPORT.md               # Final consolidated benchmark performance report
└── README.md               # Project documentation (This file)
```

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup
Ensure you have Python 3.10+ installed. It is highly recommended to use a virtual environment:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate #for Mac
venv\Scripts\activate #for Windows
```

## Install required packages

```
pip install torch torchvision scikit-learn numpy
```

### 2. Dataset Placement
Download the pristine emergency back-up datasets and place them within a root directory named data/:

```bash
mkdir data
# Place cells.pt, chest.pt, lesions.pt, and orgs.pt into the data/ directory
```

### 3. Running the Complete Benchmark Suite
To replicate all final production runs across all datasets using their optimal architecture pairings under isolated GPU memory conditions, execute the automated orchestration runner:

```bash
python run_benchmarks.py
```

## 🛠️ Model Persistence & Performance Highlights

Automated Weight Caching: To minimize redundant compute, the pipeline automatically caches Phase 1 chest pre-training weights (chest_pretrained_base.pth). Subsequent execution runs bypass re-training, cutting execution runtime by over 85%.

Dynamic Channel Alignment: Data loaders dynamically adapt 1-channel (grayscale) and 3-channel (RGB) images to meet backbone architectural requirements seamlessly.

Verified Performance: All pipelines meet or surpass their target evaluation thresholds. For detailed performance matrices and engineering deep-dives, see REPORT.md and AUDIT_LOG.md.