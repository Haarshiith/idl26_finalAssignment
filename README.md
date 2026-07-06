# MAI - IDL 2026 - Final Project Assignment

# BioHealth Histology Pipeline Reconstitution

**Course:** Introduction to Deep Learning (SS26) - Final Assignment 

**Author:** Harshith Babu Prakash Babu

**Matriculation Number:** 10001198

**Program:** Master's in Artificial Intelligence, THWS

## Project Overview
This repository contains the successfully audited, repaired, and optimized deep learning pipeline for BioHealth Diagnostics Global. Following a critical system wipe, the pipeline has been reconstructed from legacy draft caches to train and evaluate core model registries (`AlexNet`, `VGG16`, `ResNet18`) across standardized diagnostic image profiles (`cells`, `chest`, `lesions`, `orgs` and `organs`).

The architecture features dynamic input channel alignment, tailored data normalization distributions, strict regularization to combat overfitting, and a modular configuration system.

---

## Repository Structure
```text
├── Code/
│   ├── data.py             # Volatile tensor data loading & train/val splitting
│   ├── inspect_data.py     # Diagnostic data checking tool for shape and label alignment
│   ├── models.py           # Deep learning model (AlexNet, VGG, ResNet, SlimResNet)
│   ├── train.py            # Standard training and metric execution entry point
│   ├── trainer.py          # Training, evaluation, and latency tracking
│   └── transfer.py         # Knowledge Transfer framework
├── data/                   # Target diagnostic datasets (.pt files) [Git ignored]
├── config.json             # Execution configuration registry (automatically generated)
├── run_benchmarks.py       # Automated multi-dataset benchmark execution suite
├── AUDIT_LOG.md            # Technical inventory of code defects and engineering fixes
├── EFFICIENCY_MATRIX.md    # Profiled training runtimes, latencies, and peak GPU footprints
├── REPORT.md               # Final consolidated benchmark performance report
└── README.md               # Project documentation (This file)
```

## Getting Started

### 1. Prerequisites & Environment Setup
Ensure Python 3.10+ is installed locally. Build a clean virtual environment and load required packages:

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
# Place cells.pt, chest.pt, lesions.pt, orgs.pt, and organs.pt into the data/ directory
```

### 3. Running the Complete Benchmark Suite
To replicate all final production runs across all datasets using their optimal architecture pairings under isolated GPU memory conditions, execute the automated orchestration runner:

```bash
python run_benchmarks.py
```

## Model Persistence & Performance Highlights

**Automated Weight Caching & Checkpointing:** The pipeline uses dynamic checkpointing (`best_model.pth`) to recover the highest-performing epoch prior to test-set evaluation, preventing late-epoch overfitting from degrading the reported metrics. To minimize redundant compute during the data-scarcity experiment, Phase 1 `orgs` pre-training weights are cached to `orgs_pretrained_base.pth`; subsequent runs bypass Phase 1 re-training. The ~21s fine-tuning time reflects a cache hit — end-to-end transfer execution requires an initial ~469s Phase 1 pre-training pass.

**Dynamic Channel Alignment:** Data loaders dynamically adapt 1-channel (grayscale) and 3-channel (RGB) images to each backbone's expected input without hardcoded dimension limits.

**Reproducibility:** All runs are governed by a fixed global seed (42) for consistency across executions.

**Verified Performance:** All pipelines meet or surpass their target test-accuracy thresholds. The custom `SlimResNet` architecture demonstrates the Green Initiative by reducing peak training memory by **88.8%** (522.52 MB → 58.28 MB) and training runtime by **75.8%** on the `chest` benchmark, for a 2.24-point accuracy trade-off. See `REPORT.md` and `EFFICIENCY_MATRIX.md` for full matrices.