# MAI - IDL 2026 - Final Project Assignment

# BioHealth Histology Pipeline Reconstitution

**Course:** Introduction to Deep Learning (SS26) - Final Assignment 

**Authors:** Harshith Babu Prakash Babu, Muhammad Talha Khan

**Matriculation Number:** 10001198, 10013383

**Program:** Master's in Artificial Intelligence, THWS

## Project Overview
This repository contains the successfully audited, repaired, and optimized deep learning pipeline for BioHealth Diagnostics Global. Following a critical system wipe, the pipeline has been reconstructed from legacy draft caches to train and evaluate core model registries (`AlexNet`, `VGG16`, `ResNet18`) across standardized diagnostic image profiles (`cells`, `chest`, `lesions`, `orgs` and `organs`).

The architecture features dynamic input channel alignment, tailored data normalization distributions, strict regularization to combat overfitting, and a modular configuration system.

## Repository Structure
```text
├── Code/
│   ├── data/                 # Target diagnostic datasets (.pt files)
│   ├── benchmark.py          # Automated multi-dataset benchmark execution suite
│   ├── config.json           # Execution configuration registry
│   ├── data.py               # Volatile tensor data loading & train/val splitting
│   ├── inspect_data.py       # Diagnostic data checking tool
│   ├── models.py             # Deep learning models (AlexNet, VGG, ResNet, GreenNet)
│   ├── pretrain.py           # Pre-training execution script
│   ├── train.py              # Standard training and metric execution entry point
│   ├── trainer.py            # Training, evaluation, and latency tracking engine
│   └── transfer.py           # Knowledge Transfer framework
├── .gitignore                # Git exclusion rules
├── AUDIT_LOG.md              # Technical inventory of code defects and engineering fixes
├── README.md                 # Project documentation (This file)
├── REPORT.md                 # Final consolidated benchmark performance report
├── requirements.txt          # Python environment dependencies
└── results.csv               # Exported benchmark metrics output
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

### Install dependencies
```
pip install -r requirements.txt
```

### 2. Dataset Placement
Download the pristine emergency back-up datasets and place them within the nested data directory `data/`:

```bash
mkdir data
# Place cells.pt, chest.pt, lesions.pt, orgs.pt, and organs.pt into the data/ directory
```

### 3. Running the Complete Benchmark Suite
To replicate all final production runs across all datasets using their optimal architecture pairings under isolated GPU memory conditions, execute the automated orchestration runner:

```bash
python benchmarks.py
```

## Model Persistence & Performance Highlights

**Automated Weight Caching & Checkpointing:** The pipeline uses dynamic checkpointing (`best_model.pt`) to recover the highest-performing epoch prior to test-set evaluation, preventing late-epoch overfitting from degrading the reported metrics. To minimize redundant compute during the data-scarcity experiment, source-domain pre-training weights are cached to `pretrained_orgs.pt`. Subsequent runs automatically load this cache to bypass redundant Phase 1 computation.

**Dynamic Channel Alignment:** Data loaders dynamically adapt 1-channel (grayscale) and 3-channel (RGB) images to each backbone's expected input without hardcoded dimension limits.

**Reproducibility:** All runs are governed by a fixed global seed (42) for consistency across executions.

All optimized model pipelines successfully meet their clinical accuracy thresholds (with `chest` serving as a known data-limited borderline). The custom `GreenNet` architecture fully satisfies the Green Initiative on the `chest` benchmark, it reduces peak GPU training memory by `~84% (605.5 MB → 96.7 MB)` and slashes training runtime by `~95.8% (224.0s → 9.3s)` while maintaining a nearly identical accuracy to the massive ResNet18 baseline (only a 0.01% difference).

(For the complete empirical analysis, efficiency matrix, and scarcity post-mortem, please refer to `REPORT.md`).