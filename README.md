# MAI - IDL 2026 - Final Project Assignment

# BioHealth Histology Pipeline Reconstitution

**Course:** Introduction to Deep Learning (SS26) - Final Assignment 

**Authors:** Harshith Babu Prakash Babu, Muhammad Talha Khan

**Matriculation Number:** 10001198, 10013383

**Program:** Master's in Artificial Intelligence, THWS

## Branch Purpose

This `development` branch contains the **Part 3: Data-Scarcity / Knowledge Transfer** work, committed separately from the reconstruction and Green Initiative deliverables on `main`, as required by the assignment brief.

It builds directly on the audited pipeline and the `GreenNet` architecture established on `main`, adding a source-domain pre-training stage and a scarce-data fine-tuning framework for the 500-sample `organs` dataset.

## Repository Structure

```text
├── Code/
│   ├── data/                 # Target diagnostic datasets (.pt files) [Git ignored]
│   ├── benchmark.py          # Automated multi-dataset benchmark execution suite
│   ├── config.json           # Execution configuration registry
│   ├── data.py               # Tensor data loading & train/val splitting
│   ├── inspect_data.py       # Diagnostic data checking tool
│   ├── models.py             # Deep learning models (AlexNet, VGG, ResNet, GreenNet)
│   ├── pretrain.py           # Source-domain (orgs) pre-training script
│   ├── train.py              # Standard training and metric execution entry point
│   ├── trainer.py            # Training, evaluation, and latency tracking engine
│   └── transfer.py           # Knowledge Transfer framework (orgs → organs)
├── .gitignore                # Git exclusion rules
├── AUDIT_LOG.md              # Technical inventory of code defects and engineering fixes
├── README.md                 # Project documentation (This file)
├── REPORT.md                 # Consolidated report, including the Data-Scarcity Post-Mortem
├── requirements.txt          # Python environment dependencies
└── results.csv               # Exported benchmark metrics output
```

## Getting Started

### 1. Prerequisites & Environment Setup

Ensure Python 3.10+ is installed. Build a clean virtual environment and load the dependencies:

```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
```

### Install dependencies
```
pip install -r requirements.txt
```

### 2. Dataset Placement
Download the pristine emergency back-up datasets and place them within the nested data directory `data/`:

```bash
mkdir Code/data
# Place cells.pt, chest.pt, lesions.pt, orgs.pt, and organs.pt into the data/ directory
```

### 3. Running the Knowledge Transfer Pipeline

The data-scarcity experiment runs in two stages. First, pre-train `GreenNet` on the larger `orgs` source domain:

```bash
python Code/pretrain.py
```

This caches the learned source weights to `pretrained_orgs.pt`. Then fine-tune on the scarce `organs` target:

```bash
python Code/transfer.py
```

To reproduce the full benchmark suite (all datasets, all models, including the scratch-vs-transfer comparison):

```bash
python Code/benchmark.py
```

## Part 3 Highlights

**Knowledge Transfer Adaptation:** `GreenNet` is first trained on the 15k-sample `orgs` dataset, whose 11-class grayscale task matches `organs` exactly. The learned feature extractor is cached and reloaded as the initialization for fine-tuning on the 500-sample target, so the scarce dataset never has to learn low-level features from scratch.

**Scarce-Data Benchmark Matrix:** The runner tracks `organs` under both training states random initialization (from scratch) and transferred `orgs` features, logging test accuracy, macro F1, and best validation accuracy for each.

**Weight Caching:** Source-domain weights are cached to `pretrained_orgs.pt`. Subsequent runs load the cache and skip Phase 1, avoiding redundant computation. Checkpointing (`best_model.pt`) recovers the best-validation epoch before test evaluation.

**Reproducibility:** All runs use a fixed global seed (42) for consistency across executions.

**Result:** Transfer learning clears the 40% mandate comfortably and beats training from scratch by **+13.25 accuracy points (67.70% vs 54.45%)** and **+16.54 macro F1 points (60.02 vs 43.48)**. The macro F1 gain is the more meaningful one, it shows the transferred features helped across all eleven classes, not just the common ones.

(For the full quantitative analysis, training-curve discussion, and recommendations as more data arrives, see the Data-Scarcity Post-Mortem in `REPORT.md`.)