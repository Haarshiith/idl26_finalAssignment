"""
Part 2 - Green Initiative: efficiency profiling + Part 1 test metrics.
Profiles TRAINING (runtime + peak memory) and INFERENCE (latency/sample + peak memory),
and records test accuracy, precision, recall, macro-F1. Appends one row to results.csv.
"""
import json
import time
import csv
import os
import random              # <-- NEW
import numpy as np  
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from data import get_loaders
import models
from fit import Trainer

def set_seed(seed=42):     # <-- NEW: define the function here, after imports
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True    # force repeatable GPU ops
    # torch.backends.cudnn.benchmark = False       # turn off algorithm auto-tuning

def main():
    set_seed() 
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")
    print(f"Model: {config['MODEL']} | Dataset: {config['DATA']}")

    train_loader, val_loader, test_loader = get_loaders(
        data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"]
    )

    model_class = getattr(models, config["MODEL"])
    model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"],
                        drop_rate=config["DROP_RATE"], activation_str=None).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    trainer = Trainer(model, criterion, optimizer, device)

    # ---- profile the TRAINING phase (Part 2) ----
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_runtime = time.perf_counter() - start

    if device.type == "cuda":
        peak_train_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_train_mem = float("nan")

    # ---- profile the INFERENCE phase (Part 2) ----
    model.eval()

    optimizer.zero_grad(set_to_none=True)
    del optimizer, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():                      # warm-up (don't measure GPU/cuDNN startup)
        for images, _ in test_loader:
            model(images.to(device))
            break

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    correct, total = 0, 0
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_runtime = time.perf_counter() - start

    test_acc = (correct / total) * 100
    latency_ms = (infer_runtime / total) * 1000

    if device.type == "cuda":
        peak_infer_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_infer_mem = float("nan")

    # ---- classification metrics (Part 1): separate UNTIMED pass so Part 2 numbers stay clean ----
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu())     # small tensors -> move off GPU
            all_labels.append(labels)             # already on CPU from the loader
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    # ---- report to screen ----
    print("\n--- Results Summary ---")
    print(f"Test accuracy           : {test_acc:.2f}%")
    print(f"Precision (macro)       : {precision:.2f}%")
    print(f"Recall (macro)          : {recall:.2f}%")
    print(f"Macro F1                : {macro_f1:.2f}%")
    print(f"Total training runtime  : {train_runtime:.2f} s")
    print(f"Peak training memory    : {peak_train_mem:.1f} MB")
    print(f"Inference latency       : {latency_ms:.4f} ms/sample")
    print(f"Peak inference memory   : {peak_infer_mem:.1f} MB")

    # ---- append one row to results.csv ----
    results_path = "results.csv"
    write_header = not os.path.exists(results_path)
    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "dataset", "test_acc", "precision", "recall", "macro_f1",
                             "train_runtime_s", "peak_train_mem_mb", "latency_ms", "peak_infer_mem_mb"])
        writer.writerow([config["MODEL"], config["DATA"], f"{test_acc:.2f}",
                         f"{precision:.2f}", f"{recall:.2f}", f"{macro_f1:.2f}",
                         f"{train_runtime:.2f}", f"{peak_train_mem:.1f}",
                         f"{latency_ms:.4f}", f"{peak_infer_mem:.1f}"])
    print(f"\nAppended results to {results_path}")


if __name__ == "__main__":
    main()