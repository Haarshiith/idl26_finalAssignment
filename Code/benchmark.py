"""
Part 2 - Green Initiative: efficiency profiling.
Profiles TRAINING (runtime + peak memory) and INFERENCE (latency/sample + peak memory),
records test accuracy, and appends one row per run to results.csv.
"""
import json
import time
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer


def main():
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

    # ---- profile the TRAINING phase ----
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

    # ---- profile the INFERENCE phase ----
    model.eval()

    # free training-only memory so inference reflects ONLY the forward pass
    optimizer.zero_grad(set_to_none=True)
    del optimizer, trainer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # warm-up pass (don't measure one-time GPU/cuDNN startup cost)
    with torch.no_grad():
        for images, _ in test_loader:
            model(images.to(device))
            break

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    correct, total = 0, 0
    start = time.perf_counter()
    with torch.no_grad():
        for images, labels in test_loader:          # now also read labels
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)            # predicted class
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_runtime = time.perf_counter() - start

    test_acc = (correct / total) * 100
    latency_ms = (infer_runtime / total) * 1000      # total time / samples -> ms per sample

    if device.type == "cuda":
        peak_infer_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_infer_mem = float("nan")

    # ---- report to screen ----
    print("\n--- Efficiency Summary ---")
    print(f"Test accuracy           : {test_acc:.2f}%")
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
            writer.writerow(["model", "dataset", "test_acc", "train_runtime_s",
                             "peak_train_mem_mb", "latency_ms", "peak_infer_mem_mb"])
        writer.writerow([config["MODEL"], config["DATA"], f"{test_acc:.2f}",
                         f"{train_runtime:.2f}", f"{peak_train_mem:.1f}",
                         f"{latency_ms:.4f}", f"{peak_infer_mem:.1f}"])
    print(f"\nAppended results to {results_path}")


if __name__ == "__main__":
    main()