"""
Part 2 - Green Initiative: efficiency profiling.
Profiles the TRAINING phase: total runtime + peak memory.
"""
import json
import time

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
        torch.cuda.reset_peak_memory_stats(device)   # start the memory high-water mark fresh
        torch.cuda.synchronize(device)               # make sure GPU is idle before we start timing

    start = time.perf_counter()
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)               # wait for ALL GPU work to finish before stopping clock
    train_runtime = time.perf_counter() - start

    if device.type == "cuda":
        peak_train_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # bytes -> MB
    else:
        peak_train_mem = float("nan")  # CPU has no equivalent

    print("\n--- Efficiency (training phase) ---")
    print(f"Total training runtime : {train_runtime:.2f} s")
    print(f"Peak training memory   : {peak_train_mem:.1f} MB")


if __name__ == "__main__":
    main()