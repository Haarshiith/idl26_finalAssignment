"""
Part 2 - Green Initiative: efficiency profiling.
Profiles TRAINING (runtime + peak memory) and INFERENCE (latency per sample + peak memory).
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

    # warm-up pass: the FIRST inference triggers one-time GPU/cuDNN setup,
    # which would inflate the timing. Run one batch first and don't measure it.
    optimizer.zero_grad(set_to_none=True)   # drop the stored gradients
    del optimizer, trainer                  # release Adam's internal state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with torch.no_grad():
        for images, _ in test_loader:
            model(images.to(device))
            break

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)   # reset gauge for the inference phase
        torch.cuda.synchronize(device)

    num_samples = 0
    start = time.perf_counter()
    with torch.no_grad():                            # no gradients needed for prediction
        for images, _ in test_loader:
            model(images.to(device))
            num_samples += images.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_runtime = time.perf_counter() - start

    latency_ms = (infer_runtime / num_samples) * 1000   # total time / samples -> ms per sample

    if device.type == "cuda":
        peak_infer_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_infer_mem = float("nan")

    # ---- report ----
    print("\n--- Efficiency Summary ---")
    print(f"Total training runtime  : {train_runtime:.2f} s")
    print(f"Peak training memory    : {peak_train_mem:.1f} MB")
    print(f"Inference latency       : {latency_ms:.3f} ms/sample")
    print(f"Peak inference memory   : {peak_infer_mem:.1f} MB")


if __name__ == "__main__":
    main()