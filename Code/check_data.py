import torch
from pathlib import Path

d = torch.load(Path("data") / "orgs.pt", weights_only=False)
print("keys:", list(d.keys()))
print("train_images:", d["train_images"].shape, d["train_images"].dtype)
print("train_labels:", d["train_labels"].shape, d["train_labels"].dtype)
print("test_images :", d["test_images"].shape)
print("label range :", int(d["train_labels"].min()), "->", int(d["train_labels"].max()))
print("pixel range :", float(d["train_images"].min()), "->", float(d["train_images"].max()))