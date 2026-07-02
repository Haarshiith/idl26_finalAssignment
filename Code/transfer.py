"""
Part 3 - Knowledge Transfer: transfer stage.
Loads weights pretrained on the large 'orgs' dataset, then fine-tunes on the
small 'organs' dataset. Compare its result against the from-scratch baseline.
"""
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
from data import get_loaders
import models
from fit import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed()

    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Transfer learning on device: {device}")
    print(f"Model: {config['MODEL']} | Dataset: {config['DATA']}")

    train_loader, val_loader, test_loader = get_loaders(
        data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"]
    )

    model_class = getattr(models, config["MODEL"])
    model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"],
                        drop_rate=config["DROP_RATE"], activation_str=None).to(device)

    # --- THE TRANSFER STEP: start from the orgs-pretrained weights instead of random ---
    model.load_state_dict(torch.load("pretrained_orgs.pt", map_location=device))
    print("Loaded pretrained weights from pretrained_orgs.pt")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])

    # --- evaluate on the organs TEST set ---
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu())
            all_labels.append(labels)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    correct = (all_preds == all_labels).sum()
    test_acc = (correct / len(all_labels)) * 100
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    recall    = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100

    print("\n--- Transfer Learning Results (organs) ---")
    print(f"Test accuracy     : {test_acc:.2f}%")
    print(f"Precision (macro) : {precision:.2f}%")
    print(f"Recall (macro)    : {recall:.2f}%")
    print(f"Macro F1          : {macro_f1:.2f}%")


if __name__ == "__main__":
    main()