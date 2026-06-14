import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from data import get_loaders
import models
from fit import Trainer

def evaluate_test_set(model, test_loader, device):
    """Calculates final metrics for REPORT.md"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).squeeze().long()
            
            # THE FIX: Bring over the dynamic channel alignment!
            expected_channels = list(model.parameters())[0].shape[1]
            if images.size(1) == 3 and expected_channels == 1:
                images = images.mean(dim=1, keepdim=True)
            elif images.size(1) == 1 and expected_channels == 3:
                images = images.repeat(1, 3, 1, 1)
                
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)
    
    print("\n" + "="*50)
    print("FINAL TEST SET METRICS FOR REPORT.MD")
    print("="*50)
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {f1:.4f}")
    print("="*50 + "\n")

def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Knowledge Transfer on: {device}")

    # 1. Pre-train on the large 'cells' dataset
    print("\n--- PHASE 1: Pre-training features on 'cells' dataset ---")
    train_loader_src, val_loader_src, _ = get_loaders(data="chest", data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

    model_class = getattr(models, config["MODEL"])
    model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"], drop_rate=0.5, activation_str=config["ACTIVATION"]).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_src = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    
    trainer_src = Trainer(model, criterion, optimizer_src, device)
    trainer_src.fit(train_loader_src, val_loader_src, epochs=5) # 5 epochs to learn features

    # 2. Prepare for Full-Network Fine-Tuning
    print("\n--- PHASE 2: Adapting architecture for 'orgs' ---")
    
    # Fully Unfreeze the Backbone
    for param in model.parameters():
        param.requires_grad = True
        
    # Fresh classifier with 0.5 Dropout
    model.model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(512, config["NUM_CLASSES"])
    ).to(device)

    # 3. Fine-tune on the 'organs' dataset
    train_loader_tgt, val_loader_tgt, test_loader_tgt = get_loaders(data="organs", data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])
    
    # THE FIX: Warm fine-tuning rate (0.0002) with strict Weight Decay
    # We remove Label Smoothing to allow the network to confidently map the standard metrics.
    optimizer_tgt = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-3)
    criterion_tgt = nn.CrossEntropyLoss()
    
    # Sync to 20 epochs (update your run_benchmarks.py pairing to EPOCHS: 20)
    trainer_tgt = Trainer(model, criterion_tgt, optimizer_tgt, device)
    trainer_tgt.fit(train_loader_tgt, val_loader_tgt, epochs=20)

    # Load the optimal weights before final evaluation
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))

    # 4. Generate Final Metrics for the Report
    evaluate_test_set(model, test_loader_tgt, device)

if __name__ == "__main__":
    main()