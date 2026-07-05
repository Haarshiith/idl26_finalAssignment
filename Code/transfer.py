import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data import get_loaders
from trainer import Trainer
import torchvision.transforms as T
import numpy as np
import os
import torchvision.models as tv_models
import torch.nn.functional as F

class TransferResNet18(nn.Module):
    def __init__(self, num_classes=11, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = tv_models.resnet18(weights=weights)
        
        # Fresh head with dropout, matching the Scratch model perfectly
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
        
    def forward(self, x):
        # Equalizes resolution to 224x224, matching the Scratch model
        if x.size(2) < 224 or x.size(3) < 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)

def evaluate_test_set(model, test_loader, device):
    """Calculates final metrics for REPORT.md"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device).squeeze().long()
            
            # Dynamic Channel Alignment
            expected_channels = list(model.parameters())[0].shape[1]
            if images.size(1) == 3 and expected_channels == 1:
                images = images.mean(dim=1, keepdim=True)
            elif images.size(1) == 1 and expected_channels == 3:
                images = images.repeat(1, 3, 1, 1)
                
            # Dynamic Normalization
            if images.size(1) == 1:
                normalize = T.Normalize(mean=[0.485], std=[0.229])
            else:
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
            images = normalize(images)
                
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)

    # Calculate aggregate confusion matrix metrics
    cm = confusion_matrix(all_targets, all_preds)
    TP = np.diag(cm).sum()
    FP = (cm.sum(axis=0) - np.diag(cm)).sum()
    
    print("\n" + "="*50)
    print("FINAL TEST SET METRICS FOR REPORT.MD")
    print("="*50)
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {f1:.4f}")
    print("-" * 50)
    print(f"Standard Metrics Total -> Correct (TP): {TP} | Misclassified: {FP}")
    print("="*50 + "\n")

def main():
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Knowledge Transfer Pipeline on: {device}")

    # ---------------------------------------------------------
    # PHASE 1: Source Domain Pre-training (Massive 'orgs' dataset)
    # ---------------------------------------------------------
    print("\n--- PHASE 1: Pre-training features on massive 'orgs' baseline ---")
    
    model = TransferResNet18(num_classes=11, pretrained=True).to(device)

    cache_path = "orgs_pretrained_base.pth"
    
    if os.path.exists(cache_path):
        print(f"[MLOps] Found cached Phase 1 weights at '{cache_path}'. Bypassing redundant training...")
        model.load_state_dict(torch.load(cache_path, weights_only=True))
    else:
        # Load the massive 15k 'orgs' dataset for feature extraction
        train_loader_src, val_loader_src, _ = get_loaders(data="orgs", data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])
        
        optimizer_src = optim.Adam(model.parameters(), lr=0.001)
        criterion_src = nn.CrossEntropyLoss()
        
        trainer_src = Trainer(model, criterion_src, optimizer_src, device)
        trainer_src.fit(train_loader_src, val_loader_src, epochs=15)
        
        torch.save(model.state_dict(), cache_path)
        print(f"\n[MLOps] Phase 1 weights cached successfully to '{cache_path}'.")

    # ---------------------------------------------------------
    # PHASE 2: Target Domain Fine-Tuning (Scarce Data)
    # ---------------------------------------------------------
    print(f"\n--- PHASE 2: Fine-tuning on target scarce dataset '{config['DATA']}' ---")
    
    # Unfreeze everything for gentle fine-tuning
    for param in model.parameters():
        param.requires_grad = True 
        
    train_loader_tgt, val_loader_tgt, test_loader_tgt = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])
    
    # Add gentle L2 regularization (weight_decay) to prevent overfitting the tiny dataset
    optimizer_tgt = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-3)
    criterion_tgt = nn.CrossEntropyLoss()

    scheduler_tgt = optim.lr_scheduler.ReduceLROnPlateau(optimizer_tgt, mode='max', factor=0.5, patience=3)
    
    trainer_tgt = Trainer(model, criterion_tgt, optimizer_tgt, device)
    trainer_tgt.fit(train_loader_tgt, val_loader_tgt, epochs=config["EPOCHS"], scheduler=scheduler_tgt)

    # Load the optimal weights before final evaluation
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    evaluate_test_set(model, test_loader_tgt, device)

if __name__ == "__main__":
    main()