import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from data import get_loaders
import models
from fit import Trainer
import torchvision.transforms as T
import numpy as np

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
                
            # Dynamic Normalization based on the aligned channels
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

    # Calculate aggregate confusion matrix metrics for the report
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
    print(f"Training executing on device: {device}")

    train_loader, val_loader, test_loader = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

    model_class = getattr(models, config["MODEL"])
    model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"], drop_rate=0.5, activation_str=config["ACTIVATION"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-3)

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])

    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    
    evaluate_test_set(model, test_loader, device)

if __name__ == "__main__":
    main()