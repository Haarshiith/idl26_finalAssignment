"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):
    d_path = Path(data_path) / f"{data}.pt"          # fixed: was f"{data}_data.pt"
    data_dict = torch.load(d_path, weights_only=False)

    total_samples = data_dict['train_images'].shape[0]
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    # NOTE (Phase 2): train still includes the val slice -> train/val leak to fix later.
    train_data = data_dict['train_images']
    train_labels = data_dict['train_labels'].squeeze(1)            # fixed: [N,1] -> [N]
    val_data = data_dict['train_images'][val_start:]
    val_labels = data_dict['train_labels'][val_start:].squeeze(1)  # fixed: [N,1] -> [N]

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(data_dict['test_images'], data_dict['test_labels'].squeeze(1))  # fixed

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader