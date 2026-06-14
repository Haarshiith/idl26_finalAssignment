"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):
    d_path = Path(data_path) / f"{data}.pt"
    data_dict = torch.load(d_path, weights_only=False)

    images       = data_dict['train_images']
    labels       = data_dict['train_labels'].squeeze(1)   # [N,1] -> [N]
    test_images  = data_dict['test_images']
    test_labels  = data_dict['test_labels'].squeeze(1)

    total_samples = images.shape[0]
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    # FIX (leak): train must EXCLUDE the val slice
    train_data   = images[:val_start]
    train_labels = labels[:val_start]
    val_data     = images[val_start:]
    val_labels   = labels[val_start:]

    # Input normalization (z-score). Statistics computed on TRAIN ONLY, then applied to all splits.
    mean = train_data.mean(dim=[0, 2, 3], keepdim=True)   # per-channel mean -> shape [1, C, 1, 1]
    std  = train_data.std(dim=[0, 2, 3], keepdim=True)
    std  = std.clamp(min=1e-7)                            # guard against divide-by-zero

    train_data  = (train_data  - mean) / std
    val_data    = (val_data    - mean) / std
    test_images = (test_images - mean) / std             # same train stats applied to test

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset   = TensorDataset(val_data, val_labels)
    test_dataset  = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader