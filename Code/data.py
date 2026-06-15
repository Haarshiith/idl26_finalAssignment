"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):
    d_path = Path(data_path) / f"{data}.pt"
    data_dict = torch.load(d_path)

    # Calculate validation split boundaries
    total_samples = data_dict['train_images'].shape[0]
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    # Segment training and validation partitions
    train_data = data_dict['train_images']
    train_labels = data_dict['train_labels']
    val_data = data_dict['train_images'][val_start:]
    train_data = data_dict['train_images'][:val_start]
    val_labels = data_dict['train_labels'][val_start:]
    train_labels = data_dict['train_labels'][:val_start]
    
    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)
    test_dataset = TensorDataset(data_dict['test_images'], data_dict['test_labels'])
    
    # Configure loaders (drop_last prevents batch norm crashes on uneven final batches)    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader