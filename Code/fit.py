"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
import time
import torchvision.transforms as T

class Trainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        correct, sum = 0, 0
        
        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device).squeeze().long()

            expected_channels = list(self.model.parameters())[0].shape[1]
            if images.size(1) == 3 and expected_channels == 1:
                images = images.mean(dim=1, keepdim=True)
            elif images.size(1) == 1 and expected_channels == 3:
                images = images.repeat(1, 3, 1, 1)
            
            augmentations = T.Compose([
                T.RandomAffine(degrees=0, translate=(0.08, 0.08)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            images = augmentations(images)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            sum += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        return running_loss / sum, (correct / sum) * 100

    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct, total = 0, 0
        
        start_time = time.time()

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device).squeeze().long()

                expected_channels = list(self.model.parameters())[0].shape[1]
                if images.size(1) == 3 and expected_channels == 1:
                    images = images.mean(dim=1, keepdim=True)
                elif images.size(1) == 1 and expected_channels == 3:
                    images = images.repeat(1, 3, 1, 1)
                
                normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                images = normalize(images)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        end_time = time.time()
        latency_per_sample = (end_time - start_time) / total if total > 0 else 0
                
        return running_loss / total, (correct / total) * 100, latency_per_sample

    def fit(self, train_loader, val_loader, epochs):
        print("\n Starting Training Routine...")
        print("-" * 50)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            
        start_time = time.time()  

        best_val_acc = 0.0    

        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_latency = self.evaluate(val_loader)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_model.pth")
            
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")
        
        total_time = time.time() - start_time
        peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2) if torch.cuda.is_available() else 0

        print("-" * 50)
        print("Training Complete!")
        print(f"Total Training Time: {total_time:.2f} seconds")
        print(f"Inference Latency: {val_latency:.6f} seconds/sample")
        print(f"Peak GPU Memory: {peak_memory:.2f} MB\n")
