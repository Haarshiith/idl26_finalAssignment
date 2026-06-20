"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch

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
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()  # FIX: clear gradients from previous batch
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

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss / total, (correct / total) * 100

    def fit(self, train_loader, val_loader, epochs, checkpoint_path="best_model.pt"):
        print("\n Starting Training Routine...")
        print("-" * 50)

        best_val_acc = 0.0
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            if val_acc > best_val_acc:                       # checkpoint: save weights when val improves
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), checkpoint_path)

            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
                  f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # restore the BEST weights so the model ends in its best state, not the final-epoch state
        self.model.load_state_dict(torch.load(checkpoint_path))

        print("-" * 50)
        print(f"Training Complete! Best Val Acc: {best_val_acc:.2f}%")