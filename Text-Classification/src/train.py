import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.data_loader import get_data_loaders
from src.model import SpamClassifier

def train_model(data_path='data/spam.csv', epochs=5, batch_size=32, lr=1e-3, max_features=5000):
    print("Loading data and setting up DataLoaders...")
    train_loader, val_loader, vectorizer = get_data_loaders(data_path, batch_size, max_features)
    
    input_dim = len(vectorizer.vocabulary_)
    print(f"TF-IDF Vocabulary Size: {input_dim}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    model = SpamClassifier(input_dim=input_dim).to(device)
    
    criterion = nn.BCELoss() # Binary Cross Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        # Training Loop
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
            # Record Accuracy
            preds = (outputs > 0.5).float()
            train_correct += (preds == batch_y).sum().item()
            total_train += batch_y.size(0)
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                
                preds = (outputs > 0.5).float()
                val_correct += (preds == batch_y).sum().item()
                total_val += batch_y.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        # Save Model on best validation score
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/saved_model.pth')
            print(f"  -> Saved new best model with Validation Accuracy: {best_val_acc:.4f}")
            
    print("Training Complete!")

if __name__ == '__main__':
    train_model()