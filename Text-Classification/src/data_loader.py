import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os

from src.utils import preprocess_text, label_to_int

class SpamDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # features is a scipy sparse matrix from TF-IDF
        x = torch.tensor(self.features[idx].toarray().squeeze(), dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

def get_data_loaders(csv_path: str, batch_size: int = 32, max_features: int = 5000):
    """
    Loads data, preprocesses, splits, vectorizes, and returns DataLoaders.
    """
    # Load dataset with fallback encoding
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    
    # Handle both common format variations gracefully
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['label', 'text']
    
    # Ensure there are no NA values disrupting training
    df.dropna(subset=['label', 'text'], inplace=True)
    
    # Apply text preprocessing and label conversion
    df['text'] = df['text'].apply(preprocess_text)
    df['label'] = df['label'].apply(label_to_int)
    
    X = df['text'].values
    y = df['label'].values
    
    # 80-20 Train validation split ensuring class distribution ratio remains the same
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Upgrade to TF-IDF instead of CountVectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # Save the vectorizer using pickle so it can be re-loaded accurately in predict.py
    os.makedirs('models', exist_ok=True)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    # Standard PyTorch mapping datasets
    train_dataset = SpamDataset(X_train_vec, y_train)
    val_dataset = SpamDataset(X_val_vec, y_val)
    
    # DataLoaders for mini-batching iteration
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, vectorizer