import torch
import torch.nn as nn

class SpamClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout_rate: float = 0.5):
        """
        A PyTorch text classification neural network designed for dynamic inputs.
        Input size connects directly to the TF-IDF feature space size.
        """
        super(SpamClassifier, self).__init__()
        
        # Deep Neural Network Architecture
        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second hidden layer with bottlenecking
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output Layer
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() # Squashes output between 0 and 1 (binary format)
        )

    def forward(self, x):
        # [batch_size, 1] output squeezed to [batch_size] to match target vector
        return self.network(x).squeeze(1)