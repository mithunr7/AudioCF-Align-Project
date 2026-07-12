import torch
import torch.nn as nn
import torch.nn.functional as F

class ArtistBridge(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=128, output_dim=128, dropout=0.1):
        super(ArtistBridge, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # We do NOT use a final activation like tanh/sigmoid here, 
        # as the BPR latent space is continuous and unbounded.
        return x
