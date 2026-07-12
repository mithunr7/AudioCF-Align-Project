import torch
import torch.nn as nn

class CFTeacher(nn.Module):
    def __init__(self, num_users, num_tracks, embed_dim=128):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.track_embed = nn.Embedding(num_tracks, embed_dim)
        
        # Initialize embeddings using normal distribution
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.track_embed.weight, std=0.01)
        
    def forward(self, user_indices, pos_track_indices, neg_track_indices):
        """
        user_indices: (B,)
        pos_track_indices: (B,)
        neg_track_indices: (B, num_negatives)
        """
        u = self.user_embed(user_indices) # (B, D)
        pos_i = self.track_embed(pos_track_indices) # (B, D)
        neg_i = self.track_embed(neg_track_indices) # (B, num_negatives, D)
        
        # Positive scores
        pos_scores = (u * pos_i).sum(dim=1) # (B,)
        
        # Negative scores
        u_expanded = u.unsqueeze(1) # (B, 1, D)
        neg_scores = (u_expanded * neg_i).sum(dim=2) # (B, num_negatives)
        
        return pos_scores, neg_scores
    
    def bpr_loss(self, pos_scores, neg_scores):
        """
        Calculates BPR loss.
        pos_scores: (B,)
        neg_scores: (B, num_negatives)
        """
        pos_scores = pos_scores.unsqueeze(1) # (B, 1)
        # BPR Loss = -mean(log(sigmoid(pos_score - neg_score)))
        loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores).mean()
        return loss
        
    def predict(self, user_indices, track_indices):
        u = self.user_embed(user_indices)
        i = self.track_embed(track_indices)
        return (u * i).sum(dim=1)
