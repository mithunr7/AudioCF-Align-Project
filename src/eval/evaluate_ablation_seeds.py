import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
import sys

sys.path.append('.')
from src.models.artistbridge import ArtistBridge
from src.models.cf_teacher import CFTeacher

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    train_df = pd.read_parquet("data/processed/train.parquet")
    test_semi = pd.read_parquet("data/processed/test_semi_cold.parquet")
    
    # We only care about tracks
    audio_df = pd.read_parquet("data/embeddings/audio_embeddings.parquet")
    
    model = CFTeacher(num_users=59345, num_tracks=54538, embed_dim=128).to(device)
    model.load_state_dict(torch.load("data/embeddings/cf_model.pt", map_location=device))
    model.eval()
    cf_weights_warm = model.track_embed.weight.detach().cpu().numpy()
    
    num_warm_tracks = cf_weights_warm.shape[0]
    
    import json
    with open("data/embeddings/user2idx.json", "r") as f:
        user2idx = json.load(f)
    with open("data/embeddings/track2idx.json", "r") as f:
        track2idx_warm = json.load(f)
    
    unified_track2idx = {k: v for k, v in track2idx_warm.items()}
    test_tracks = test_semi['track_id'].unique()
    for i, t in enumerate(test_tracks):
        if t not in unified_track2idx:
            unified_track2idx[t] = num_warm_tracks + i
            
    total_pool_size = len(unified_track2idx)
    
    # Precompute train histories map (user -> set of warm track indices)
    train_histories = train_df.groupby('user_id')['track_id'].apply(lambda x: set(track2idx_warm[t] for t in x if t in track2idx_warm)).to_dict()
    
    all_audio_tensor = torch.tensor(audio_df.drop(columns=['track_id']).values, dtype=torch.float32).to(device)
    
    seeds = [1, 2, 3]
    lambdas = [0.0, 0.5]
    
    results = {}
    raw_hits = {}
    
    # Evaluate each model
    for s in seeds:
        for l in lambdas:
            path = f"data/embeddings/ablation_l{l}_s{s}.pt"
            ab_model = ArtistBridge(input_dim=100, hidden_dim=128, output_dim=128).to(device)
            ab_model.load_state_dict(torch.load(path, map_location=device))
            ab_model.eval()
            
            with torch.no_grad():
                ab_embeddings = ab_model(all_audio_tensor).cpu().numpy()
                
            ab_item_matrix = np.zeros((total_pool_size, 128))
            for idx, t in enumerate(audio_df['track_id'].values):
                if t in unified_track2idx:
                    ab_item_matrix[unified_track2idx[t]] = ab_embeddings[idx]
                    
            # Normalize pool norms
            pool = list(range(num_warm_tracks))
            ab_pool_norms = [ab_item_matrix[t] / (np.linalg.norm(ab_item_matrix[t]) + 1e-9) for t in pool]
            
            # Score
            np.random.seed(42)
            u_ab_cos_rec = []
            raw_hits_list = []
            
            batch_size = 512
            users = test_semi['user_id'].unique()
            user_mapping = user2idx
            
            from torch.utils.data import DataLoader
            dataloader = DataLoader(list(test_semi.groupby('user_id')), batch_size=batch_size, collate_fn=lambda x: ( [u for u, df in x], [df['track_id'].tolist() for u, df in x] ))
            
            for pool_idx, (batch_users, batch_tracks) in enumerate(dataloader):
                for i in range(len(batch_users)):
                    u = batch_users[i]
                    u_idx = user_mapping.get(str(u), 0)
                    true_tracks = [unified_track2idx[t] for t in batch_tracks[i]]
                    history = train_histories.get(u, set())
                    
                    u_emb = model.user_embed.weight[u_idx].detach().cpu().numpy()
                    u_emb_norm = u_emb / (np.linalg.norm(u_emb) + 1e-9)
                    
                    for true_t in true_tracks:
                        negs = []
                        while len(negs) < 99:
                            cand = np.random.randint(0, num_warm_tracks)
                            if cand not in history and cand != true_t:
                                negs.append(cand)
                        
                        target_norm = ab_item_matrix[true_t] / (np.linalg.norm(ab_item_matrix[true_t]) + 1e-9)
                        target_score = np.dot(u_emb_norm, target_norm)
                        
                        neg_scores = [np.dot(u_emb_norm, ab_pool_norms[c]) for c in negs]
                        
                        rank = sum(1 for s in neg_scores if s > target_score)
                        hit = 1.0 if rank < 10 else 0.0
                        
                        raw_hits_list.append(hit)
            
            results[(s, l)] = np.mean(raw_hits_list)
            raw_hits[(s, l)] = raw_hits_list
            print(f"Seed {s}, lambda {l} -> Recall@10: {results[(s, l)]:.4f}")
            
    print("\n--- Summary ---")
    for s in seeds:
        hits_0 = raw_hits[(s, 0.0)]
        hits_05 = raw_hits[(s, 0.5)]
        t_stat, p_val = stats.ttest_rel(hits_0, hits_05)
        print(f"Seed {s}: lambda=0.0: {results[(s, 0.0)]:.4f} vs lambda=0.5: {results[(s, 0.5)]:.4f} | Paired t={t_stat:.4f}, p={p_val:.4e}")

if __name__ == "__main__":
    main()
