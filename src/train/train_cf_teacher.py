import os
import argparse
import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CFDataset, encode_data
from src.models.cf_teacher import CFTeacher

def evaluate_metrics(model, val_loader, device, train_df, val_df, user2idx, track2idx):
    """
    Computes strict Recall@10 and NDCG@10 for the CF model and a Popularity baseline.
    """
    model.eval()
    
    # 1. Popularity Baseline
    track_popularity = train_df['track_id'].value_counts().to_dict()
    # Map to encoded tracks
    pop_scores = np.zeros(len(track2idx))
    for t, count in track_popularity.items():
        if t in track2idx:
            pop_scores[track2idx[t]] = count
            
    # Precompute user histories to filter out warm tracks during evaluation
    user_histories = train_df.groupby('user_id')['track_id'].apply(lambda x: set(track2idx[t] for t in x if t in track2idx)).to_dict()
    
    # Group validation ground truth by user
    val_encoded = val_df.copy()
    val_encoded['user_id'] = val_encoded['user_id'].astype(str).map(user2idx)
    val_encoded['track_id'] = val_encoded['track_id'].astype(str).map(track2idx)
    val_encoded = val_encoded.dropna()
    val_ground_truth = val_encoded.groupby('user_id')['track_id'].apply(set).to_dict()
    
    cf_recall = []
    cf_ndcg = []
    pop_recall = []
    pop_ndcg = []
    
    # Evaluate users who have validation data
    users_to_eval = list(val_ground_truth.keys())
    
    # Batch the evaluation
    batch_size = 512
    all_tracks = torch.arange(len(track2idx)).to(device)
    
    for i in range(0, len(users_to_eval), batch_size):
        batch_users = users_to_eval[i:i+batch_size]
        
        # --- Popularity ---
        for u in batch_users:
            history = user_histories.get(u, set())
            gt = val_ground_truth[u]
            
            # Mask out history
            u_pop = pop_scores.copy()
            u_pop[list(history)] = -np.inf
            
            # Top 10
            top10 = np.argsort(u_pop)[-10:][::-1]
            hits = len(set(top10).intersection(gt))
            pop_recall.append(hits / min(len(gt), 10))
            
            # NDCG
            dcg = sum([1.0 / np.log2(idx + 2) for idx, t in enumerate(top10) if t in gt])
            idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(gt), 10))])
            pop_ndcg.append(dcg / idcg if idcg > 0 else 0)
            
        # --- CF Model ---
        users_tensor = torch.tensor(batch_users, dtype=torch.long).to(device)
        with torch.no_grad():
            u_embeds = model.user_embed(users_tensor) # (B, D)
            i_embeds = model.track_embed.weight # (num_tracks, D)
            scores = torch.matmul(u_embeds, i_embeds.T).cpu().numpy() # (B, num_tracks)
            
        for b_idx, u in enumerate(batch_users):
            history = user_histories.get(u, set())
            gt = val_ground_truth[u]
            
            u_scores = scores[b_idx].copy()
            u_scores[list(history)] = -np.inf
            
            top10 = np.argsort(u_scores)[-10:][::-1]
            hits = len(set(top10).intersection(gt))
            cf_recall.append(hits / min(len(gt), 10))
            
            dcg = sum([1.0 / np.log2(idx + 2) for idx, t in enumerate(top10) if t in gt])
            idcg = sum([1.0 / np.log2(idx + 2) for idx in range(min(len(gt), 10))])
            cf_ndcg.append(dcg / idcg if idcg > 0 else 0)
            
    res = {
        "pop_recall": np.mean(pop_recall),
        "pop_ndcg": np.mean(pop_ndcg),
        "cf_recall": np.mean(cf_recall),
        "cf_ndcg": np.mean(cf_ndcg),
    }
    return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    print("Loading processed data...")
    train_df = pd.read_parquet("data/processed/train.parquet")
    val_df = pd.read_parquet("data/processed/val_warm.parquet")
    
    train_enc, val_enc, user2idx, track2idx = encode_data(train_df, val_df)
    num_users = len(user2idx)
    num_tracks = len(track2idx)
    
    print(f"Num Users: {num_users}, Num Tracks: {num_tracks}")
    
    train_dataset = CFDataset(train_enc, num_tracks, is_train=True)
    val_dataset = CFDataset(val_enc, num_tracks, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    model = CFTeacher(num_users, num_tracks, embed_dim=128).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            users, pos_tracks, neg_tracks = batch
            users = users.to(args.device)
            pos_tracks = pos_tracks.to(args.device)
            neg_tracks = neg_tracks.to(args.device)
            
            optimizer.zero_grad()
            pos_scores, neg_scores = model(users, pos_tracks, neg_tracks)
            loss = model.bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
    print("Evaluating Popularity Baseline vs CF Model...")
    metrics = evaluate_metrics(model, val_loader, args.device, train_df, val_df, user2idx, track2idx)
    
    print(f"\n--- Final Verification Metrics ---")
    print(f"Popularity Baseline Recall@10: {metrics['pop_recall']:.4f}")
    print(f"CF Model Validation Recall@10: {metrics['cf_recall']:.4f}")
    print(f"Popularity Baseline NDCG@10:   {metrics['pop_ndcg']:.4f}")
    print(f"CF Model Validation NDCG@10:   {metrics['cf_ndcg']:.4f}")
    
    # Extract track embeddings
    model.eval()
    with torch.no_grad():
        track_weights = model.track_embed.weight.cpu().numpy()
        
    idx2track = {v: k for k, v in track2idx.items()}
    track_ids = [idx2track[i] for i in range(num_tracks)]
    
    embeddings_df = pd.DataFrame(track_weights)
    embeddings_df['track_id'] = track_ids
    cols = ['track_id'] + [c for c in embeddings_df.columns if c != 'track_id']
    embeddings_df = embeddings_df[cols]
    
    os.makedirs("data/embeddings", exist_ok=True)
    embeddings_df.to_parquet("data/embeddings/cf_embeddings.parquet")
    
    torch.save(model.state_dict(), "data/embeddings/cf_model.pt")
    
    with open("data/embeddings/user2idx.json", "w") as f:
        json.dump(user2idx, f)
    with open("data/embeddings/track2idx.json", "w") as f:
        json.dump(track2idx, f)
        
    print("Saved CF teacher model, embeddings, and mappings.")
    print("Stage 3 Completed successfully!")

if __name__ == "__main__":
    main()
