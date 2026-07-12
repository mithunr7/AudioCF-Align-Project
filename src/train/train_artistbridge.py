import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sys

sys.path.append('.')
from src.models.artistbridge import ArtistBridge
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ProjectionDataset(Dataset):
    def __init__(self, df, audio_dict, cf_dict, track_to_artist, artist_to_tracks):
        self.data = []
        for _, row in tqdm(df.iterrows(), desc="Building Dataset", total=len(df)):
            t = row['track_id']
            if t not in audio_dict or t not in cf_dict:
                continue
                
            audio_vec = audio_dict[t]
            true_cf = cf_dict[t]
            artist = track_to_artist.get(t, "")
            
            has_artist = 0.0
            loo_cf = np.zeros(128, dtype=np.float32)
            
            if artist in artist_to_tracks:
                artist_tracks = artist_to_tracks[artist]
                loo_tracks = [x for x in artist_tracks if x != t and x in cf_dict]
                if len(loo_tracks) > 0:
                    loo_cfs = [cf_dict[x] for x in loo_tracks]
                    loo_cf = np.mean(loo_cfs, axis=0)
                    has_artist = 1.0
                    
            self.data.append({
                'audio': torch.tensor(audio_vec, dtype=torch.float32),
                'true_cf': torch.tensor(true_cf, dtype=torch.float32),
                'loo_cf': torch.tensor(loo_cf, dtype=torch.float32),
                'has_artist': torch.tensor(has_artist, dtype=torch.float32)
            })
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_artist", type=float, default=0.5, help="Weight of LOO Artist Anchor Loss")
    parser.add_argument("--out_model", type=str, default="data/embeddings/artistbridge_model.pt", help="Path to save the model weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    set_seed(args.seed)

    print("Loading data...")
    train_df = pd.read_parquet("data/processed/train.parquet")
    
    # We only want UNIQUE tracks for training the projection
    all_warm_tracks = train_df[['track_id']].drop_duplicates()
    
    # Split 90% for training the projection, 10% for validation
    all_warm_tracks = all_warm_tracks.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_size = int(len(all_warm_tracks) * 0.1)
    
    val_tracks = all_warm_tracks.iloc[:val_size].copy()
    train_tracks = all_warm_tracks.iloc[val_size:].copy()
    
    metadata = pd.read_csv("data/raw/track_metadata.csv")
    track_to_artist = dict(zip(metadata['track_id'], metadata['artist_name']))
    
    # Build artist_to_tracks ONLY from the training set to prevent leakage
    train_with_meta = train_tracks.merge(metadata, on='track_id', how='inner')
    artist_to_tracks_train = train_with_meta.groupby('artist_name')['track_id'].apply(list).to_dict()
    
    audio_df = pd.read_parquet("data/embeddings/audio_embeddings.parquet")
    cf_df = pd.read_parquet("data/embeddings/cf_embeddings.parquet")
    
    audio_dict = audio_df.set_index('track_id').T.to_dict('list')
    cf_dict = cf_df.set_index('track_id').T.to_dict('list')
    
    print("Building datasets...")
    train_dataset = ProjectionDataset(train_tracks, audio_dict, cf_dict, track_to_artist, artist_to_tracks_train)
    val_dataset = ProjectionDataset(val_tracks, audio_dict, cf_dict, track_to_artist, artist_to_tracks_train)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ArtistBridge(input_dim=100, hidden_dim=128, output_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    mse_criterion = nn.MSELoss()
    cos_criterion = nn.CosineEmbeddingLoss()
    
    print(f"Training on {device}...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            audio = batch['audio'].to(device)
            true_cf = batch['true_cf'].to(device)
            loo_cf = batch['loo_cf'].to(device)
            has_artist = batch['has_artist'].to(device)
            
            optimizer.zero_grad()
            pred_cf = model(audio)
            
            # Primary MSE
            loss_mse = mse_criterion(pred_cf, true_cf)
            
            # Cosine Loss (target=1 means we want them to be parallel)
            target_cos = torch.ones(audio.size(0)).to(device)
            loss_cos = cos_criterion(pred_cf, true_cf, target_cos)
            
            # LOO Artist Anchor Loss
            # Only apply for tracks that have a LOO artist proxy
            artist_mse = F.mse_loss(pred_cf, loo_cf, reduction='none').mean(dim=1)
            loss_artist = (artist_mse * has_artist).mean()
            
            loss = loss_mse + loss_cos + (args.lambda_artist * loss_artist)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        model.eval()
        total_val_mse = 0.0
        total_val_cos = 0.0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(device)
                true_cf = batch['true_cf'].to(device)
                
                pred_cf = model(audio)
                total_val_mse += mse_criterion(pred_cf, true_cf).item()
                
                target_cos = torch.ones(audio.size(0)).to(device)
                total_val_cos += cos_criterion(pred_cf, true_cf, target_cos).item()
                
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_mse = total_val_mse / len(val_loader)
        avg_val_cos = total_val_cos / len(val_loader)
        
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val MSE: {avg_val_mse:.4f} | Val Cosine Loss: {avg_val_cos:.4f}")
        
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            torch.save(model.state_dict(), args.out_model)
            print(f"  --> Saved new best model to {args.out_model}")

if __name__ == "__main__":
    main()
