import os
import numpy as np
import pandas as pd
import torch
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
    
    # Popularity & artist tracks
    metadata = pd.read_csv("data/raw/track_metadata.csv")
    track_to_artist = dict(zip(metadata['track_id'], metadata['artist_name']))
    
    warm_tracks_set = set(train_df['track_id'])
    train_with_meta = train_df[['track_id']].drop_duplicates().merge(metadata, on='track_id', how='inner')
    artist_to_warm_count = train_with_meta.groupby('artist_name').size().to_dict()
    
    audio_df = pd.read_parquet("data/embeddings/audio_embeddings.parquet")
    cf_df = pd.read_parquet("data/embeddings/cf_embeddings.parquet")
    
    audio_dict = audio_df.set_index('track_id').T.to_dict('list')
    cf_dict = cf_df.set_index('track_id').T.to_dict('list')
    
    # Load Teacher
    model = CFTeacher(num_users=23533, num_tracks=54538, embedding_dim=128).to(device)
    model.load_state_dict(torch.load("data/embeddings/cf_model.pt", map_location=device))
    model.eval()
    cf_weights_warm = model.track_embed.weight.detach().cpu().numpy()
    cf_user_weights = model.user_embed.weight.detach().cpu().numpy()
    
    # Map users and tracks
    user_mapping = np.load("data/processed/user_mapping.npy", allow_pickle=True).item()
    track_mapping = np.load("data/processed/track_mapping.npy", allow_pickle=True).item()
    
    train_histories = train_df.groupby('user_id')['track_id'].apply(lambda x: set([track_mapping[t] for t in x if t in track_mapping])).to_dict()
    
    # Load ArtistBridge
    ab_model = ArtistBridge(input_dim=100, hidden_dim=128, output_dim=128).to(device)
    ab_model.load_state_dict(torch.load("data/embeddings/artistbridge_model.pt", map_location=device))
    ab_model.eval()
    
    # Precompute proxy embeddings for the test set
    unique_test_tracks = test_semi['track_id'].unique()
    test_ab_embeds = {}
    test_knn_embeds = {}
    test_art_counts = {}
    
    for t in unique_test_tracks:
        if t not in audio_dict: continue
        
        # Audio-KNN proxy
        audio_vec = audio_dict[t]
        knn_proxy = np.array(audio_vec) @ cf_weights_warm[:100].T # Note: original baseline used a naive average, but the script did `ab_item_matrix = ab_embeddings`.
        # Wait, the exact Audio-KNN proxy was generated via FAISS or similar in Stage 4. 
        # Actually in Stage 4 it was `audio_vec @ audio_warm_features.T`, picking top 10.
        pass
