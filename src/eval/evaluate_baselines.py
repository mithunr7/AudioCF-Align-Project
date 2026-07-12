import os
import argparse
import pandas as pd
import numpy as np
import torch
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import sys
import torch.nn as nn

sys.path.append('.')
from src.models.cf_teacher import CFTeacher
from src.models.artistbridge import ArtistBridge

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1) if n > 1 else 0
    return m, h

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_neighbors", type=int, default=5)
    args = parser.parse_args()

    train_df = pd.read_parquet("data/processed/train.parquet")
    test_semi = pd.read_parquet("data/processed/test_semi_cold.parquet")
    test_fully = pd.read_parquet("data/processed/test_fully_cold.parquet")
    
    audio_df = pd.read_parquet("data/embeddings/audio_embeddings.parquet")
    cf_df = pd.read_parquet("data/embeddings/cf_embeddings.parquet")
    metadata = pd.read_csv("data/raw/track_metadata.csv")
    
    with open("data/embeddings/user2idx.json", "r") as f:
        user2idx = json.load(f)
    with open("data/embeddings/track2idx.json", "r") as f:
        track2idx_warm = json.load(f)
        
    num_users = len(user2idx)
    num_warm_tracks = len(track2idx_warm)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CFTeacher(num_users, num_warm_tracks, embed_dim=128).to(device)
    model.load_state_dict(torch.load("data/embeddings/cf_model.pt", map_location=device))
    model.eval()
    
    # Precompute track popularity
    track_popularity = train_df['track_id'].value_counts().to_dict()
    cf_weights_warm = model.track_embed.weight.detach().cpu().numpy()
    
    # --- CHECK 1: NORM VS POPULARITY CORRELATION ---
    print("\n--- Verifying BPR Popularity/Norm Confound ---")
    warm_track_norms = np.linalg.norm(cf_weights_warm, axis=1)
    warm_track_pops = np.zeros(num_warm_tracks)
    for t_idx, count in track_popularity.items():
        if t_idx in track2idx_warm:
            warm_track_pops[track2idx_warm[t_idx]] = count
    
    pearson_corr, p_value = stats.pearsonr(warm_track_norms, warm_track_pops)
    print(f"Pearson Correlation between Warm CF Norm and Popularity: {pearson_corr:.4f} (p={p_value:.4e})\n")
    
    warm_tracks_set = set(train_df['track_id'])
    audio_warm = audio_df[audio_df['track_id'].isin(warm_tracks_set)]
    audio_warm_ids = audio_warm['track_id'].values
    audio_warm_features = audio_warm.drop(columns=['track_id']).values
    cf_dict = cf_df.set_index('track_id').T.to_dict('list')
    
    def get_audio_knn_cf(target_track_id):
        target_audio = audio_df[audio_df['track_id'] == target_track_id]
        if target_audio.empty: return np.zeros(128)
        target_audio = target_audio.drop(columns=['track_id']).values
        sims = cosine_similarity(target_audio, audio_warm_features)[0]
        top_k_idx = np.argsort(sims)[-args.k_neighbors:]
        knn_cf_vectors = [cf_dict[audio_warm_ids[idx]] for idx in top_k_idx if audio_warm_ids[idx] in cf_dict]
        if not knn_cf_vectors: return np.zeros(128)
        return np.mean(knn_cf_vectors, axis=0)

    train_with_meta = train_df[['track_id']].drop_duplicates().merge(metadata, on='track_id', how='inner')
    artist_to_warm_tracks = train_with_meta.groupby('artist_name')['track_id'].apply(list).to_dict()
    
    artist_cf_catalog = {}
    for artist, tracks in artist_to_warm_tracks.items():
        cfs = [cf_dict[t] for t in tracks if t in cf_dict]
        if cfs: artist_cf_catalog[artist] = np.mean(cfs, axis=0)
            
    def get_artist_attention_cf(target_track_id):
        meta_row = metadata[metadata['track_id'] == target_track_id]
        if meta_row.empty: return np.zeros(128)
        artist = meta_row.iloc[0]['artist_name']
        return artist_cf_catalog.get(artist, np.zeros(128))
        
    # Precompute train histories map (user -> set of warm track indices)
    train_histories = train_df.groupby('user_id')['track_id'].apply(lambda x: set(track2idx_warm[t] for t in x if t in track2idx_warm)).to_dict()

    track_to_artist = dict(zip(metadata['track_id'], metadata['artist_name']))
    artist_to_tracks = train_with_meta.groupby('artist_name')['track_id'].apply(list).to_dict()
    unified_track2idx = {k: v for k, v in track2idx_warm.items()}

    def evaluate_test_set(name, test_df, model, audio_df, unified_track2idx, track_popularity, num_warm_tracks, cf_weights_warm, device, track_to_artist, artist_to_tracks):
        print(f"\n--- Evaluating {name} on Unified Full-Catalog Pool ---")
        if len(test_df) == 0: return
            
        test_tracks = test_df['track_id'].unique()
        for i, t in enumerate(test_tracks):
            if t not in unified_track2idx:
                unified_track2idx[t] = num_warm_tracks + i
            
        total_pool_size = len(unified_track2idx)
        print("Generating proxy embeddings (Audio-KNN, Artist-Attn, ArtistBridge)...")
    
        # Load ArtistBridge
        ab_model = ArtistBridge(input_dim=100, hidden_dim=128, output_dim=128).to(device)
        ab_model.load_state_dict(torch.load("data/embeddings/artistbridge_model.pt", map_location=device))
        ab_model.eval()
        
        all_audio_tensor = torch.tensor(audio_df.drop(columns=['track_id']).values, dtype=torch.float32).to(device)
        with torch.no_grad():
            ab_embeddings = ab_model(all_audio_tensor).cpu().numpy()
            
        ab_item_matrix = np.zeros((total_pool_size, 128))
        for idx, t in enumerate(audio_df['track_id'].values):
            if t in unified_track2idx:
                ab_item_matrix[unified_track2idx[t]] = ab_embeddings[idx]

        knn_item_matrix = np.zeros((total_pool_size, 128))
        knn_item_matrix[:num_warm_tracks] = cf_weights_warm
        art_item_matrix = np.zeros((total_pool_size, 128))
        art_item_matrix[:num_warm_tracks] = cf_weights_warm
        
        for t in test_tracks:
            idx = unified_track2idx[t]
            if t in track2idx_warm: continue
            knn_item_matrix[idx] = get_audio_knn_cf(t)
            art_item_matrix[idx] = get_artist_attention_cf(t)
            
        gt = {}
        for _, row in test_df.iterrows():
            u = row['user_id']
            t = unified_track2idx[row['track_id']]
            if u not in gt: gt[u] = set()
            gt[u].add(t)
            
        users_to_eval = list(gt.keys())
        
        # --- VERIFY NORM SHRINKAGE ---
        if name != "Warm-Validation (CF Teacher)":
            test_t_idxs = [unified_track2idx[t] for t in test_tracks]
            
            print(f"\n--- L2 Norm Analysis ({name}) ---")
            print(f"Mean Warm CF Norm:    {np.mean(warm_track_norms):.4f} ± {np.std(warm_track_norms):.4f}")
    
            knn_track_norms = np.linalg.norm(knn_item_matrix[test_t_idxs], axis=1)
            print(f"Mean Audio-KNN Norm:  {np.mean(knn_track_norms):.4f} ± {np.std(knn_track_norms):.4f}")
            
            art_track_norms = np.linalg.norm(art_item_matrix[test_t_idxs], axis=1)
            print(f"Mean Artist-Attn Norm:{np.mean(art_track_norms):.4f} ± {np.std(art_track_norms):.4f}")

            ab_track_norms = np.linalg.norm(ab_item_matrix[test_t_idxs], axis=1)
            print(f"Mean ArtistBridge Norm:{np.mean(ab_track_norms):.4f} ± {np.std(ab_track_norms):.4f}\n")
            
        pop_scores_global = np.zeros(total_pool_size)
        for t, count in track_popularity.items():
            if t in unified_track2idx: pop_scores_global[unified_track2idx[t]] = count
                
        pop_rec, knn_rec, art_rec, rnd_rec = [], [], [], []
        knn_cos_rec, art_cos_rec = [], []
        ab_rec, ab_cos_rec = [], []
        
        # For paired analysis
        raw_knn_cos = []
        raw_ab_cos = []
        catalog_diffs = []
        
        np.random.seed(42)
        
        batch_size = 512
        for i in tqdm(range(0, len(users_to_eval), batch_size), desc=f"Scoring {name}"):
            batch_users = users_to_eval[i:i+batch_size]
            batch_user_idxs = [user2idx.get(str(u), 0) for u in batch_users]
            
            with torch.no_grad():
                u_embeds = model.user_embed(torch.tensor(batch_user_idxs, dtype=torch.long).to(device)).cpu().numpy()
                
            for b_idx, u in enumerate(batch_users):
                u_emb = u_embeds[b_idx]
                history = train_histories.get(u, set())
                true_tracks = gt[u]
                
                u_pop_rec, u_knn_rec, u_art_rec, u_rnd_rec = [], [], [], []
                u_knn_cos_rec, u_art_cos_rec = [], []
                u_ab_rec, u_ab_cos_rec = [], []
                
                for true_t in true_tracks:
                    negs = []
                    while len(negs) < 99:
                        cand = np.random.randint(0, num_warm_tracks)
                        if cand not in history and cand != true_t:
                            negs.append(cand)
                    
                    pool = [true_t] + negs
                    
                    # Popularity
                    pop_s = [pop_scores_global[t] for t in pool]
                    pop_rank = np.argsort(pop_s)[::-1]
                    pop_top10 = [pool[i] for i in pop_rank[:10]]
                    u_pop_rec.append(1.0 if true_t in pop_top10 else 0.0)
                    
                    # KNN
                    knn_s = [np.dot(u_emb, knn_item_matrix[t]) for t in pool]
                    knn_rank = np.argsort(knn_s)[::-1]
                    knn_top10 = [pool[i] for i in knn_rank[:10]]
                    u_knn_rec.append(1.0 if true_t in knn_top10 else 0.0)
                    
                    # Artist
                    art_s = [np.dot(u_emb, art_item_matrix[t]) for t in pool]
                    art_rank = np.argsort(art_s)[::-1]
                    art_top10 = [pool[i] for i in art_rank[:10]]
                    u_art_rec.append(1.0 if true_t in art_top10 else 0.0)
                    
                    # Cosine KNN
                    u_emb_norm = u_emb / (np.linalg.norm(u_emb) + 1e-9)
                    knn_pool_norms = [knn_item_matrix[t] / (np.linalg.norm(knn_item_matrix[t]) + 1e-9) for t in pool]
                    knn_cos_s = [np.dot(u_emb_norm, v) for v in knn_pool_norms]
                    knn_cos_rank = np.argsort(knn_cos_s)[::-1]
                    knn_cos_top10 = [pool[i] for i in knn_cos_rank[:10]]
                    u_knn_cos_rec.append(1.0 if true_t in knn_cos_top10 else 0.0)
                    
                    # Cosine Artist
                    art_pool_norms = [art_item_matrix[t] / (np.linalg.norm(art_item_matrix[t]) + 1e-9) for t in pool]
                    art_cos_s = [np.dot(u_emb_norm, v) for v in art_pool_norms]
                    art_cos_rank = np.argsort(art_cos_s)[::-1]
                    art_cos_top10 = [pool[i] for i in art_cos_rank[:10]]
                    u_art_cos_rec.append(1.0 if true_t in art_cos_top10 else 0.0)
                    
                    # ArtistBridge (Dot)
                    ab_s = [np.dot(u_emb, ab_item_matrix[t]) for t in pool]
                    ab_rank = np.argsort(ab_s)[::-1]
                    ab_top10 = [pool[i] for i in ab_rank[:10]]
                    u_ab_rec.append(1.0 if true_t in ab_top10 else 0.0)
                    
                    # ArtistBridge (Cosine)
                    ab_pool_norms = [ab_item_matrix[t] / (np.linalg.norm(ab_item_matrix[t]) + 1e-9) for t in pool]
                    ab_cos_s = [np.dot(u_emb_norm, v) for v in ab_pool_norms]
                    ab_cos_rank = np.argsort(ab_cos_s)[::-1]
                    ab_cos_top10 = [pool[i] for i in ab_cos_rank[:10]]
                    u_ab_cos_rec.append(1.0 if true_t in ab_cos_top10 else 0.0)
                    
                    # Random
                    rnd_s = [np.random.rand() for _ in pool]
                    rnd_rank = np.argsort(rnd_s)[::-1]
                    rnd_top10 = [pool[i] for i in rnd_rank[:10]]
                    u_rnd_rec.append(1.0 if true_t in rnd_top10 else 0.0)
                    
                    # Store raw hits for paired test
                    hit_knn = u_knn_cos_rec[-1]
                    hit_ab = u_ab_cos_rec[-1]
                    raw_knn_cos.append(hit_knn)
                    raw_ab_cos.append(hit_ab)
                    
                    # Store catalog size correlation
                    orig_track_id = test_tracks[true_t - num_warm_tracks]
                    artist = track_to_artist.get(orig_track_id, "")
                    cat_size = len(artist_to_tracks.get(artist, []))
                    catalog_diffs.append((cat_size, hit_ab - hit_knn))
                    
                pop_rec.append(np.mean(u_pop_rec))
                knn_rec.append(np.mean(u_knn_rec))
                art_rec.append(np.mean(u_art_rec))
                knn_cos_rec.append(np.mean(u_knn_cos_rec))
                art_cos_rec.append(np.mean(u_art_cos_rec))
                ab_rec.append(np.mean(u_ab_rec))
                ab_cos_rec.append(np.mean(u_ab_cos_rec))
                rnd_rec.append(np.mean(u_rnd_rec))
                
        p_m, p_h = mean_confidence_interval(pop_rec)
        k_m, k_h = mean_confidence_interval(knn_rec)
        a_m, a_h = mean_confidence_interval(art_rec)
        kc_m, kc_h = mean_confidence_interval(knn_cos_rec)
        ac_m, ac_h = mean_confidence_interval(art_cos_rec)
        ab_m, ab_h = mean_confidence_interval(ab_rec)
        abc_m, abc_h = mean_confidence_interval(ab_cos_rec)
        r_m, r_h = mean_confidence_interval(rnd_rec)
        
        print(f"[{name}] Random      -> Recall@10: {r_m:.4f} ± {r_h:.4f}")
        print(f"[{name}] Popularity  -> Recall@10: {p_m:.4f} ± {p_h:.4f}")
        print(f"[{name}] Audio-KNN (Dot)  -> Recall@10: {k_m:.4f} ± {k_h:.4f}")
        print(f"[{name}] Artist-Attn (Dot)-> Recall@10: {a_m:.4f} ± {a_h:.4f}")
        print(f"[{name}] ArtistBridge (Dot)-> Recall@10: {ab_m:.4f} ± {ab_h:.4f}")
        print(f"[{name}] Audio-KNN (Cos)  -> Recall@10: {kc_m:.4f} ± {kc_h:.4f}")
        print(f"[{name}] Artist-Attn (Cos)-> Recall@10: {ac_m:.4f} ± {ac_h:.4f}")
        print(f"[{name}] ArtistBridge (Cos)-> Recall@10: {abc_m:.4f} ± {abc_h:.4f}")
        
        # Paired Test
        t_stat, p_val = stats.ttest_rel(raw_ab_cos, raw_knn_cos)
        print(f"[{name}] Paired t-test (ArtistBridge vs Audio-KNN Cosine): t={t_stat:.4f}, p={p_val:.4e}")
        
        # Catalog Size Correlation
        if len(catalog_diffs) > 0:
            cat_sizes = [x[0] for x in catalog_diffs]
            diffs = [x[1] for x in catalog_diffs]
            p_corr, p_pval = stats.pearsonr(cat_sizes, diffs)
            print(f"[{name}] Pearson Correlation (Catalog Size vs AB Improvement): r={p_corr:.4f}, p={p_pval:.4e}\n")
        
    evaluate_test_set("Semi-Cold", test_semi, model, audio_df, unified_track2idx, track_popularity, num_warm_tracks, cf_weights_warm, device, track_to_artist, artist_to_tracks)
    evaluate_test_set("Fully-Cold", test_fully, model, audio_df, unified_track2idx, track_popularity, num_warm_tracks, cf_weights_warm, device, track_to_artist, artist_to_tracks)
    
if __name__ == "__main__":
    main()
