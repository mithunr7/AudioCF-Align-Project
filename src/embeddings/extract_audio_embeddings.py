import os
import argparse
import bz2
import pandas as pd
import numpy as np
import subprocess

def main():
    parser = argparse.ArgumentParser()
    # We no longer need --mock_audio because we strictly use real i-vectors
    args = parser.parse_args()
    
    os.makedirs("data/embeddings", exist_ok=True)
    out_path = "data/embeddings/audio_embeddings.parquet"
    
    url = "https://zenodo.org/api/records/6609677/files/id_ivec256.tsv.bz2/content"
    bz2_path = "data/embeddings/id_ivec256.tsv.bz2"
    print(f"Downloading real acoustic embeddings (i-vectors) from {url} (resumable)...")
    
    try:
        subprocess.run(["curl", "-L", "-C", "-", "-o", bz2_path, url], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Curl failed: {e}. Retrying without resume just in case...")
        subprocess.run(["curl", "-L", "-o", bz2_path, url], check=True)
    
    print("Decompressing and parsing...")
    # Read in chunks or pandas can read compressed csv directly if installed
    df = pd.read_csv(bz2_path, sep='\t')
    
    os.remove(bz2_path)
    
    # Typically Zenodo TSVs have 'id' as the first column. We rename it to 'track_id'
    if 'id' in df.columns:
        df = df.rename(columns={'id': 'track_id'})
    
    print(f"Loaded {len(df)} real audio embeddings. Dimensionality: {len(df.columns) - 1}")
    
    df.to_parquet(out_path, index=False)
    print(f"Saved real audio embeddings to {out_path}")

if __name__ == "__main__":
    main()
