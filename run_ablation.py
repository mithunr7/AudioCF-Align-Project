import subprocess
import os

def main():
    seeds = [1, 2, 3]
    lambdas = [0.0, 0.5]
    
    os.makedirs("data/embeddings", exist_ok=True)
    
    for s in seeds:
        for l in lambdas:
            out = f"data/embeddings/ablation_l{l}_s{s}.pt"
            print(f"--- Training lambda={l}, seed={s} ---")
            cmd = f"python src/train/train_artistbridge.py --lambda_artist {l} --seed {s} --out_model {out}"
            subprocess.run(cmd, shell=True)
            
if __name__ == "__main__":
    main()
