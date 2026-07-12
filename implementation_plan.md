# Stage 5: ArtistBridge Projection Network

Now that we have established a rigorously stable, statistically powered baseline using the 100-item Sampled-Negative protocol—and conclusively proven that naive proxies perform **worse than random chance** due to structural norm penalties in dot-product spaces—we are ready to build the core ArtistBridge neural network.

## Goal

Train a neural projection network that explicitly maps raw 100-D audio embeddings (i-vectors) into the 128-D Collaborative Filtering (CF) manifold learned by the Teacher model in Stage 3. 

The network must learn to bridge the modality gap so that the projected audio embeddings can survive in a global ranking scenario, natively adopting the scale (norm) and topological structure of the BPR-optimized warm items.

## Proposed Architecture

As established in the revised constraints, the input audio embeddings are 100-dimensional i-vectors, and the target CF space is 128-dimensional. 

### Model: `ArtistBridge`
- **Input:** 100-D (Audio i-vector)
- **Hidden Layer:** Linear (100 → 128) + ReLU + Dropout(0.1)
- **Output Layer:** Linear (128 → 128)
*(No final activation, allowing it to span the continuous, unconstrained BPR latent space)*

### Training Objective (Knowledge Distillation)
The objective is to regress the output of the network against the frozen 128-D embeddings from the CF Teacher model. 
- **Primary Loss (MSE):** Mean Squared Error between the projected audio embedding and the true BPR CF embedding.
- **Directional Loss (Cosine):** Cosine embedding loss to ensure the angular ranking properties of the BPR space are preserved, preventing the predicted vectors from collapsing to the origin.
- **Artist Anchoring Regularizer (`lambda`):** We will add an optional regularization term: `MSE(pred, artist_cf_proxy)`. This encourages the network to cluster tracks by the same artist in the latent space if the acoustic evidence is ambiguous. 
  - **CRITICAL CONSTRAINT:** The `artist_cf_proxy` for a training track MUST be computed using **Leave-One-Out (LOO)** averaging (averaging all *other* tracks by the artist, explicitly excluding the current training track's true CF embedding). Without LOO, the network could trivially cheat by regressing toward its own true label. We will track `lambda` as a hyperparameter to ablate its effect.

## Proposed Changes

### [NEW] [src/models/artistbridge.py](file:///d:/Audio/artistbridge/src/models/artistbridge.py)
- Define the PyTorch `ArtistBridge` MLP module (100 → 128 → 128).

### [NEW] [src/train/train_artistbridge.py](file:///d:/Audio/artistbridge/src/train/train_artistbridge.py)
- Load the `audio_embeddings.parquet` and `cf_embeddings.parquet` (Teacher targets).
- Create a PyTorch Dataset yielding `(audio_vector, true_cf_vector, artist_cf_vector)`.
- Train the MLP using AdamW, logging validation MSE/Cosine loss on a holdout set of warm tracks.
- Save the trained projection model to `data/embeddings/artistbridge_model.pt`.

## Open Questions for Review

1. **Loss Function:** Is the proposed `MSE + Cosine + Artist Anchor` loss acceptable, or would you prefer a pure BPR contrastive loss for the projection network (where we sample triplets and train the projection to maximize dot products directly)?
2. **Data Split:** I will train the projection network exclusively on the `train` split of warm tracks, and validate its MSE on the `val_warm` tracks.

Once approved, I will implement the network, train it, and then proceed to Stage 6 (Final Evaluation) to see if ArtistBridge beats the `0.0314` / `0.3019` baselines we just established!
