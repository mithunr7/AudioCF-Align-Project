# AudioCF-Align

**Bridging the modality gap for cold-start music recommendation.**

Recommender systems can't rank a brand-new song — it has zero listening
history for a collaborative filtering (CF) model to learn from. The usual
fallback is to use raw audio content (tempo, timbre, etc.) as a proxy for
listener preference, but there's a well-documented gap between what audio
*sounds like* and what actually makes someone *prefer* it.

**AudioCF-Align** is a small neural network that closes most of that gap. It
learns to project raw acoustic embeddings directly into the same preference
space a CF model learns from real listening behavior — so a brand-new track
can be ranked meaningfully from its audio alone, with zero interaction
history required.

## Result

Trained and evaluated on **5M+ real interactions** from the
[Music4All-Onion](https://zenodo.org/records/6609677) dataset (109K tracks),
using a strict time-based cold-start split and a sampled-negative ranking
protocol with paired significance testing.

| Split | Method | Recall@10 | vs. Audio-KNN baseline |
|---|---|---|---|
| Fully-Cold (unseen track, unseen artist) | Random | 0.116 | — |
| Fully-Cold | Audio-KNN (raw content) | 0.081 | — |
| Fully-Cold | **AudioCF-Align** | **0.146** | t=3.28, **p < 0.002** |
| Semi-Cold (unseen track, known artist) | Audio-KNN (raw content) | 0.064 | — |
| Semi-Cold | **AudioCF-Align** | **~0.22–0.23** | **p < 1e-60** |
| Semi-Cold | Artist-history oracle (upper bound) | 0.317 | — |

*Fully-Cold and Semi-Cold scores are not directly comparable to each other —
see [EXPERIMENTS.md](./EXPERIMENTS.md) for why.*

## How it works

A **Teacher–Student** setup:

1. **Teacher:** a Bayesian Personalized Ranking (BPR) matrix factorization
   model trained on warm interactions, learning a 128-D embedding space that
   encodes real listener preference.
2. **Student (AudioCF-Align):** a small MLP trained to regress raw 100-D
   audio embeddings (i-vectors) toward the Teacher's 128-D CF embeddings,
   using a combined MSE + cosine loss.
3. **Inference:** any new track's audio embedding is passed through the
   trained MLP, landing directly in CF space — no listening history needed.

## Notable finding: a negative result that shaped the final design

The project originally hypothesized that anchoring training to an artist's
existing catalog (a "Leave-One-Out Artist Anchor" loss term) would be the
key mechanism for closing the gap. A multi-seed ablation with paired
significance testing disproved this — the anchor term provided no
consistent benefit across three random seeds (one seed even significantly
*favored* dropping it, p=0.006 after Bonferroni correction). The real
driver of the improvement turned out to be simpler: direct non-linear
alignment between audio and CF space, with no artist information needed at
all.

The project was renamed from "ArtistBridge" to "AudioCF-Align" to reflect
this — the name now describes the mechanism that actually works, not the
one originally hypothesized.

Full diagnostic process — including catching data leakage in a baseline,
an L2-norm/popularity confound, and a statistically underpowered evaluation
protocol along the way — is documented in
[EXPERIMENTS.md](./EXPERIMENTS.md).

## Repo structure

```
artistbridge/
├── data/               # raw, processed, and cached embeddings (gitignored)
├── src/
│   ├── data/           # download, preprocessing, time-based cold-start splits
│   ├── embeddings/     # audio embedding extraction
│   ├── models/         # CF teacher, AudioCF-Align MLP, baselines
│   ├── train/           # training loops
│   ├── eval/            # metrics, sampled-negative evaluation, significance tests
│   └── utils/
├── notebooks/
├── results/            # logs, tables, plots
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

python src/data/download.py          # streams + samples the Music4All-Onion interaction log
python src/data/preprocess.py        # time-based cold-start split
python src/embeddings/extract_audio_embeddings.py
python src/train/train_cf_teacher.py
python src/train/train_baselines.py
python src/train/train_artistbridge.py
python src/eval/evaluate.py          # final results table
```

## Tech stack

PyTorch · Pandas · NumPy · SciPy · BPR-based Collaborative Filtering

## Future work

The Semi-Cold result (~0.22–0.23) still trails the artist-history oracle
(0.317). The negative ablation result doesn't mean artist information is
useless — only that using it as a training-time loss anchor wasn't the
right mechanism. Feeding artist identity as an explicit model *input* (e.g.
a learned artist embedding concatenated with the audio vector) is a
meaningfully different approach and the most promising direction for
closing the remaining gap.

## Data

[Music4All-Onion](https://zenodo.org/records/6609677) (Zenodo, Santana et
al.) — interactions and precomputed audio features. Track–artist metadata
via [seungheondoh/enrich-music4all](https://huggingface.co/datasets/seungheondoh/enrich-music4all).
