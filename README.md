# NetRepro

NetRepro is a graph-based framework for cross-domain drug repurposing from transcriptomic data. It learns shared network representations across tissue and cell-line domains, then prioritizes compounds by how strongly their perturbation vectors reverse the disease state.

This repository contains the manuscript companion implementation of:

- the NetRepro backbone model
- perturbation reversion scoring
- the chemistry-based zero-shot extension

## License

This project is released under the [MIT License](LICENSE).

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Notes:

- `PyWGCNA` is optional and is only required when `--graph-method wgcna` is used.
- The chemistry extension expects precomputed molecular fingerprints.

## Input format

Expression matrices should be stored as CSV files with gene identifiers in the first column.

Supported layouts:

- `genes x samples`
- `samples x genes`

All CLI scripts support:

- `--matrix-layout auto`
- `--matrix-layout genes_by_samples`
- `--matrix-layout samples_by_genes`

## Repository structure

- `netrepro.model`, `netrepro.losses`, `netrepro.train`, `netrepro.scoring`: backbone training and scoring
- `netrepro.chem`: chemistry-based zero-shot extension
- `netrepro.io`, `netrepro.datasets`: matrix loading and treated-compound dataset utilities
- `scripts/train_netrepro.py`: backbone training
- `scripts/score_netrepro.py`: reversion score calculation
- `scripts/export_dmso_centroid.py`: DMSO centroid export for chemistry fine-tuning
- `scripts/finetune_chem_netrepro.py`: chemistry extension fine-tuning
- `scripts/make_demo_data.py`: synthetic quick-start dataset generation

## Quick start

Generate a small synthetic dataset:

```bash
python scripts/make_demo_data.py --output-dir demo_data
```

Train a small model:

```bash
python scripts/train_netrepro.py \
  --tissue-normal-train demo_data/tissue_normal_train.csv \
  --tissue-cancer-train demo_data/tissue_cancer_train.csv \
  --cell-dmso-train demo_data/cell_dmso_train.csv \
  --cell-treated-train demo_data/cell_treated_train.csv \
  --tissue-normal-val demo_data/tissue_normal_val.csv \
  --tissue-cancer-val demo_data/tissue_cancer_val.csv \
  --cell-dmso-val demo_data/cell_dmso_val.csv \
  --cell-treated-val demo_data/cell_treated_val.csv \
  --matrix-layout samples_by_genes \
  --epochs 1 \
  --batch-size 2 \
  --num-graphs-train 4 \
  --num-graphs-val 2 \
  --max-repeats 18 \
  --edge-threshold 0.3 \
  --save-path checkpoints/demo_netrepro.pt
```

Score a treated state:

```bash
python scripts/score_netrepro.py \
  --checkpoint checkpoints/demo_netrepro.pt \
  --tissue-normal demo_data/tissue_normal_val.csv \
  --tissue-cancer demo_data/tissue_cancer_val.csv \
  --cell-dmso demo_data/cell_dmso_val.csv \
  --cell-treated demo_data/cell_treated_val.csv \
  --matrix-layout samples_by_genes \
  --num-graphs 4 \
  --max-repeats 18 \
  --edge-threshold 0.3
```

## Backbone training

Train the main NetRepro model from:

- healthy tissue
- cancer tissue
- untreated cell line or DMSO
- treated cell line

Example:

```bash
python scripts/train_netrepro.py \
  --tissue-normal-train data/tissue_normal_train.csv \
  --tissue-cancer-train data/tissue_cancer_train.csv \
  --cell-dmso-train data/cell_dmso_train.csv \
  --cell-treated-train data/cell_treated_train.csv \
  --tissue-normal-val data/tissue_normal_val.csv \
  --tissue-cancer-val data/tissue_cancer_val.csv \
  --cell-dmso-val data/cell_dmso_val.csv \
  --cell-treated-val data/cell_treated_val.csv \
  --graph-method correlation \
  --edge-threshold 0.8 \
  --epochs 50 \
  --batch-size 8 \
  --save-path checkpoints/netrepro_best.pt
```

## Reversion score

After training, NetRepro prioritizes compounds from the relationship between:

- disease perturbation = `cancer - healthy`
- drug perturbation = `treated - dmso`

The provided score is:

- `reversion_score = -cosine_similarity(disease_perturbation, drug_perturbation)`

Higher `reversion_score` indicates stronger predicted reversal.

## Zero-shot chemistry workflow

1. Train the backbone model.
2. Export DMSO node centroids:

```bash
python scripts/export_dmso_centroid.py \
  --checkpoint checkpoints/netrepro_best.pt \
  --cell-dmso data/cell_dmso.csv \
  --output-dir outputs/dmso_centroids
```

3. Fine-tune the chemistry extension:

```bash
python scripts/finetune_chem_netrepro.py \
  --checkpoint checkpoints/netrepro_best.pt \
  --treated-dir data/treated_compounds \
  --fingerprints data/compound_fingerprints.csv \
  --dmso-shared-centroid outputs/dmso_centroids/dmso_shared_node_centroid.csv \
  --dmso-specific-centroid outputs/dmso_centroids/dmso_specific_node_centroid.csv \
  --save-path checkpoints/netrepro_chem_best.pt
```
