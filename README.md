# NetRepro

NetRepro is an unsupervised graph learning framework for cross-domain drug repurposing using gene co-expression networks from tissues and cell lines.

This repository is organized as a release-ready companion codebase for the manuscript. It includes:

- the main NetRepro backbone for shared and domain-specific graph representation learning
- perturbation reversion scoring for drug prioritization
- the chemistry-based zero-shot extension adapted from the original research prototype
- runnable CLI scripts for training, scoring, DMSO centroid export, demo-data generation, and chemistry fine-tuning

## License

This project is released under the [MIT License](LICENSE).

## Installation

Create a Python environment with the dependencies in `requirements.txt`.

```bash
pip install -r requirements.txt
```

Notes:

- `PyWGCNA` is optional and is only required when `--graph-method wgcna` is used.
- The chemistry extension expects precomputed molecular fingerprints. RDKit is not required at runtime if fingerprints are already prepared.

## Input matrices

Each expression table should be a CSV file with gene identifiers in the first column.

Supported layouts:

- `genes x samples`
- `samples x genes`

All scripts default to `--matrix-layout auto` and attempt to infer orientation from labels and matrix shape. If your files are unusual, pass one of these explicitly:

- `--matrix-layout genes_by_samples`
- `--matrix-layout samples_by_genes`

## Repository structure

- `netrepro.model`, `netrepro.losses`, `netrepro.train`, and `netrepro.scoring`: main NetRepro training and ranking workflow
- `netrepro.chem`: chemistry-based zero-shot extension
- `netrepro.io` and `netrepro.datasets`: transcriptomic matrix loading and treated-compound dataset utilities
- `scripts/train_netrepro.py`: train the backbone model
- `scripts/score_netrepro.py`: compute the NetRepro reversion score
- `scripts/export_dmso_centroid.py`: export DMSO node centroids for the chemistry extension
- `scripts/finetune_chem_netrepro.py`: fine-tune the chemistry extension
- `scripts/make_demo_data.py`: generate a small synthetic demo dataset

## Quick demo

Users without access to the original data can still verify that the pipeline runs end to end by generating a small synthetic dataset:

```bash
python scripts/make_demo_data.py --output-dir demo_data
```

Then train a small demo model:

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

Train the main NetRepro backbone on transcriptomic matrices from:

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

After training, the manuscript's core prioritization logic is exposed through `scripts/score_netrepro.py` and `netrepro.compute_reversion_score(...)`.

This computes:

- disease perturbation = `cancer - healthy`
- drug perturbation = `treated - dmso`
- `reversion_score = -cosine_similarity(disease_perturbation, drug_perturbation)`

Higher `reversion_score` means stronger predicted reversal.

## Zero-shot chemistry workflow

1. Train the backbone model.
2. Export DMSO node centroids:

```bash
python scripts/export_dmso_centroid.py \
  --checkpoint checkpoints/netrepro_best.pt \
  --cell-dmso data/cell_dmso.csv \
  --output-dir outputs/dmso_centroids
```

3. Fine-tune the chemistry extension from treated transcriptomic profiles and a fingerprint table:

```bash
python scripts/finetune_chem_netrepro.py \
  --checkpoint checkpoints/netrepro_best.pt \
  --treated-dir data/treated_compounds \
  --fingerprints data/compound_fingerprints.csv \
  --dmso-shared-centroid outputs/dmso_centroids/dmso_shared_node_centroid.csv \
  --dmso-specific-centroid outputs/dmso_centroids/dmso_specific_node_centroid.csv \
  --save-path checkpoints/netrepro_chem_best.pt
```

## About demo data vs real data

Without the original study data, users cannot reproduce the manuscript's biological results directly.

However, a demo dataset is still useful because it lets users:

- verify installation
- test CLI usage
- understand required file formats
- confirm that training, scoring, and centroid export work before switching to real data

That is why this release includes a demo-data generator rather than bundling private or large real datasets.

## Implementation notes

This repository provides the manuscript-oriented implementation of NetRepro, including the backbone model, perturbation reversion scoring, and the chemistry-based zero-shot extension.
