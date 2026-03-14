# NetRepro

NetRepro is an unsupervised graph learning framework for cross-domain drug repurposing using gene co-expression networks from tissues and cell lines.

## Features

- Shared and domain-specific graph encoders
- Adversarial domain adaptation
- Triplet loss for biologically structured alignment
- Graph reconstruction regularization using positive and negative sampled edges
- Support for correlation-based or WGCNA-based graph construction

## Input format

Each expression table should be a CSV file with:

- rows = genes
- columns = samples
- first column = gene identifiers

## Training example

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
