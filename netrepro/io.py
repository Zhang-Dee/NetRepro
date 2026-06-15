from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd


MatrixLayout = Literal["auto", "genes_by_samples", "samples_by_genes"]


@dataclass(frozen=True)
class MatrixLoadResult:
    matrix: pd.DataFrame
    layout_used: str
    source_path: str


def _gene_like_fraction(labels: list[str]) -> float:
    if not labels:
        return 0.0

    matches = 0
    for raw in labels:
        label = str(raw).strip()
        if not label:
            continue
        if label.startswith("ENSG") or label.startswith("ENSMUSG"):
            matches += 1
            continue
        if label.count("-") <= 1 and label.replace("-", "").replace(".", "").isalnum():
            alpha = sum(ch.isalpha() for ch in label)
            digits = sum(ch.isdigit() for ch in label)
            if 2 <= len(label) <= 20 and alpha >= 2 and digits <= 8:
                matches += 1
    return matches / max(1, len(labels))


def infer_matrix_layout(df: pd.DataFrame) -> str:
    """
    Infer whether a transcriptomic matrix is genes x samples or samples x genes.

    Preference order:
    1. axis labels that strongly resemble gene identifiers
    2. larger axis is assumed to be genes
    """

    index_like = _gene_like_fraction(df.index.astype(str).tolist()[:200])
    column_like = _gene_like_fraction(df.columns.astype(str).tolist()[:200])

    if index_like >= 0.5 and index_like > column_like:
        return "genes_by_samples"
    if column_like >= 0.5 and column_like > index_like:
        return "samples_by_genes"

    return "genes_by_samples" if df.shape[0] >= df.shape[1] else "samples_by_genes"


def load_expression_matrix(
    path: str | Path,
    layout: MatrixLayout = "auto",
    sep: str = ",",
) -> MatrixLoadResult:
    df = pd.read_csv(path, index_col=0, sep=sep)

    if layout == "auto":
        inferred_layout = infer_matrix_layout(df)
    else:
        inferred_layout = layout

    matrix = df if inferred_layout == "genes_by_samples" else df.T
    matrix = matrix.loc[~matrix.index.duplicated(keep="first")]
    matrix = matrix.loc[:, ~matrix.columns.duplicated(keep="first")]

    return MatrixLoadResult(
        matrix=matrix,
        layout_used=inferred_layout,
        source_path=str(path),
    )


def align_gene_space(*matrices: pd.DataFrame) -> list[pd.DataFrame]:
    if not matrices:
        return []

    common_genes = set(matrices[0].index)
    for matrix in matrices[1:]:
        common_genes &= set(matrix.index)

    common_genes = sorted(common_genes)
    if not common_genes:
        raise ValueError("The provided matrices do not share any common genes.")

    return [matrix.loc[common_genes].copy() for matrix in matrices]


def load_fingerprint_table(
    path: str | Path,
    index_col: int | str = 0,
    sep: str = ",",
) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=index_col, sep=sep)
    return table.loc[~table.index.duplicated(keep="first")].copy()
