from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .data import GraphBuildConfig, build_correlation_edge_index
from .io import load_expression_matrix


@dataclass
class TreatedCompoundDatasetConfig:
    matrix_layout: str = "auto"
    graph_config: GraphBuildConfig = field(default_factory=GraphBuildConfig)
    prefer_24h: bool = True
    group_size: int = 20
    seed: int = 1234


class TreatedCompoundDataset(Dataset):
    """
    Dataset for the chemistry-based extension.

    Each item loads one treated transcriptomic matrix and one precomputed
    fingerprint vector, then converts the treated matrix to a single graph.
    """

    def __init__(
        self,
        treated_dir: str | Path,
        fingerprint_table: pd.DataFrame,
        config: TreatedCompoundDatasetConfig | None = None,
    ) -> None:
        super().__init__()
        self.treated_dir = Path(treated_dir)
        self.fingerprint_table = fingerprint_table
        self.config = config or TreatedCompoundDatasetConfig()
        self.random = random.Random(self.config.seed)

        files = sorted(self.treated_dir.glob("*.csv"))
        available = []
        for path in files:
            compound_id = path.stem
            if compound_id in self.fingerprint_table.index:
                available.append((compound_id, path))
        if not available:
            raise ValueError(
                "No treated CSV files in the directory matched rows in the fingerprint table."
            )
        self.items = available

    def __len__(self) -> int:
        return len(self.items)

    def _choose_columns(self, matrix: pd.DataFrame) -> pd.DataFrame:
        if matrix.shape[1] == self.config.group_size:
            return matrix

        if matrix.shape[1] > self.config.group_size:
            if self.config.prefer_24h:
                h24 = [col for col in matrix.columns if "_24H_" in str(col)]
                other = [col for col in matrix.columns if col not in h24]
                if len(h24) >= self.config.group_size:
                    selected = self.random.sample(h24, self.config.group_size)
                    return matrix.loc[:, selected]
                if h24:
                    need = self.config.group_size - len(h24)
                    selected_other = self.random.sample(other, need)
                    return matrix.loc[:, h24 + selected_other]
            selected = self.random.sample(list(matrix.columns), self.config.group_size)
            return matrix.loc[:, selected]

        # Fewer than group_size samples: synthesize linear combinations, mirroring the prototype.
        output = matrix.copy()
        while output.shape[1] < self.config.group_size:
            left, right = self.random.sample(list(output.columns), 2)
            synthetic = 0.5 * output[left] + 0.5 * output[right]
            output[f"synthetic_{output.shape[1]}"] = synthetic
        return output

    def __getitem__(self, index: int) -> tuple[Data, torch.Tensor, str]:
        compound_id, path = self.items[index]
        load_result = load_expression_matrix(path, layout=self.config.matrix_layout)
        matrix = self._choose_columns(load_result.matrix)

        scaler = StandardScaler()
        sample_by_gene = scaler.fit_transform(matrix.T)
        sample_by_gene_df = pd.DataFrame(sample_by_gene, index=matrix.columns, columns=matrix.index)
        edge_index = build_correlation_edge_index(
            sample_by_gene_df,
            edge_threshold=self.config.graph_config.edge_threshold,
        )

        graph = Data(
            x=torch.tensor(sample_by_gene_df.T.values, dtype=torch.float32),
            edge_index=edge_index,
            domain=torch.tensor([1], dtype=torch.long),
        )
        fingerprint = torch.tensor(
            self.fingerprint_table.loc[compound_id].values,
            dtype=torch.float32,
        )
        return graph, fingerprint, compound_id
