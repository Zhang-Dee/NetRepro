from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data

import PyWGCNA


GraphMethod = Literal["correlation", "wgcna"]


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class GraphBuildConfig:
    edge_threshold: float = 0.8
    graph_method: GraphMethod = "correlation"
    wgcna_type: Literal["unsigned", "signed"] = "unsigned"
    wgcna_power: int = 6
    scale_each_matrix: bool = False


class PairedGraphDataset(Dataset):
    """
    Dataset that samples graph pairs from two expression matrices.

    Each expression matrix is gene x sample.
    Each returned graph is built from 20 sampled columns.
    Node features are the 20-dimensional expression values for each gene.
    Graph labels:
        tissue domain -> 0
        cell-line domain -> 1
    """

    def __init__(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        domain_label: int,
        num_graphs: int = 1000,
        group_size: int = 20,
        max_repeats: Tuple[int, int] = (10, 10),
        seed: int = 1234,
        config: GraphBuildConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GraphBuildConfig()
        self.domain_label = int(domain_label)
        self.num_graphs = int(num_graphs)
        self.group_size = int(group_size)
        self.seed = int(seed)

        set_random_seed(self.seed)

        self.df_a = self._prepare_df(df_a, self.config.scale_each_matrix)
        self.df_b = self._prepare_df(df_b, self.config.scale_each_matrix)

        self.groups_a = self._sample_column_groups(self.df_a, max_repeats[0])
        self.groups_b = self._sample_column_groups(self.df_b, max_repeats[1])

    @staticmethod
    def _prepare_df(df: pd.DataFrame, scale: bool) -> pd.DataFrame:
        if not scale:
            return df.copy()
        scaler = StandardScaler()
        values = scaler.fit_transform(df.values)
        return pd.DataFrame(values, index=df.index, columns=df.columns)

    def _sample_column_groups(self, df: pd.DataFrame, max_overlap: int) -> List[List[int]]:
        total_columns = df.shape[1]
        if total_columns < self.group_size:
            raise ValueError(
                f"Expected at least {self.group_size} columns, but got {total_columns}."
            )

        groups: List[List[int]] = []
        column_ids = list(range(total_columns))

        while len(groups) < self.num_graphs:
            current = set(random.sample(column_ids, self.group_size))
            if all(len(current.intersection(prev)) <= max_overlap for prev in groups):
                groups.append(list(current))

        return groups

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: int) -> Tuple[Data, Data]:
        cols_a = self.groups_a[idx]
        cols_b = self.groups_b[idx]

        expr_a = self.df_a.iloc[:, cols_a]  # gene x 20
        expr_b = self.df_b.iloc[:, cols_b]  # gene x 20

        edge_index_a = self._build_graph(expr_a.T)
        edge_index_b = self._build_graph(expr_b.T)

        data_a = Data(
            x=torch.tensor(expr_a.values, dtype=torch.float32),
            edge_index=edge_index_a,
            domain=torch.tensor([self.domain_label], dtype=torch.long),
        )
        data_b = Data(
            x=torch.tensor(expr_b.values, dtype=torch.float32),
            edge_index=edge_index_b,
            domain=torch.tensor([self.domain_label], dtype=torch.long),
        )
        return data_a, data_b

    def _build_graph(self, sample_by_gene: pd.DataFrame) -> torch.Tensor:
        if self.config.graph_method == "wgcna":
            return self._build_wgcna_graph(sample_by_gene)
        return self._build_correlation_graph(sample_by_gene)

    def _build_correlation_graph(self, sample_by_gene: pd.DataFrame) -> torch.Tensor:
        corr = sample_by_gene.corr()
        signed_binary = (np.abs(corr.values) > self.config.edge_threshold).astype(np.float32)
        np.fill_diagonal(signed_binary, 0.0)

        graph = nx.from_numpy_array(signed_binary)
        adj = nx.to_scipy_sparse_array(graph, format="coo")
        row = torch.from_numpy(adj.row).long()
        col = torch.from_numpy(adj.col).long()
        return torch.stack([row, col], dim=0)

    def _build_wgcna_graph(self, sample_by_gene: pd.DataFrame) -> torch.Tensor:
        wgcna = PyWGCNA.WGCNA(name="netrepro_graph", geneExp=sample_by_gene)
        expr = wgcna.geneExpr.to_df().T

        if self.config.wgcna_type == "signed":
            adjacency = ((1 + np.corrcoef(expr)) / 2) ** self.config.wgcna_power
        else:
            adjacency = np.abs(np.corrcoef(expr)) ** self.config.wgcna_power

        tom = wgcna.TOMsimilarity(adjacency, TOMType=self.config.wgcna_type).values
        np.fill_diagonal(tom, 0.0)
        binary_adj = (tom >= self.config.edge_threshold).astype(np.float32)

        graph = nx.from_numpy_array(binary_adj)
        adj = nx.to_scipy_sparse_array(graph, format="coo")
        row = torch.from_numpy(adj.row).long()
        col = torch.from_numpy(adj.col).long()
        return torch.stack([row, col], dim=0)
