"""
GNN-based TSP solver.

Uses a GIN to produce node embeddings, then scores edges by dot product
of endpoint embeddings. Constructs tour greedily from edge scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import Data
import numpy as np


class GINTsp(nn.Module):
    """GIN that produces node embeddings for TSP edge scoring."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, n_layers: int = 5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return x  # (n, hidden_dim) node embeddings


def greedy_tour_from_embeddings(embeddings: np.ndarray,
                                dist_matrix: np.ndarray) -> tuple[list, float]:
    """Construct tour greedily using embedding similarity to bias toward short edges."""
    n = len(dist_matrix)
    scores = embeddings @ embeddings.T
    combined = scores / (dist_matrix + 1e-6)

    visited = [False] * n
    tour = [0]
    visited[0] = True

    for _ in range(n - 1):
        current = tour[-1]
        best_next = -1
        best_score = -float("inf")
        for j in range(n):
            if not visited[j]:
                s = combined[current][j]
                if s > best_score:
                    best_score = s
                    best_next = j
        tour.append(best_next)
        visited[best_next] = True

    length = sum(dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
    return tour, length
