"""
GNN-based Graph Coloring solver.

Each node outputs a probability distribution over C possible colors.
Loss = constraint violations (adjacent same-color) + number of colors used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.data import Data


class GINColoring(nn.Module):
    """GIN that outputs per-node color probabilities."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 max_colors: int = 20, n_layers: int = 5):
        super().__init__()
        self.max_colors = max_colors
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

        self.head = nn.Linear(hidden_dim, max_colors)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        logits = self.head(x)
        return logits  # (n, max_colors)


def coloring_loss(logits: torch.Tensor, edge_index: torch.Tensor,
                  penalty_weight: float = 10.0) -> torch.Tensor:
    """
    Loss for graph coloring:
    1. Constraint penalty: adjacent nodes should have different colors
    2. Color count: minimize number of distinct colors used
    """
    probs = F.softmax(logits, dim=-1)
    src, dst = edge_index[0], edge_index[1]

    # Penalty: probability that adjacent nodes share a color
    conflict = (probs[src] * probs[dst]).sum(dim=-1)
    constraint_loss = conflict.mean()

    # Encourage using fewer colors (entropy-like)
    avg_probs = probs.mean(dim=0)
    color_usage = -(avg_probs * (avg_probs + 1e-8).log()).sum()

    return penalty_weight * constraint_loss - 0.1 * color_usage
