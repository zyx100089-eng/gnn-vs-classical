"""
Graph Isomorphism Network (GIN) for Max-Cut.

GIN (Xu et al., 2019) is the most expressive message-passing GNN —
it can distinguish any two graphs that the Weisfeiler-Leman test can.

For Max-Cut, each node outputs a probability p_v in [0, 1] representing
which side of the partition it belongs to. The loss maximizes the expected
cut without needing ground-truth labels (unsupervised).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data


class GINMaxCut(nn.Module):
    """GIN model that outputs per-node partition probabilities."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 n_layers: int = 5, dropout: float = 0.0):
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

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data with x (node features) and edge_index

        Returns:
            p: (N,) tensor of partition probabilities in [0, 1]
        """
        x, edge_index = data.x, data.edge_index

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.head(x).squeeze(-1)
        p = torch.sigmoid(logits)
        return p


def maxcut_loss(p: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
    """
    Unsupervised Max-Cut loss: MINIMIZE the negative expected cut.

    Expected cut = sum_{(i,j)} w_ij * [p_i(1-p_j) + (1-p_i)p_j]
                 = sum_{(i,j)} w_ij * [p_i + p_j - 2*p_i*p_j]

    We negate it because we want to maximize.
    """
    src, dst = edge_index[0], edge_index[1]
    p_src = p[src]
    p_dst = p[dst]

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=p.device)

    cut_contrib = edge_weight * (p_src + p_dst - 2 * p_src * p_dst)
    expected_cut = cut_contrib.sum() / 2  # each edge counted twice in undirected

    return -expected_cut
