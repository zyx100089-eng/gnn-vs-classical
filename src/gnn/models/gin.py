"""
Graph Isomorphism Network (GIN) for Max-Cut.

GIN (Xu et al., 2019) is as expressive as the 1-WL test — it can distinguish
any two graphs that the Weisfeiler-Leman test can. Higher-order GNNs (e.g.
k-GNN) are strictly more expressive, but GIN is the most expressive
*standard* message-passing GNN.

For Max-Cut, each node outputs a probability p_v in [0, 1] representing
which side of the partition it belongs to. The loss maximizes the expected
cut without needing ground-truth labels (unsupervised).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GraphNorm
from torch_geometric.data import Data


class GINMaxCut(nn.Module):
    """GIN model that outputs per-node partition probabilities.

    The per-node features supplied by the caller must break the symmetry
    between nodes (the trainer uses Laplacian positional encodings): with
    constant features the loss has an exact critical point at the
    all-0.5 output — the random-cut baseline — and a deterministic
    message-passing net initialised there stays there.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 128,
                 n_layers: int = 5, dropout: float = 0.0,
                 logit_init_std: float = 1.0):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            # GraphNorm (per-graph statistics) instead of BatchNorm:
            # BatchNorm couples graphs within a batch and applies running
            # statistics (fitted on the training distribution) to every
            # test graph. LayerNorm was tried and discarded: it rescales
            # each node's feature vector by its own norm, which here
            # divides out exactly the per-node signal the cut objective
            # needs (measured 100x across-node collapse).
            self.norms.append(GraphNorm(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.dropout = dropout
        # The max-cut loss is quadratic around the uniform output p=0.5
        # (its exact gradient there is zero), so a default near-zero head
        # init leaves the model on that saddle forever. Seeding the final
        # layer with O(1) logits gives the initial per-node asymmetry the
        # gradient dynamics need to flow away from the random baseline.
        if logit_init_std > 0:
            nn.init.normal_(self.head[-1].weight, std=logit_init_std)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data with x (node features) and edge_index

        Returns:
            p: (N,) tensor of partition probabilities in [0, 1]
        """
        x, edge_index = data.x, data.edge_index
        batch_vec = getattr(data, "batch", None)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x, batch_vec)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.head(x).squeeze(-1)
        p = torch.sigmoid(logits)
        return p


def maxcut_loss(p: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
    """
    Unsupervised Max-Cut loss: MINIMIZE the negative expected cut,
    NORMALISED by the number of (directed) edge slots so the value reads
    directly as the fraction of edges cut and does not scale with batch
    size / graph size.

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
    n_undirected = edge_weight.numel() / 2

    return -expected_cut / n_undirected
