"""Diagnostics for the Max-Cut GNN's symmetry/collapse behaviour.

Reproduces the mechanism claims made in the README and paper:

  1. With constant node features the unsupervised cut loss has an exact
     critical point at the uniform output p=0.5, whose value is exactly
     -|E|/2 (the random-cut baseline), with zero gradient.
  2. A GIN with constant features collapses to near-constant outputs, so
     thresholding it produces a cut close to (or below) a coin flip.
  3. With Laplacian positional-encoding features the across-node signal
     survives message passing under GraphNorm but is destroyed by
     LayerNorm (which rescales each node's vector by its own norm).

Run: python3 experiments/run_gnn_symmetry_diagnostics.py
"""

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GraphNorm

from src.graphs.generators import generate_batch, generate_instance
from src.gnn.training.maxcut_trainer import nx_to_pyg
from src.gnn.models.gin import GINMaxCut, maxcut_loss


class GINVariant(nn.Module):
    """GIN identical to GINMaxCut but with a selectable norm layer."""

    def __init__(self, input_dim, hidden_dim=128, n_layers=5, norm="graphnorm"):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norm_kind = norm
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp, train_eps=True))
            if norm == "graphnorm":
                self.norms.append(GraphNorm(hidden_dim))
            elif norm == "batchnorm":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif norm == "layernorm":
                self.norms.append(nn.LayerNorm(hidden_dim))
            else:
                self.norms.append(nn.Identity())
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                  nn.ReLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, data, return_layer_std=False):
        x, edge_index = data.x, data.edge_index
        batch_vec = getattr(data, "batch", None)
        stds = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x, batch_vec) if self.norm_kind == "graphnorm" else norm(x)
            x = F.relu(x)
            stds.append(x.std(axis=0).mean().item())
        p = torch.sigmoid(self.head(x).squeeze(-1))
        return (p, stds) if return_layer_std else p


def cut_fraction(G, S):
    return sum(1 for u, v in G.edges() if (u in S) != (v in S)) / G.number_of_edges()


def main():
    G = generate_instance("erdos_renyi", 100, seed=2042)
    m = G.number_of_edges()

    print("1) Saddle check (constant-feature objective)")
    p_half = torch.full((100,), 0.5, requires_grad=True)
    data = nx_to_pyg(G, feature_seed=0, posenc="random")
    loss = maxcut_loss(p_half, data.edge_index, data.edge_weight)
    loss.backward()
    n_undirected = data.edge_weight.numel() / 2
    print(f"   |E| = {m}; normalised loss at all-p=0.5 = {loss.item():.4f} "
          f"(= -1/2); unnormalised = {loss.item() * n_undirected:.1f} "
          f"(= -|E|/2 = {-m/2:.1f}); grad norm = {p_half.grad.norm():.3e}")
    train = generate_batch("erdos_renyi", 100, count=200, base_seed=42)
    mean_m = np.mean([g.number_of_edges() for g in train])
    print(f"   training ER(100): mean |E| = {mean_m:.1f}; "
          f"batch-32 unnormalised loss at p=0.5 = {-32*mean_m/2:.1f} "
          f"(the original broken run's epoch-1 loss was -11763.8)")

    print("\n2) Constant-feature GIN output collapse (BatchNorm, original config)")
    torch.manual_seed(42)
    const = GINVariant(1, norm="batchnorm").eval()
    data_const = nx_to_pyg(G, posenc="random")
    data_const.x = torch.ones(100, 1)  # constant features, as the original code
    with torch.no_grad():
        p = const(data_const)
    S = {i for i in range(100) if p[i] > 0.5}
    if not S:
        S.add(0)
    elif len(S) == 100:
        S.discard(0)
    print(f"   p std = {p.std():.5f}; raw cut = {cut_fraction(G, S):.3f} of edges "
          f"(a coin flip cuts ~0.5)")

    print("\n3) Feature schemes: across-node output std (untrained, LapPE)")
    data_pe = nx_to_pyg(G, feature_seed=0, posenc="laplacian")
    for norm in ("graphnorm", "layernorm"):
        torch.manual_seed(42)
        model = GINVariant(5, norm=norm).eval()
        with torch.no_grad():
            p, stds = model(data_pe, return_layer_std=True)
        print(f"   {norm:10s}: p std = {p.std():.5f}; "
              f"layer across-node stds = {[round(s,4) for s in stds]}")
        if norm == "layernorm":
            print(f"      -> LayerNorm shrinks the across-node signal "
                  f"{stds[0]/stds[-1]:.0f}x from layer 1 to layer 5")


if __name__ == "__main__":
    main()
