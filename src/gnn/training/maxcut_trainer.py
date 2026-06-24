"""
Training loop for GNN Max-Cut solver.

Trains on a distribution of random graphs (Erdos-Renyi by default)
using the unsupervised cut maximization loss. No ground-truth labels needed.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from src.gnn.models.gin import GINMaxCut, maxcut_loss
from src.graphs.generators import generate_batch


def nx_to_pyg(G) -> Data:
    """Convert a networkx graph to PyTorch Geometric Data."""
    import networkx as nx
    nodes = sorted(G.nodes())
    node_map = {v: i for i, v in enumerate(nodes)}

    edges = []
    weights = []
    for u, v, d in G.edges(data=True):
        i, j = node_map[u], node_map[v]
        edges.append([i, j])
        edges.append([j, i])
        w = d.get("weight", 1.0)
        weights.append(w)
        weights.append(w)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(weights, dtype=torch.float)
    x = torch.ones(len(nodes), 1)  # constant node features

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                num_nodes=len(nodes))


def train_maxcut_gnn(
    hidden_dim: int = 128,
    n_layers: int = 5,
    train_graphs: int = 2000,
    train_n: int = 100,
    train_family: str = "erdos_renyi",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    verbose: bool = True,
    seed: int = 42,
) -> GINMaxCut:
    """Train a GNN Max-Cut solver on a distribution of random graphs."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate training graphs
    if verbose:
        print(f"Generating {train_graphs} training graphs ({train_family}, n={train_n})...")
    nx_graphs = generate_batch(train_family, train_n, count=train_graphs, base_seed=seed)
    pyg_graphs = [nx_to_pyg(G) for G in nx_graphs]

    loader = DataLoader(pyg_graphs, batch_size=batch_size, shuffle=True)

    model = GINMaxCut(input_dim=1, hidden_dim=hidden_dim, n_layers=n_layers)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            p = model(batch)
            loss = maxcut_loss(p, batch.edge_index, batch.edge_weight)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")

    return model


def gnn_solve_maxcut(model: GINMaxCut, G, device: str = "cpu",
                     refine: bool = True) -> tuple[set, float]:
    """Use a trained GNN to solve Max-Cut on a single graph, then optionally refine."""
    import networkx as nx

    # Always run inference on CPU to avoid MPS size-mismatch issues
    model_cpu = model.cpu()
    data = nx_to_pyg(G)
    model_cpu.eval()
    with torch.no_grad():
        p = model_cpu(data)
    model.to(device)

    nodes = sorted(G.nodes())
    S = {nodes[i] for i in range(len(nodes)) if p[i].item() > 0.5}
    if len(S) == 0:
        S.add(nodes[0])
    elif len(S) == len(nodes):
        S.remove(nodes[0])

    # Refine with local search (standard practice — GNN gives initial partition)
    if refine:
        improved = True
        while improved:
            improved = False
            for v in nodes:
                in_S = v in S
                gain = 0.0
                for u in G.neighbors(v):
                    w = G[v][u].get("weight", 1.0)
                    if (u in S) == in_S:
                        gain += w
                    else:
                        gain -= w
                if gain > 1e-10:
                    if in_S:
                        S.remove(v)
                    else:
                        S.add(v)
                    improved = True

    cut_value = sum(
        G[u][v].get("weight", 1.0)
        for u, v in G.edges()
        if (u in S) != (v in S)
    )
    return S, cut_value
