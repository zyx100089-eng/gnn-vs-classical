"""
Training loop for GNN Max-Cut solver.

Trains on a distribution of random graphs (Erdos-Renyi by default)
using the unsupervised cut maximization loss. No ground-truth labels needed.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

from src.gnn.models.gin import GINMaxCut, maxcut_loss
from src.graphs.generators import generate_batch
from src.classical.maxcut.local_search import local_search_refine  # noqa: F401


def nx_to_pyg(G, feature_seed: int = None, posenc: str = "laplacian",
              pe_dim: int = 4) -> Data:
    """Convert a networkx graph to PyTorch Geometric Data.

    Node features must break the symmetry between nodes — with identical
    inputs the max-cut loss has an exact critical point at the all-0.5
    output (the random-cut baseline) and a deterministic message-passing
    net cannot distinguish nodes there. Two schemes:

      * 'laplacian' (default): the node's normalised degree plus its
        coordinates in the top-`pe_dim` eigenvectors of the graph
        Laplacian — smooth positional encodings that survive message
        passing and tie the learned model to the spectral relaxation.
      * 'random': fresh N(0,1) noise per node (`feature_seed` makes it
        reproducible). Iid noise is largely averaged away by repeated
        neighbourhood aggregation, so this is the weaker of the two.
    """
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

    n = len(nodes)
    if posenc == "laplacian":
        L = nx.laplacian_matrix(G, nodelist=nodes).astype(float).toarray()
        evals, evecs = np.linalg.eigh(L)
        pe = evecs[:, -pe_dim:]  # top-pe_dim eigenvectors (non-trivial)
        # canonical global sign per eigenvector (eigh's sign is arbitrary)
        for j in range(pe.shape[1]):
            if pe[np.argmax(np.abs(pe[:, j])), j] < 0:
                pe[:, j] = -pe[:, j]
        # scale to unit std so PE entries are O(1) like the degree feature
        pe = pe / np.maximum(pe.std(axis=0), 1e-12)
        deg = np.array([G.degree(v) for v in nodes], dtype=float)
        x = torch.tensor(
            np.column_stack([deg / max(deg.max(), 1.0), pe]), dtype=torch.float
        )
    elif posenc == "random":
        if feature_seed is None:
            x = torch.randn(n, 1)
        else:
            g = torch.Generator().manual_seed(int(feature_seed))
            x = torch.randn(n, 1, generator=g)
    else:
        raise ValueError(f"unknown posenc {posenc!r}")

    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight,
                num_nodes=n)


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
    val_fraction: float = 0.1,
    posenc: str = "laplacian",
    pe_dim: int = 4,
    logit_init_std: float = 1.0,
    run_tag: str = "",
) -> GINMaxCut:
    """Train a GNN Max-Cut solver on a distribution of random graphs.

    A held-out validation slice (last `val_fraction` of the generated
    graphs) is scored every epoch and the best checkpoint by validation
    loss is returned — the training curve alone can pick a checkpoint
    that is worse on unseen graphs.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate training graphs
    if verbose:
        print(f"Generating {train_graphs} training graphs ({train_family}, n={train_n})...")
    nx_graphs = generate_batch(train_family, train_n, count=train_graphs, base_seed=seed)
    n_val = max(1, int(round(train_graphs * val_fraction)))
    val_graphs = nx_graphs[-n_val:]
    train_only = nx_graphs[:-n_val]
    pyg_train = [nx_to_pyg(G, feature_seed=seed + i, posenc=posenc, pe_dim=pe_dim)
                 for i, G in enumerate(train_only)]
    pyg_val = [nx_to_pyg(G, feature_seed=seed + 10_000 + i, posenc=posenc, pe_dim=pe_dim)
               for i, G in enumerate(val_graphs)]
    input_dim = pyg_train[0].x.shape[1]

    loader = DataLoader(pyg_train, batch_size=batch_size, shuffle=True)
    val_data = Batch.from_data_list(pyg_val).to(device)

    def val_loss(model) -> float:
        model.eval()
        with torch.no_grad():
            p = model(val_data)
            v = maxcut_loss(p, val_data.edge_index, val_data.edge_weight)
        model.train()
        return v.item()

    model = GINMaxCut(input_dim=input_dim, hidden_dim=hidden_dim,
                      n_layers=n_layers, logit_init_std=logit_init_std)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log_rows = []
    best_val = float("inf")
    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

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
        v = val_loss(model)
        if v < best_val:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | train: {avg_loss:.4f} | val: {v:.4f}")
        log_rows.append({"epoch": epoch, "avg_loss": avg_loss, "val_loss": v})

    model.load_state_dict(best_state)

    # Persist the training log and weights alongside the results, so
    # the training is reproducible from the repo. `run_tag` lets
    # multi-seed runs write distinct files instead of clobbering the
    # canonical model (e.g. run_tag="_seed43").
    results_dir = Path("results/analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(
        results_dir / f"maxcut_training_log{run_tag}.csv", index=False)
    torch.save(model.state_dict(),
               results_dir / f"maxcut_gnn_weights{run_tag}.pt")

    return model


def gnn_solve_maxcut(model: GINMaxCut, G, device: str = "cpu",
                     refine: bool = True, n_roundings: int = 50,
                     seed: int = 0, posenc: str = "laplacian",
                     pe_dim: int = 4) -> tuple[set, float]:
    """Use a trained GNN to solve Max-Cut on a single graph.

    Decoding: one forward pass over the node features gives per-node
    probabilities p; the partition is then sampled as Bernoulli(p) coin
    flips `n_roundings` times and the best cut kept — the same
    best-of-many-rounding budget the Goemans-Williamson solver gets.
    With `refine`, the best sample is improved by the same 1-opt local
    search used across the classical baselines. `posenc`/`pe_dim` must
    match the configuration the model was trained with.
    """
    import networkx as nx

    # Always run inference on CPU to avoid MPS size-mismatch issues
    model_cpu = model.cpu()
    data = nx_to_pyg(G, feature_seed=seed, posenc=posenc, pe_dim=pe_dim)
    model_cpu.eval()
    with torch.no_grad():
        p = model_cpu(data)
    model.to(device)

    nodes = sorted(G.nodes())
    probs = p.detach().cpu().numpy()
    rng = np.random.RandomState(seed)
    best_S, best_cut = None, -1.0
    for _ in range(max(1, n_roundings)):
        S = {nodes[i] for i in range(len(nodes)) if rng.random() < probs[i]}
        if len(S) == 0:
            S.add(nodes[0])
        elif len(S) == len(nodes):
            S.discard(nodes[0])
        cut = sum(
            G[u][v].get("weight", 1.0)
            for u, v in G.edges()
            if (u in S) != (v in S)
        )
        if cut > best_cut:
            best_cut, best_S = cut, S

    # Refine with local search (standard practice — GNN gives initial partition)
    if refine:
        best_S = local_search_refine(G, best_S)
        best_cut = sum(
            G[u][v].get("weight", 1.0)
            for u, v in G.edges()
            if (u in best_S) != (v in best_S)
        )

    return best_S, best_cut
