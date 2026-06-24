"""
TSP comparison: Nearest Neighbor, Christofides, 2-opt, GNN.
Uses Euclidean TSP instances (random points in [0,1]^2).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.classical.tsp.nearest_neighbor import nearest_neighbor_tsp
from src.classical.tsp.christofides import christofides_tsp
from src.classical.tsp.two_opt import two_opt_improve
from src.gnn.models.tsp_gnn import GINTsp, greedy_tour_from_embeddings


def generate_tsp_instance(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate random Euclidean TSP instance."""
    rng = np.random.RandomState(seed)
    coords = rng.rand(n, 2)
    dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
    return coords, dist


def coords_to_pyg(coords: np.ndarray) -> Data:
    """Convert TSP coordinates to PyG data (complete graph)."""
    n = len(coords)
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(coords, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, num_nodes=n)


def train_tsp_gnn(sizes: list, n_train: int = 1000, epochs: int = 80,
                  seed: int = 42, device: str = "cpu") -> GINTsp:
    """Train GNN on TSP instances using REINFORCE-like loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_n = 50

    print(f"Generating {n_train} training TSP instances (n={train_n})...")
    train_data = []
    train_dists = []
    for i in range(n_train):
        coords, dist = generate_tsp_instance(train_n, seed=seed + i)
        train_data.append(coords_to_pyg(coords))
        train_dists.append(dist)

    model = GINTsp(input_dim=2, hidden_dim=128, n_layers=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0

        for idx in range(0, n_train, 32):
            batch_indices = range(idx, min(idx + 32, n_train))
            batch_loss = torch.tensor(0.0, device=device)

            for i in batch_indices:
                data = train_data[i].to(device)
                embeddings = model(data).detach().cpu().numpy()
                _, gnn_length = greedy_tour_from_embeddings(embeddings, train_dists[i])
                _, nn_length = nearest_neighbor_tsp(train_dists[i])
                reward = (nn_length - gnn_length) / nn_length

                emb = model(data)
                # Simple loss: encourage embeddings that produce shorter tours
                edge_scores = (emb[data.edge_index[0]] * emb[data.edge_index[1]]).sum(dim=-1)
                batch_loss = batch_loss - reward * edge_scores.mean()

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += batch_loss.item()
            n_batches += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {total_loss/max(n_batches,1):.4f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="20,50,100")
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--gnn_epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    device = "cpu"  # CPU for stability

    # Train GNN
    print("=" * 60)
    print("Training GNN TSP solver")
    print("=" * 60)
    model = train_tsp_gnn(sizes, epochs=args.gnn_epochs, seed=args.seed, device=device)

    # Evaluate
    print("\n" + "=" * 60)
    print("Running TSP comparison")
    print("=" * 60)

    all_results = []

    for n in sizes:
        print(f"\n--- n={n} ({args.instances} instances) ---")

        for inst in range(args.instances):
            seed_i = args.seed + inst * 1000
            coords, dist = generate_tsp_instance(n, seed=seed_i)

            row = {"n": n, "instance": inst}

            # Nearest Neighbor
            t0 = time.perf_counter()
            nn_tour, nn_len = nearest_neighbor_tsp(dist)
            row["nn_length"] = nn_len
            row["nn_time"] = time.perf_counter() - t0

            # Christofides
            t0 = time.perf_counter()
            ch_tour, ch_len = christofides_tsp(dist)
            row["christofides_length"] = ch_len
            row["christofides_time"] = time.perf_counter() - t0

            # 2-opt (starting from NN)
            t0 = time.perf_counter()
            opt_tour, opt_len = two_opt_improve(nn_tour, dist)
            row["twoopt_length"] = opt_len
            row["twoopt_time"] = time.perf_counter() - t0

            # GNN
            t0 = time.perf_counter()
            data = coords_to_pyg(coords).to(device)
            model.eval()
            with torch.no_grad():
                embeddings = model(data).cpu().numpy()
            gnn_tour, gnn_len = greedy_tour_from_embeddings(embeddings, dist)
            # Refine with 2-opt
            gnn_tour, gnn_len = two_opt_improve(gnn_tour, dist)
            row["gnn_length"] = gnn_len
            row["gnn_time"] = time.perf_counter() - t0

            row["best_classical"] = min(row["christofides_length"], row["twoopt_length"])
            row["gnn_wins"] = 1 if gnn_len < row["best_classical"] else 0

            all_results.append(row)

        batch = [r for r in all_results if r["n"] == n]
        print(f"  NN: {np.mean([r['nn_length'] for r in batch]):.3f} | "
              f"Christofides: {np.mean([r['christofides_length'] for r in batch]):.3f} | "
              f"2-opt: {np.mean([r['twoopt_length'] for r in batch]):.3f} | "
              f"GNN+2opt: {np.mean([r['gnn_length'] for r in batch]):.3f} | "
              f"GNN win: {np.mean([r['gnn_wins'] for r in batch]):.0%}")

    df = pd.DataFrame(all_results)

    results_dir = Path("results/analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "tsp_comparison.csv", index=False)

    fig_dir = Path("analysis/figures/tsp")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Tour length comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, color, label in [
        ("gnn_length", "blue", "GNN+2opt"),
        ("christofides_length", "red", "Christofides"),
        ("twoopt_length", "green", "NN+2opt"),
        ("nn_length", "gray", "Nearest Neighbor"),
    ]:
        means = df.groupby("n")[col].mean()
        stds = df.groupby("n")[col].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    label=label, marker="o", capsize=3, color=color)
    ax.set_xlabel("Number of Cities")
    ax.set_ylabel("Tour Length")
    ax.set_title("TSP: Algorithm Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "tsp_comparison.png", dpi=300)
    plt.close()
    print("\nSaved tsp_comparison.png")

    # Summary
    print(f"\nGNN overall win rate vs best classical: {df['gnn_wins'].mean():.1%}")
    df["gnn_relative"] = df["gnn_length"] / df["best_classical"]
    print(f"GNN avg relative length: {df['gnn_relative'].mean():.4f}")


if __name__ == "__main__":
    main()
