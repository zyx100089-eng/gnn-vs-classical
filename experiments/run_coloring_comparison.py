"""
Graph Coloring comparison: Greedy, Welsh-Powell, DSatur, GNN.
Tests across multiple graph families and sizes.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from src.graphs.generators import generate_instance, GRAPH_FAMILIES
from src.graphs.properties import compute_properties
from src.classical.coloring.greedy import greedy_coloring, welsh_powell_coloring
from src.classical.coloring.dsatur import dsatur_coloring, verify_coloring
from src.gnn.models.coloring_gnn import GINColoring, coloring_loss


def nx_to_pyg_coloring(G) -> Data:
    """Convert networkx graph to PyG data with degree features."""
    import networkx as nx
    n = G.number_of_nodes()
    node_list = sorted(G.nodes())
    node_map = {v: i for i, v in enumerate(node_list)}

    edges = []
    for u, v in G.edges():
        edges.append([node_map[u], node_map[v]])
        edges.append([node_map[v], node_map[u]])

    if not edges:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    degrees = [G.degree(v) for v in node_list]
    x = torch.tensor(degrees, dtype=torch.float).unsqueeze(-1)

    return Data(x=x, edge_index=edge_index, num_nodes=n)


def train_coloring_gnn(n_train: int = 1500, epochs: int = 100,
                       seed: int = 42, device: str = "cpu") -> GINColoring:
    """Train GNN on random graphs for coloring."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Generating {n_train} training graphs...")
    train_data = []
    for i in range(n_train):
        G = generate_instance("erdos_renyi", n=50, seed=seed + i)
        train_data.append(nx_to_pyg_coloring(G))

    model = GINColoring(input_dim=1, hidden_dim=128, max_colors=20, n_layers=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        np.random.shuffle(train_data)

        for data in train_data[:500]:
            data = data.to(device)
            logits = model(data)
            loss = coloring_loss(logits, data.edge_index)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / 500
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f}")
        log_rows.append({"epoch": epoch, "avg_loss": avg_loss})

    # Persist the training log and weights alongside the results, so
    # the training is reproducible from the repo.
    results_dir = Path("results/analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(results_dir / "coloring_training_log.csv",
                                  index=False)
    torch.save(model.state_dict(), results_dir / "coloring_gnn_weights.pt")

    return model


def gnn_coloring(model: GINColoring, G, device: str = "cpu") -> tuple[dict, int]:
    """Run GNN coloring inference with conflict repair."""
    import networkx as nx
    data = nx_to_pyg_coloring(G).to(device)
    node_list = sorted(G.nodes())

    model.eval()
    with torch.no_grad():
        logits = model(data)
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    colors = probs.argmax(axis=1)

    coloring = {node_list[i]: int(colors[i]) for i in range(len(node_list))}

    # Repair conflicts
    for u, v in G.edges():
        if coloring[u] == coloring[v]:
            neighbor_colors = {coloring[w] for w in G.neighbors(v) if w in coloring}
            c = 0
            while c in neighbor_colors:
                c += 1
            coloring[v] = c

    n_colors = max(coloring.values()) + 1 if coloring else 0
    return coloring, n_colors


def main():
    device = "cpu"
    sizes = [20, 50, 100, 200]
    families = ["erdos_renyi", "planted_partition", "random_regular",
                "barabasi_albert", "watts_strogatz"]
    n_instances = 30

    # Train GNN
    print("=" * 60)
    print("Training GNN Coloring solver")
    print("=" * 60)
    model = train_coloring_gnn(epochs=80, device=device)

    # Evaluate
    print("\n" + "=" * 60)
    print("Running Coloring comparison")
    print("=" * 60)

    all_results = []

    for family in families:
        for n in sizes:
            print(f"\n--- {family} n={n} ---")
            for inst in range(n_instances):
                seed_i = 42 + inst * 1000
                G = generate_instance(family, n=n, seed=seed_i)
                props = compute_properties(G)

                row = {"family": family, "n": n, "instance": inst}
                row.update({k: v for k, v in props.items()
                           if isinstance(v, (int, float))})

                # Greedy
                t0 = time.perf_counter()
                _, greedy_c = greedy_coloring(G, seed=seed_i)
                row["greedy_colors"] = greedy_c
                row["greedy_time"] = time.perf_counter() - t0

                # Welsh-Powell
                t0 = time.perf_counter()
                _, wp_c = welsh_powell_coloring(G)
                row["welsh_powell_colors"] = wp_c
                row["welsh_powell_time"] = time.perf_counter() - t0

                # DSatur
                t0 = time.perf_counter()
                dsatur_col, dsatur_c = dsatur_coloring(G)
                row["dsatur_colors"] = dsatur_c
                row["dsatur_time"] = time.perf_counter() - t0
                row["dsatur_valid"] = verify_coloring(G, dsatur_col)

                # GNN
                t0 = time.perf_counter()
                gnn_col, gnn_c = gnn_coloring(model, G, device=device)
                row["gnn_colors"] = gnn_c
                row["gnn_time"] = time.perf_counter() - t0
                row["gnn_valid"] = verify_coloring(G, gnn_col)

                row["best_classical"] = min(greedy_c, wp_c, dsatur_c)
                row["gnn_wins"] = 1 if gnn_c < row["best_classical"] else 0

                all_results.append(row)

            batch = [r for r in all_results
                     if r["family"] == family and r["n"] == n]
            print(f"  Greedy: {np.mean([r['greedy_colors'] for r in batch]):.1f} | "
                  f"WP: {np.mean([r['welsh_powell_colors'] for r in batch]):.1f} | "
                  f"DSatur: {np.mean([r['dsatur_colors'] for r in batch]):.1f} | "
                  f"GNN: {np.mean([r['gnn_colors'] for r in batch]):.1f} | "
                  f"Win: {np.mean([r['gnn_wins'] for r in batch]):.0%}")

    df = pd.DataFrame(all_results)

    results_dir = Path("results/analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "coloring_comparison.csv", index=False)

    fig_dir = Path("analysis/figures/coloring")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Win rate heatmap
    pivot = df.pivot_table(values="gnn_wins", index="family", columns="n",
                          aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=0.5,
                   aspect="auto")
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(sizes)):
            ax.text(j, i, f"{pivot.values[i, j]:.0%}",
                   ha="center", va="center", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="GNN Win Rate")
    ax.set_title("Graph Coloring: GNN Win Rate vs Best Classical")
    ax.set_xlabel("Graph Size (n)")
    plt.tight_layout()
    plt.savefig(fig_dir / "coloring_win_rate.png", dpi=300)
    plt.close()
    print("\nSaved coloring_win_rate.png")

    # Figure 2: Colors used comparison
    fig, axes = plt.subplots(1, len(families), figsize=(20, 4), sharey=True)
    for idx, family in enumerate(families):
        sub = df[df["family"] == family]
        ax = axes[idx]
        for col, color, label in [
            ("gnn_colors", "blue", "GNN"),
            ("dsatur_colors", "red", "DSatur"),
            ("welsh_powell_colors", "green", "Welsh-Powell"),
            ("greedy_colors", "gray", "Greedy"),
        ]:
            means = sub.groupby("n")[col].mean()
            ax.plot(means.index, means.values, marker="o", color=color, label=label)
        ax.set_xlabel("n")
        if idx == 0:
            ax.set_ylabel("Colors Used")
        ax.set_title(family.replace("_", "\n"))
        ax.grid(True, alpha=0.3)
    axes[-1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.suptitle("Graph Coloring: Colors Used by Algorithm", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "coloring_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved coloring_comparison.png")

    # Summary
    print(f"\nGNN overall win rate: {df['gnn_wins'].mean():.1%}")
    print(f"GNN all valid: {df['gnn_valid'].all()}")
    df["gnn_relative"] = df["gnn_colors"] / df["best_classical"]
    print(f"GNN avg relative colors: {df['gnn_relative'].mean():.4f}")


if __name__ == "__main__":
    main()
