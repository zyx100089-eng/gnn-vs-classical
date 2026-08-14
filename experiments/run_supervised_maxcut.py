"""
Follow-up experiment: supervised GNN Max-Cut with a 5x training budget.

The original paper trained the GNN *unsupervised* (maximising the
expected cut) for 80 epochs and it lost on 98.2% of instances. Two of
the paper's stated caveats were: "try a supervised Max-Cut baseline
instead of unsupervised only" and "give the GNN a serious training
budget". This experiment answers both at once:

1. Labels come from the spectral relaxation (which achieves ~97.8% of
   the best classical cut) — a strong, cheap teacher.
2. The GNN trains for 300 epochs (5x the original budget) on n=60
   graphs, with early-stopping-free cosine decay.

Question: does a supervised training signal plus a real training
budget change the conclusion of the paper? Same evaluation protocol:
held-out instances across 5 graph families and 3 sizes, compared
against random, greedy, spectral, and Goemans-Williamson.

Usage:
    python3 experiments/run_supervised_maxcut.py [--epochs 300] ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.evaluation.metrics import evaluate_solver
from src.classical.maxcut.greedy import greedy_maxcut
from src.classical.maxcut.random_cut import random_cut
from src.classical.maxcut.spectral import spectral_maxcut
from src.classical.maxcut.goemans_williamson import goemans_williamson
from src.gnn.models.gin import GINMaxCut
from src.gnn.training.maxcut_trainer import nx_to_pyg
from src.graphs.generators import GRAPH_FAMILIES, generate_batch, generate_instance
from src.graphs.properties import compute_properties


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_labels(G) -> torch.Tensor:
    """Supervised targets: node membership from the spectral relaxation."""
    S, _ = spectral_maxcut(G, refine=True)
    nodes = sorted(G.nodes())
    y = torch.tensor([1.0 if v in S else 0.0 for v in nodes])
    return y


def train_supervised(device: str, seed: int, epochs: int,
                     train_graphs: int, train_n: int) -> GINMaxCut:
    """Train the GNN with BCE against spectral-relaxation labels."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Generating {train_graphs} training graphs (erdos_renyi, n={train_n})...")
    graphs = generate_batch("erdos_renyi", train_n, count=train_graphs, base_seed=seed)
    labels = [make_labels(G) for G in graphs]

    model = GINMaxCut(input_dim=1, hidden_dim=128, n_layers=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    batch = 32
    log_rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(len(graphs))
        total = 0.0
        n_b = 0
        for start in range(0, len(perm), batch):
            idx = perm[start:start + batch]
            data = Batch.from_data_list([nx_to_pyg(graphs[i]) for i in idx])
            data = data.to(device)
            ys = torch.cat([labels[i] for i in idx]).to(device)

            optimizer.zero_grad()
            p = model(data)
            loss = F.binary_cross_entropy(p, ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n_b += 1
        scheduler.step()
        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | BCE: {total / n_b:.4f}")
        log_rows.append({"epoch": epoch, "avg_loss": total / n_b})

    results_dir = Path("results/analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(results_dir / "supervised_maxcut_training_log.csv",
                                  index=False)
    torch.save(model.state_dict(), results_dir / "supervised_maxcut_gnn_weights.pt")
    return model


from torch_geometric.data import Batch  # noqa: E402


def solve_with(model: GINMaxCut, G, device: str) -> tuple[set, float]:
    """Solve via GNN + local search (same post-processing as the paper)."""
    import networkx as nx
    model_cpu = model.cpu()
    model_cpu.eval()
    with torch.no_grad():
        p = model_cpu(nx_to_pyg(G))
    model.to(device)
    nodes = sorted(G.nodes())
    S = {nodes[i] for i in range(len(nodes)) if p[i].item() > 0.5}
    if len(S) == 0:
        S.add(nodes[0])
    elif len(S) == len(nodes):
        S.remove(nodes[0])
    # local search refinement
    improved = True
    while improved:
        improved = False
        for v in nodes:
            in_S = v in S
            gain = 0.0
            for u in G.neighbors(v):
                w = G[v][u].get("weight", 1.0)
                gain += w if (u in S) == in_S else -w
            if gain > 1e-10:
                S.add(v) if not in_S else S.remove(v)
                improved = True
    cut = sum(G[u][v].get("weight", 1.0) for u, v in G.edges()
              if (u in S) != (v in S))
    return S, cut


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--train_graphs", type=int, default=1200)
    ap.add_argument("--train_n", type=int, default=60)
    ap.add_argument("--sizes", type=str, default="20,50,100")
    ap.add_argument("--instances", type=int, default=30)
    ap.add_argument("--gw_max_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    # PHASE 1: supervised training
    print("=" * 60)
    print("PHASE 1: Supervised training (spectral labels, 5x budget)")
    print("=" * 60)
    model = train_supervised(device, args.seed, args.epochs,
                             args.train_graphs, args.train_n)
    print("training complete.")

    # PHASE 2: evaluation on the same protocol as the paper
    print("=" * 60)
    print("PHASE 2: Evaluation (5 families x 3 sizes x 30 instances)")
    print("=" * 60)
    sizes = [int(s) for s in args.sizes.split(",")]
    families = list(GRAPH_FAMILIES.keys())
    rows = []
    for family in families:
        for n in sizes:
            for inst in range(args.instances):
                seed_i = args.seed + inst * 1000
                G = generate_instance(family, n, seed=seed_i)
                row = {"family": family, "n": n, "instance": inst,
                       "m": G.number_of_edges(),
                       **{f"prop_{k}": v for k, v in
                          compute_properties(G).items()}}
                res = evaluate_solver(random_cut, G, seed=seed_i)
                row["random_cut"], row["random_time"] = res["cut_value"], res["runtime"]
                res = evaluate_solver(greedy_maxcut, G, seed=seed_i)
                row["greedy_cut"], row["greedy_time"] = res["cut_value"], res["runtime"]
                res = evaluate_solver(spectral_maxcut, G)
                row["spectral_cut"], row["spectral_time"] = res["cut_value"], res["runtime"]
                if n <= args.gw_max_n:
                    res = evaluate_solver(goemans_williamson, G,
                                          n_roundings=50, seed=seed_i,
                                          max_n=args.gw_max_n)
                    row["gw_cut"], row["gw_time"] = res["cut_value"], res["runtime"]
                else:
                    row["gw_cut"], row["gw_time"] = np.nan, np.nan
                res = evaluate_solver(lambda g, **kw: solve_with(model, g, device), G)
                row["gnn_cut"], row["gnn_time"] = res["cut_value"], res["runtime"]
                classical = [row["greedy_cut"], row["spectral_cut"]]
                if not np.isnan(row["gw_cut"]):
                    classical.append(row["gw_cut"])
                row["best_classical_cut"] = max(classical)
                row["gnn_wins"] = 1 if row["gnn_cut"] > row["best_classical_cut"] else 0
                row["gnn_relative"] = row["gnn_cut"] / max(row["best_classical_cut"], 1)
                rows.append(row)

    df = pd.DataFrame(rows)
    out = Path("results/analysis/supervised_maxcut_comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # PHASE 3: summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Instances: {len(df)} | GNN win rate: {df['gnn_wins'].mean():.1%}")
    print(f"GNN avg relative to best classical: {df['gnn_relative'].mean():.3f}")
    print(f"\nvs paper (unsupervised, 80 epochs, 2000 graphs): 1.8% wins")
    print("\nBy family:")
    for fam in families:
        sub = df[df.family == fam]
        print(f"  {fam:<22} wins {sub['gnn_wins'].mean():.0%} "
              f"| relative {sub['gnn_relative'].mean():.3f}")
    print("\nBy size:")
    for n in sizes:
        sub = df[df.n == n]
        print(f"  n={n:<5} wins {sub['gnn_wins'].mean():.0%} "
              f"| relative {sub['gnn_relative'].mean():.3f}")
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
