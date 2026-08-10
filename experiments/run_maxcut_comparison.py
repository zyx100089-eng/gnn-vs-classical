"""
Main Max-Cut comparison experiment.

1. Train GNN on Erdos-Renyi graphs
2. Generate test instances across all families and sizes
3. Run all solvers (random, greedy, spectral, GW, GNN)
4. Collect results and generate analysis
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

from src.graphs.generators import GRAPH_FAMILIES, generate_instance
from src.graphs.properties import compute_properties
from src.classical.maxcut.random_cut import random_cut
from src.classical.maxcut.greedy import greedy_maxcut
from src.classical.maxcut.spectral import spectral_maxcut
from src.classical.maxcut.goemans_williamson import goemans_williamson
from src.gnn.training.maxcut_trainer import train_maxcut_gnn, gnn_solve_maxcut
from src.evaluation.metrics import evaluate_solver


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="20,50,100,200")
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--train_graphs", type=int, default=2000)
    parser.add_argument("--train_n", type=int, default=100)
    parser.add_argument("--gnn_epochs", type=int, default=80)
    parser.add_argument("--gw_max_n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    device = get_device()
    print(f"Device: {device}")

    # =========================================================
    # PHASE 1: Train GNN
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Training GNN Max-Cut solver")
    print("=" * 60)

    model = train_maxcut_gnn(
        train_graphs=args.train_graphs,
        train_n=args.train_n,
        epochs=args.gnn_epochs,
        device=device,
        seed=args.seed,
        verbose=True,
    )
    print("GNN training complete.")

    # =========================================================
    # PHASE 2: Evaluate on all families and sizes
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Running comparison across families and sizes")
    print("=" * 60)

    families = list(GRAPH_FAMILIES.keys())
    all_results = []

    for family in families:
        for n in sizes:
            print(f"\n--- {family}, n={n} ({args.instances} instances) ---")

            for inst in range(args.instances):
                seed_i = args.seed + inst * 1000
                G = generate_instance(family, n, seed=seed_i)
                props = compute_properties(G)

                row = {
                    "family": family,
                    "n": n,
                    "instance": inst,
                    "m": G.number_of_edges(),
                    **{f"prop_{k}": v for k, v in props.items()},
                }

                # Random
                res = evaluate_solver(random_cut, G, seed=seed_i)
                row["random_cut"] = res["cut_value"]
                row["random_time"] = res["runtime"]

                # Greedy
                res = evaluate_solver(greedy_maxcut, G, seed=seed_i)
                row["greedy_cut"] = res["cut_value"]
                row["greedy_time"] = res["runtime"]

                # Spectral
                res = evaluate_solver(spectral_maxcut, G)
                row["spectral_cut"] = res["cut_value"]
                row["spectral_time"] = res["runtime"]

                # Goemans-Williamson
                if n <= args.gw_max_n:
                    res = evaluate_solver(
                        goemans_williamson, G,
                        n_roundings=50, seed=seed_i, max_n=args.gw_max_n,
                    )
                    row["gw_cut"] = res["cut_value"]
                    row["gw_time"] = res["runtime"]
                else:
                    row["gw_cut"] = np.nan
                    row["gw_time"] = np.nan

                # GNN
                res = evaluate_solver(
                    lambda g, **kw: gnn_solve_maxcut(model, g, device=device),
                    G,
                )
                row["gnn_cut"] = res["cut_value"]
                row["gnn_time"] = res["runtime"]

                # Best classical (for comparison)
                classical_cuts = [row["greedy_cut"], row["spectral_cut"]]
                if not np.isnan(row.get("gw_cut", np.nan)):
                    classical_cuts.append(row["gw_cut"])
                row["best_classical_cut"] = max(classical_cuts)
                row["gnn_wins"] = 1 if row["gnn_cut"] > row["best_classical_cut"] else 0

                all_results.append(row)

            # Print summary for this (family, size)
            batch = [r for r in all_results if r["family"] == family and r["n"] == n]
            gnn_avg = np.mean([r["gnn_cut"] for r in batch])
            greedy_avg = np.mean([r["greedy_cut"] for r in batch])
            spectral_avg = np.mean([r["spectral_cut"] for r in batch])
            gnn_win_rate = np.mean([r["gnn_wins"] for r in batch])
            print(f"  GNN: {gnn_avg:.1f} | Greedy: {greedy_avg:.1f} | "
                  f"Spectral: {spectral_avg:.1f} | GNN win rate: {gnn_win_rate:.0%}")

    # =========================================================
    # PHASE 3: Save results and generate figures
    # =========================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Analysis")
    print("=" * 60)

    df = pd.DataFrame(all_results)
    results_dir = Path("results/analysis")
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "maxcut_comparison.csv", index=False)

    fig_dir = Path("analysis/figures/maxcut")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Performance heatmap (GNN win rate by family x size) ---
    pivot = df.groupby(["family", "n"])["gnn_wins"].mean().unstack()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Graph Size (n)")
    ax.set_ylabel("Graph Family")
    ax.set_title("GNN Win Rate vs Best Classical (Max-Cut)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                    color="black" if 0.3 < val < 0.7 else "white", fontsize=10)
    plt.colorbar(im, label="GNN Win Rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "win_rate_heatmap.png", dpi=300)
    plt.close()
    print("Saved win_rate_heatmap.png")

    # --- Figure 2: Cut value comparison by size (aggregated across families) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, color, label in [
        ("gnn_cut", "blue", "GNN"),
        ("greedy_cut", "orange", "Greedy"),
        ("spectral_cut", "green", "Spectral"),
        ("gw_cut", "red", "Goemans-Williamson"),
        ("random_cut", "gray", "Random"),
    ]:
        sub = df.dropna(subset=[algo])
        if sub.empty:
            continue
        means = sub.groupby("n")[algo].mean()
        stds = sub.groupby("n")[algo].std()
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    label=label, marker="o", capsize=3)
    ax.set_xlabel("Graph Size (n)")
    ax.set_ylabel("Cut Value")
    ax.set_title("Max-Cut: Algorithm Comparison by Graph Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "cut_vs_size.png", dpi=300)
    plt.close()
    print("Saved cut_vs_size.png")

    # --- Figure 3: Relative performance (GNN / best classical) ---
    df["gnn_relative"] = df["gnn_cut"] / df["best_classical_cut"].clip(lower=1)
    fig, axes = plt.subplots(1, len(families), figsize=(4 * len(families), 4), sharey=True)
    for ax, fam in zip(axes, families):
        sub = df[df["family"] == fam]
        means = sub.groupby("n")["gnn_relative"].mean()
        stds = sub.groupby("n")["gnn_relative"].std()
        ax.bar(range(len(means)), means.values, yerr=stds.values,
               color=["green" if v >= 1 else "red" for v in means.values],
               alpha=0.7, capsize=3)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index)
        ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(fam, fontsize=9)
        ax.set_xlabel("n")
    axes[0].set_ylabel("GNN / Best Classical")
    fig.suptitle("GNN Relative Performance by Family and Size", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "relative_performance.png", dpi=300)
    plt.close()
    print("Saved relative_performance.png")

    # --- Figure 4: Runtime comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, color, label in [
        ("gnn_time", "blue", "GNN (inference)"),
        ("greedy_time", "orange", "Greedy"),
        ("spectral_time", "green", "Spectral"),
        ("gw_time", "red", "Goemans-Williamson"),
    ]:
        sub = df.dropna(subset=[algo])
        if sub.empty:
            continue
        means = sub.groupby("n")[algo].mean()
        ax.plot(means.index, means.values, label=label, marker="o", color=color)
    ax.set_xlabel("Graph Size (n)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Algorithm Runtime vs Graph Size")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "runtime_comparison.png", dpi=300)
    plt.close()
    print("Saved runtime_comparison.png")

    # --- Summary stats ---
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"\nTotal instances: {len(df)}")
    print(f"GNN overall win rate: {df['gnn_wins'].mean():.1%}")
    print(f"\nBy family:")
    for fam in families:
        sub = df[df["family"] == fam]
        print(f"  {fam:<22} GNN wins {sub['gnn_wins'].mean():.0%} "
              f"| avg relative: {sub['gnn_relative'].mean():.3f}")
    print(f"\nBy size:")
    for n in sizes:
        sub = df[df["n"] == n]
        print(f"  n={n:<5} GNN wins {sub['gnn_wins'].mean():.0%} "
              f"| avg relative: {sub['gnn_relative'].mean():.3f}")


if __name__ == "__main__":
    main()
