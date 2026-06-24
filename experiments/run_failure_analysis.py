"""
Run failure prediction analysis on completed Max-Cut comparison results.

Usage: python experiments/run_failure_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from src.analysis.failure_prediction import run_failure_prediction


def main():
    results_file = Path("results/analysis/maxcut_comparison.csv")
    if not results_file.exists():
        print("Error: Run run_maxcut_comparison.py first to generate results.")
        return

    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} instances")

    # Compute relative performance if not present
    if "gnn_relative" not in df.columns:
        df["gnn_relative"] = df["gnn_cut"] / df["best_classical_cut"].clip(lower=1)

    fig_dir = Path("analysis/figures/failure_prediction")
    results = run_failure_prediction(df, fig_dir)

    print("\n" + "=" * 60)
    print("FAILURE PREDICTION RESULTS")
    print("=" * 60)

    print(f"\nLogistic Regression accuracy: {results['lr_accuracy']:.3f}")
    print(f"Random Forest accuracy: {results['rf_accuracy']:.3f}")

    print("\nTop LR features (+ = favors GNN):")
    for name, coef in results["top_lr_features"]:
        print(f"  {name:<25} {coef:+.4f}")

    print("\nTop RF features (importance):")
    for name, imp in results["top_rf_features"]:
        print(f"  {name:<25} {imp:.4f}")

    with open(fig_dir / "prediction_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
