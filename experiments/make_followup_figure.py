"""Generate the follow-up figure: supervised vs unsupervised GNN + classical.

Reads results/analysis/maxcut_comparison.csv (paper, unsupervised) and
results/analysis/supervised_maxcut_comparison.csv (follow-up, supervised),
and plots relative-to-best-classical performance per family.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

results_dir = Path("results/analysis")
fig_dir = Path("analysis/figures/followup")
fig_dir.mkdir(parents=True, exist_ok=True)

paper = pd.read_csv(results_dir / "maxcut_comparison.csv")
followup = pd.read_csv(results_dir / "supervised_maxcut_comparison.csv")

paper["gnn_relative"] = paper["gnn_cut"] / paper["best_classical_cut"].clip(lower=1)
followup["gnn_relative"] = followup["gnn_cut"] / followup["best_classical_cut"].clip(lower=1)

families = ["erdos_renyi", "planted_partition", "random_regular",
            "barabasi_albert", "watts_strogatz"]

x = np.arange(len(families))
w = 0.35

paper_rel = [paper[paper.family == f]["gnn_relative"].mean() for f in families]
follow_rel = [followup[followup.family == f]["gnn_relative"].mean() for f in families]

fig, ax = plt.subplots(figsize=(9, 4.5))
b1 = ax.bar(x - w / 2, paper_rel, w, label="Unsupervised (paper: 80 epochs)",
            color="#c44e52", alpha=0.85)
b2 = ax.bar(x + w / 2, follow_rel, w, label="Supervised (follow-up: 300 epochs)",
            color="#4c72b0", alpha=0.85)
for b, vals in ((b1, paper_rel), (b2, follow_rel)):
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.axhline(1.0, color="black", ls="--", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(families, rotation=20, ha="right")
ax.set_ylabel("GNN / Best Classical Cut")
ax.set_title("Supervised training + 5x budget does not close the gap")
ax.legend()
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / "supervised_vs_unsupervised.png", dpi=300)
print("saved", fig_dir / "supervised_vs_unsupervised.png")
