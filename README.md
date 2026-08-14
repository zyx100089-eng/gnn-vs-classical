# GNNs vs. Proven Approximation Algorithms

[![Tests](https://github.com/zyx100089-eng/gnn-vs-classical/actions/workflows/tests.yml/badge.svg)](https://github.com/zyx100089-eng/gnn-vs-classical/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Short answer: No.** On 600 Max-Cut instances, my GNN beats the best
classical algorithm on 1.8% of graphs. On TSP, GNN+2-opt collapses to
NN+2-opt. Failure prediction from graph features sits at chance
(balanced accuracy 0.52).

> **Full write-up:** [paper/main.pdf](paper/main.pdf) — a LaTeX paper
> with the complete methodology, results tables, and analysis
> (source: `paper/main.tex`).

## Why I built this

I kept reading papers claiming GNNs can *solve* NP-hard problems like
Max-Cut and TSP. But I'd also read about approximation algorithms with
real guarantees — Goemans-Williamson (0.878·OPT), Christofides (1.5·OPT)
— and I wanted to test one against the other on equal footing, myself,
rather than trust either the papers or the hype. This project is that
test, with the training code and results committed so it's reproducible.

I did not cherry-pick the outcome. I would have been happy to find the
GNN won; instead it mostly lost, and losing is the interesting result.

## What I did

Three problems, three comparisons, one methodology:

1. Train a GNN solver (5-layer GIN) on each problem.
2. Implement the classical baselines *with* their guarantees, from
   scratch: Goemans-Williamson (SDP + random hyperplane rounding),
   Christofides, DSatur, plus simpler baselines.
3. Run both on the same held-out instances across six graph families
   and report solution quality and runtime.

![GNN win rate vs best classical, by graph size and family](docs/win_rate_heatmap.png)

*Win rate of the GNN against the best classical solver, per graph
family and size (from the Max-Cut comparison).*

![Runtime vs graph size](docs/runtime_comparison.png)

*Solver runtime vs graph size: the GNN's one clear advantage is
inference speed. Note the caveat below — this is only measured where
the SDP actually runs.*

| Problem | GNN vs classical |
|---|---|
| Max-Cut | GNN wins 11/600 instances (1.8%); achieves ~96% of the best classical cut on average — competitive but inferior. Goemans-Williamson dominates |
| TSP | GNN+2-opt ≈ NN+2-opt. Christofides+2-opt consistently wins. Trained GNN collapses to nearest-neighbour (embedding cosines all 1.0) |
| Coloring | DSatur uses fewer colors on structured graphs; GNN struggles on regular and small-world graphs |

## The honest caveats

- **Did the GNN get a fair shot?** Mostly, but not fully. It got a
  modest training budget (80 epochs on 2000 synthetic graphs), its
  architecture is a plain 5-layer GIN, and Max-Cut was trained
  unsupervised. Classical algorithms had decades of tuning behind
  them. If anything, this *understates* the gap — a better-trained GNN
  might close some of it — but the failure-prediction result suggests
  the GNN's rare wins are essentially noise, not a recoverable signal.
- **The speed claim, scoped.** "GNN inference is faster than SDP" is
  only measured where SDP actually runs (n ≤ 200 in my experiments;
  above that, my SDP falls back to spectral relaxation with a warning).
  At the sizes where both run, SDP's runtime explodes and the GNN
  wins on speed — but I did not measure SDP at large n because it
  couldn't run.
- **Failure prediction at 1.8% positives.** Balanced accuracy 0.52
  with 52 positives out of 600 is almost a foregone conclusion — the
  class imbalance makes any predictor trivially poor. The honest
  statement is: with so few GNN wins, there is almost no signal to
  learn from, and the data says so.
- **GNN speed is not a solution-quality advantage.** The GNN's only
  clear win is inference speed. It does not produce better solutions.

## Algorithms implemented

### Classical (with guarantees)

| Algorithm | Problem | Guarantee |
|-----------|---------|-----------|
| Random partition | Max-Cut | ≥ 0.5·OPT |
| Greedy local search | Max-Cut | ≥ 0.5·OPT |
| Spectral relaxation | Max-Cut | Based on λ_max(L) |
| **Goemans-Williamson** | Max-Cut | **≥ 0.878·OPT** |
| Nearest Neighbour | TSP | O(log n)·OPT |
| **Christofides** | TSP | **≤ 1.5·OPT** |
| 2-opt local search | TSP | No guarantee |
| Greedy / Welsh-Powell | Coloring | ≤ Δ+1 colors |
| **DSatur** | Coloring | **Optimal on bipartite** |

### Learned (no guarantees)

| Algorithm | Problem | Architecture |
|-----------|---------|-------------|
| GIN (unsupervised) | Max-Cut | 5-layer GIN + local search |
| GIN (REINFORCE) | TSP | 5-layer GIN + policy gradient + greedy + 2-opt |
| GIN (classification) | Coloring | 5-layer GIN + conflict repair |

## Project structure

```
src/
├── graphs/              # 6 graph family generators + spectral properties
├── classical/
│   ├── maxcut/          # Random, Greedy, Spectral, Goemans-Williamson
│   ├── tsp/             # Nearest Neighbor, Christofides, 2-opt
│   └── coloring/        # Greedy, Welsh-Powell, DSatur
├── gnn/
│   ├── models/          # GIN for Max-Cut, TSP, and Coloring
│   └── training/        # Training loops (incl. REINFORCE)
├── evaluation/          # Metrics, timing, solution comparison
└── analysis/            # Failure prediction (balanced metrics)
experiments/             # Reproducible experiment runners
tests/                   # 27 verification tests
paper/                   # LaTeX write-up
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python3 -m pytest tests/ -v

python3 experiments/run_maxcut_comparison.py
python3 experiments/run_tsp_comparison.py
python3 experiments/run_coloring_comparison.py
python3 experiments/run_failure_analysis.py
```

## How to verify my work

The headline numbers are each tied to a committed artifact. Cheapest
first:

```bash
# 1. The 27 tests (fast, no GPU): classical guarantees, GNN shapes,
#    evaluation metrics
python3 -m pytest tests/ -v

# 2. The headline numbers live in committed artifacts — check directly
python3 - <<'EOF'
import json, pandas as pd
df = pd.read_csv("results/analysis/maxcut_comparison.csv")
print("GNN win rate:", f"{df['gnn_wins'].mean():.1%}")      # expect ~1.8%
print("GNN mean cut / best cut:", round(df["gnn_cut"].mean() /
      df["best_classical_cut"].mean(), 3))                  # expect ~0.98
                                                             # (per-instance ratio
                                                             # in the paper is 0.956)
j = json.load(open("analysis/figures/failure_prediction/prediction_results.json"))
print("balanced acc:", round(j["lr_balanced_accuracy"], 2)) # expect 0.52
EOF
```

| Headline claim | Artifact |
|---|---|
| GNN wins 1.8% of 600 Max-Cut instances | `results/analysis/maxcut_comparison.csv` (column `gnn_wins`) |
| TSP GNN collapses to nearest-neighbour | `results/analysis/tsp_gnn_weights.pt` + `tsp_comparison.csv` (embedding-cosine analysis in the paper) |
| DSatur beats GNN on coloring | `results/analysis/coloring_comparison.csv` |
| Failure prediction at chance (balanced acc 0.52) | `analysis/figures/failure_prediction/prediction_results.json` || Follow-up: supervised + 5× budget still loses (0.0%) | `results/analysis/supervised_maxcut_comparison.csv` |

To regenerate any artifact, run the corresponding step in
[Reproducing the paper](#reproducing-the-paper) — every experiment
trains its own GNN and evaluates on fresh held-out instances.

## Reproducing the paper

Everything in `paper/main.pdf` is reproducible from the committed
code and results. In order:

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Tests first (27 tests: classical guarantees, GNN shapes, metrics)
python3 -m pytest tests/ -v

# 2b. Follow-up experiment (supervised GNN, 5x budget) — optional, ~30 min
python3 experiments/run_supervised_maxcut.py
python3 experiments/make_followup_figure.py

# 3. Re-run the three comparisons (each trains a GNN, then evaluates
#    on held-out instances across all graph families; results land in
#    results/analysis/*.csv)
python3 experiments/run_maxcut_comparison.py
python3 experiments/run_tsp_comparison.py
python3 experiments/run_coloring_comparison.py

# 4. Failure-prediction analysis (reads the Max-Cut results CSV)
python3 experiments/run_failure_analysis.py

# 5. Rebuild the paper (LaTeX source in paper/main.tex)
tectonic paper/main.tex   # or: pdflatex paper/main.tex
```

The committed `results/analysis/*.csv` files are the outputs of steps
3–4 as run for the paper, so the figures and tables can be reproduced
without re-running the experiments. The TSP GNN weights are committed
(`results/analysis/tsp_gnn_weights.pt`) so the TSP collapse result
can be inspected directly.

## The maths behind the guarantees

**Goemans-Williamson (Max-Cut):** relax the integer program to an SDP
over PSD matrices, round via random hyperplanes. The 0.878 guarantee
comes from E[cut] = Σ w_ij·arccos(v_i·v_j)/π. (Falls back to spectral
with a warning when the SDP solver fails or n > max_n — the fallback
is flagged in the output, not silently.)

**Christofides (TSP):** MST ≤ OPT (a lower bound minus one edge), min
matching on odd vertices ≤ 0.5·OPT, combined ≤ 1.5·OPT.

**REINFORCE (TSP GNN):** the GNN produces node embeddings; a stochastic
policy samples the next city from softmax(embedding_similarity /
distance / temperature); the log-probability of the sampled tour times
the advantage gives the policy gradient. This is a proper policy
gradient — the log-probability comes from the actual stochastic
choices, not detached.

**DSatur (Coloring):** saturation-based vertex ordering produces
optimal 2-colorings on bipartite graphs and near-optimal colorings on
structured graphs.

## What I'd do differently

- ~~Give the GNN a serious training budget~~ **Done — it did not change the
  conclusion.** See the [follow-up experiment](#follow-up-supervised-training--5x-budget)
  below: supervised training at 5× the original budget still wins 0.0% of
  instances (relative 0.951 vs 0.956 unsupervised). The gap is not a
  training-budget artefact.
- ~~Try a supervised Max-Cut baseline instead of unsupervised only~~ **Done**
  (spectral-relaxation labels, see below).
- Run SDP at n up to the solver's real limit instead of a fixed
  max_n, and measure the fallback's effect on the speed comparison.
- Add Gurobi / exact solvers as an upper bound reference.

## Follow-up: supervised training + 5x budget

The paper's conclusion ("the GNN loses, and the loss is not a training
artefact") is only as strong as the training it was based on. This
follow-up answers the two caveats above directly:

1. **Supervised labels** from the spectral relaxation (a strong,
   cheap teacher that achieves ~97.8% of the best classical cut),
   instead of the original unsupervised cut-maximisation loss.
2. **5× the training budget**: 300 epochs (vs 80), plus cosine LR
   decay — the paper itself called the 80-epoch budget "modest".

Same evaluation protocol as the paper: held-out instances across five
graph families and three sizes (450 instances), vs random, greedy,
spectral, and Goemans-Williamson.

![Supervised vs unsupervised GNN relative performance](analysis/figures/followup/supervised_vs_unsupervised.png)

| Setting | Win rate | GNN / best classical |
|---|---|---|
| Unsupervised (paper) | 1.8% | 0.956 |
| **Supervised, 300 epochs (follow-up)** | **0.0%** | **0.951** |

**The conclusion holds.** A supervised training signal and a serious
budget do not close the gap — if anything the GNN is marginally
further behind (0.951 vs 0.956 relative), consistent with the paper's
failure-prediction result that its rare wins were noise, not a
recoverable signal.

Reproduce with:

```bash
python3 experiments/run_supervised_maxcut.py          # train + evaluate (~30 min)
python3 experiments/make_followup_figure.py           # figure above
```

Results: `results/analysis/supervised_maxcut_comparison.csv`,
training log `results/analysis/supervised_maxcut_training_log.csv`,
weights `results/analysis/supervised_maxcut_gnn_weights.pt`.

## What surprised me

The TSP collapse. I expected the GNN's learned embeddings to at least
give 2-opt a better starting point than nearest-neighbour. Instead,
after training, every pairwise embedding cosine was 1.0 — the GNN
learned to produce identical embeddings for all nodes. 2-opt then
erased even that. It was the cleanest "the model learned nothing
useful" result I've seen, and it's exactly why the comparison had to
be empirical.
