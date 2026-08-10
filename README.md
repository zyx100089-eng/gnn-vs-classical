# GNNs vs. Proven Approximation Algorithms

A rigorous empirical audit comparing Graph Neural Network solvers against classical approximation algorithms with **proven guarantees** on NP-hard combinatorial optimization problems.

## The Question

GNNs are increasingly proposed as solvers for NP-hard problems. But classical algorithms like Goemans-Williamson (Max-Cut), Christofides (TSP), and DSatur (Coloring) come with **mathematical guarantees** on solution quality. Do learned solvers actually beat these guarantees?

**Short answer: No.** On 600 Max-Cut instances the GNN beats the best classical algorithm on only 1.8% of graphs. On TSP the GNN+2-opt matches NN+2-opt — the GNN initialisation does not add value over nearest-neighbour. Failure prediction from graph features is at chance level (balanced accuracy 0.52, barely above random 0.50), because the GNN wins so rarely there is almost no signal to learn.

## Key Findings

### Max-Cut: GNN wins 1.8% of 600 instances
- Goemans-Williamson SDP achieves the best cut on most instances
- GNN (with local search) beats the best classical algorithm on only 11/600 instances
- GNN achieves ~96% of the best classical cut on average — competitive but inferior
- GNN is significantly faster than SDP at large n
- **Failure prediction is at chance**: with only 1.8% positive examples, logistic regression achieves balanced accuracy 0.52 and F1 0.12 — both near the trivial baseline. The GNN wins too rarely to learn a useful predictor.

### TSP: GNN+2-opt ≈ NN+2-opt
- Christofides + 2-opt consistently outperforms GNN + 2-opt
- The GNN is trained with proper REINFORCE (policy gradient with log-probability of the sampled tour times the reward), but the learned embeddings do not produce better initial tours than nearest-neighbour
- 2-opt local search erases most of the initialisation difference

### Graph Coloring: GNN vs DSatur
- DSatur uses fewer colors on structured graphs
- GNN struggles with regular and small-world graphs

## Honest Assessment

The GNN does not beat classical algorithms with proven guarantees. The failure-prediction experiment confirms that the GNN's rare wins are not predictable from graph structure — they are essentially noise. The GNN's advantage is speed (inference is faster than SDP), not solution quality.

## Algorithms Implemented

### Classical (with guarantees)
| Algorithm | Problem | Guarantee |
|-----------|---------|-----------|
| Random partition | Max-Cut | ≥ 0.5 · OPT |
| Greedy local search | Max-Cut | ≥ 0.5 · OPT |
| Spectral relaxation | Max-Cut | Based on λ_max(L) |
| **Goemans-Williamson** | Max-Cut | **≥ 0.878 · OPT** |
| Nearest Neighbor | TSP | O(log n) · OPT |
| **Christofides** | TSP | **≤ 1.5 · OPT** |
| 2-opt local search | TSP | No guarantee |
| Greedy / Welsh-Powell | Coloring | ≤ Δ+1 colors |
| **DSatur** | Coloring | **Optimal on bipartite** |

### Learned (no guarantees)
| Algorithm | Problem | Architecture |
|-----------|---------|-------------|
| GIN (unsupervised) | Max-Cut | 5-layer GIN + local search |
| GIN (REINFORCE) | TSP | 5-layer GIN + policy gradient + greedy + 2-opt |
| GIN (classification) | Coloring | 5-layer GIN + conflict repair |

## Project Structure

```
src/
├── graphs/              # 6 graph family generators + spectral properties
├── classical/
│   ├── maxcut/          # Random, Greedy, Spectral, Goemans-Williamson
│   ├── tsp/             # Nearest Neighbor, Christofides, 2-opt
│   └── coloring/        # Greedy, Welsh-Powell, DSatur
├── gnn/models/          # GIN for Max-Cut, TSP, and Coloring
├── gnn/training/        # Training loops
├── evaluation/          # Metrics, timing, solution comparison
└── analysis/            # Failure prediction (balanced metrics)
experiments/             # Reproducible experiment runners
tests/                   # 27 verification tests
paper/                   # LaTeX paper
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run comparisons
python experiments/run_maxcut_comparison.py
python experiments/run_tsp_comparison.py
python experiments/run_coloring_comparison.py

# Run failure prediction analysis
python experiments/run_failure_analysis.py
```

## Mathematical Foundation

**Goemans-Williamson (Max-Cut):** Relax integer program to SDP over PSD matrices, round via random hyperplanes. The 0.878 guarantee comes from E[cut] = Σ w_ij · arccos(v_i · v_j) / π. (Falls back to spectral with a warning when the SDP solver fails or n > max_n.)

**Christofides (TSP):** MST ≤ OPT (lower bound minus one edge). Min matching on odd vertices ≤ 0.5 · OPT. Combined: ≤ 1.5 · OPT.

**REINFORCE (TSP GNN):** The GNN produces node embeddings. A stochastic policy samples the next city from softmax(embedding_similarity / distance / temperature). The log-probability of the sampled tour times the advantage (reward minus baseline) gives the policy gradient. This is a proper policy gradient — the log-probability is computed from the actual stochastic choices, not detached.

**DSatur (Coloring):** Saturation-based vertex ordering produces optimal 2-colorings on bipartite graphs and near-optimal colorings on structured graphs.