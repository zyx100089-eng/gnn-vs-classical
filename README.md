# GNNs vs. Proven Approximation Algorithms

A rigorous empirical audit comparing Graph Neural Network solvers against classical approximation algorithms with **proven guarantees** on NP-hard combinatorial optimization problems.

## The Question

GNNs are increasingly proposed as solvers for NP-hard problems. But classical algorithms like Goemans-Williamson (Max-Cut), Christofides (TSP), and DSatur (Coloring) come with **mathematical guarantees** on solution quality. Do learned solvers actually beat these guarantees?

**Short answer: No.** But GNNs are much faster.

## Key Findings

### Max-Cut: GNN wins 2.2% of 600 instances
- Goemans-Williamson SDP achieves 99.9% of best across all families
- GNN achieves 95.6% of best — competitive but inferior
- GNN is 60× faster than SDP at n=200
- Spectral features predict GNN failure with 88% accuracy

### TSP: GNN vs Christofides (1.5-approximation)
- Christofides + 2-opt consistently outperforms GNN + 2-opt
- GNN provides competitive initialization but not better solutions

### Graph Coloring: GNN vs DSatur
- DSatur uses fewer colors on structured graphs
- GNN struggles with regular and small-world graphs

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
| GIN (embedding) | TSP | 5-layer GIN + greedy + 2-opt |
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
└── analysis/            # Failure prediction
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

**Goemans-Williamson (Max-Cut):** Relax integer program to SDP over PSD matrices, round via random hyperplanes. The 0.878 guarantee comes from E[cut] = Σ w_ij · arccos(v_i · v_j) / π.

**Christofides (TSP):** MST ≤ OPT (lower bound minus one edge). Min matching on odd vertices ≤ 0.5 · OPT. Combined: ≤ 1.5 · OPT.

**DSatur (Coloring):** Saturation-based vertex ordering produces optimal 2-colorings on bipartite graphs and near-optimal colorings on structured graphs.
