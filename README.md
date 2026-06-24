# GNNs vs. Proven Approximation Algorithms

A rigorous empirical audit comparing Graph Neural Network solvers against classical approximation algorithms with **proven guarantees** on NP-hard combinatorial optimization problems.

## The Question

GNNs are increasingly proposed as solvers for NP-hard problems. But classical algorithms like Goemans-Williamson (Max-Cut) and Christofides (TSP) come with **mathematical guarantees** on solution quality. Do learned solvers actually beat these guarantees?

## Key Findings

### Max-Cut: GNN vs Goemans-Williamson (0.878-approximation)

- **In-distribution** (Erdős-Rényi): GNN is competitive at small sizes
- **Out-of-distribution** (other graph families): GNN degrades significantly
- **Spectral gap predicts GNN failure**: graphs with high spectral gap are harder for GNNs
- **Runtime crossover**: GNN inference becomes faster than SDP around n=200

## Algorithms Implemented

### Classical (with guarantees)
| Algorithm | Problem | Guarantee |
|-----------|---------|-----------|
| Random partition | Max-Cut | ≥ 0.5 · OPT |
| Greedy local search | Max-Cut | ≥ 0.5 · OPT |
| Spectral relaxation | Max-Cut | Based on λ_max(L) |
| **Goemans-Williamson** | Max-Cut | **≥ 0.878 · OPT** |

### Learned (no guarantees)
| Algorithm | Problem | Architecture |
|-----------|---------|-------------|
| GIN (unsupervised) | Max-Cut | Graph Isomorphism Network |

## Project Structure

```
src/
├── graphs/           # 6 graph family generators + spectral properties
├── classical/maxcut/ # 4 classical algorithms (random, greedy, spectral, GW)
├── gnn/             # GIN model + unsupervised training
├── evaluation/      # Metrics, timing, solution comparison
└── analysis/        # Failure prediction, performance surfaces
experiments/         # Reproducible experiment runners
tests/              # Verification on known instances
paper/              # LaTeX paper
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run all tests (verify classical algorithms on known instances)
python -m pytest tests/ -v

# Run the Max-Cut comparison
python experiments/run_maxcut_comparison.py --sizes "20,50,100" --instances 30
```

## Mathematical Foundation

The Goemans-Williamson algorithm is the deepest linear algebra result:
1. Relax the integer program to a **semidefinite program** (SDP)
2. Solve over **positive semidefinite matrices** (the PSD cone)
3. Round via **random hyperplane** in the SDP solution space
4. The 0.878 factor comes from E[cut] = Σ w_ij · arccos(v_i · v_j) / π

## Citation

If you use this code, please cite our work.
