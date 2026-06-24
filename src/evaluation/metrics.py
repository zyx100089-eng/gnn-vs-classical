"""
Evaluation metrics for comparing GNN vs classical solvers.
"""

import time
import networkx as nx
from typing import Callable


def evaluate_solver(solver_fn: Callable, G: nx.Graph,
                    **kwargs) -> dict:
    """
    Run a solver and measure quality + runtime.

    Args:
        solver_fn: function(G, **kwargs) -> (partition, cut_value)
        G: input graph

    Returns:
        dict with partition, cut_value, runtime
    """
    start = time.perf_counter()
    S, cut_value = solver_fn(G, **kwargs)
    elapsed = time.perf_counter() - start

    return {
        "partition": S,
        "cut_value": cut_value,
        "runtime": elapsed,
    }


def compute_approximation_ratio(solution_value: float,
                                optimal_value: float) -> float:
    """Approximation ratio (1.0 = optimal)."""
    if optimal_value <= 0:
        return 1.0
    return solution_value / optimal_value


def solution_jaccard(S1: set, S2: set, n: int) -> float:
    """Jaccard similarity between two partitions (accounting for complement symmetry)."""
    S1_comp = set(range(n)) - S1
    j1 = len(S1 & S2) / len(S1 | S2) if len(S1 | S2) > 0 else 0
    j2 = len(S1_comp & S2) / len(S1_comp | S2) if len(S1_comp | S2) > 0 else 0
    return max(j1, j2)
