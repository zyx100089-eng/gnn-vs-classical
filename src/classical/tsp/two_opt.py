"""
2-opt local search for TSP.

Repeatedly reverses segments of the tour to reduce length.
No approximation guarantee, but excellent in practice.
"""

import numpy as np


def two_opt_improve(tour: list, dist_matrix: np.ndarray,
                    max_iter: int = 1000) -> tuple[list, float]:
    """
    Improve a tour using 2-opt swaps.

    A 2-opt swap reverses the segment between positions i and j.
    If reversing reduces tour length, accept the swap.
    """
    n = len(tour)
    tour = list(tour)

    def tour_length(t):
        return sum(dist_matrix[t[i]][t[(i + 1) % n]] for i in range(n))

    best_length = tour_length(tour)

    for _ in range(max_iter):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # Cost of current edges
                old_cost = (dist_matrix[tour[i - 1]][tour[i]] +
                           dist_matrix[tour[j]][tour[(j + 1) % n]])
                # Cost after reversing segment [i, j]
                new_cost = (dist_matrix[tour[i - 1]][tour[j]] +
                           dist_matrix[tour[i]][tour[(j + 1) % n]])

                if new_cost < old_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    best_length -= (old_cost - new_cost)
                    improved = True
        if not improved:
            break

    return tour, tour_length(tour)
