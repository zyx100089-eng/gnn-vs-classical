"""
Nearest Neighbor heuristic for TSP.

Greedy construction: start at a city, always go to the nearest unvisited city.
Guarantee: O(log n) * OPT for metric TSP (weak but very fast).
"""

import numpy as np


def nearest_neighbor_tsp(dist_matrix: np.ndarray, start: int = 0) -> tuple[list, float]:
    """
    Args:
        dist_matrix: (n, n) symmetric distance matrix
        start: starting city index

    Returns:
        tour: list of city indices (visiting order)
        length: total tour length
    """
    n = len(dist_matrix)
    visited = [False] * n
    tour = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = tour[-1]
        best_next = -1
        best_dist = float("inf")
        for j in range(n):
            if not visited[j] and dist_matrix[current][j] < best_dist:
                best_dist = dist_matrix[current][j]
                best_next = j
        tour.append(best_next)
        visited[best_next] = True

    length = sum(dist_matrix[tour[i]][tour[i + 1]] for i in range(n - 1))
    length += dist_matrix[tour[-1]][tour[0]]
    return tour, length
