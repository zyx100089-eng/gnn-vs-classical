"""
Greedy graph coloring algorithms.

Greedy (random order): assigns smallest available color to each vertex.
Welsh-Powell: processes vertices in decreasing degree order.
Both guarantee at most Delta+1 colors (Delta = max degree).
"""

import networkx as nx
import numpy as np


def greedy_coloring(G: nx.Graph, seed: int = 0) -> tuple[dict, int]:
    """
    Greedy coloring in random order.

    Returns:
        coloring: dict mapping node -> color (0-indexed)
        n_colors: number of colors used
    """
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())
    rng.shuffle(nodes)

    coloring = {}
    for v in nodes:
        neighbor_colors = {coloring[u] for u in G.neighbors(v) if u in coloring}
        color = 0
        while color in neighbor_colors:
            color += 1
        coloring[v] = color

    n_colors = max(coloring.values()) + 1 if coloring else 0
    return coloring, n_colors


def welsh_powell_coloring(G: nx.Graph) -> tuple[dict, int]:
    """
    Welsh-Powell: greedy coloring in decreasing degree order.
    Often better than random-order greedy.
    """
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v), reverse=True)

    coloring = {}
    for v in nodes:
        neighbor_colors = {coloring[u] for u in G.neighbors(v) if u in coloring}
        color = 0
        while color in neighbor_colors:
            color += 1
        coloring[v] = color

    n_colors = max(coloring.values()) + 1 if coloring else 0
    return coloring, n_colors
