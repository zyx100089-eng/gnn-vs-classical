"""
Random Max-Cut baseline.

Each node assigned to S or V\\S with equal probability.
Expected cut value = 0.5 * total_edge_weight.
"""

import networkx as nx
import numpy as np


def random_cut(G: nx.Graph, seed: int = 0) -> tuple[set, float]:
    """
    Returns:
        partition: set of nodes in S
        cut_value: total weight of edges crossing the partition
    """
    rng = np.random.RandomState(seed)
    S = {v for v in G.nodes() if rng.random() < 0.5}
    if len(S) == 0:
        S.add(list(G.nodes())[0])
    elif len(S) == len(G):
        S.remove(list(G.nodes())[0])
    cut_value = sum(
        G[u][v].get("weight", 1.0)
        for u, v in G.edges()
        if (u in S) != (v in S)
    )
    return S, cut_value
