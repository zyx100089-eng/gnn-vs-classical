"""
Greedy local search for Max-Cut.

Start from a random partition, then repeatedly move the node that
increases the cut the most, until no improvement is possible.
Guarantee: >= 0.5 * OPT.
"""

import networkx as nx
import numpy as np


def _cut_value(G: nx.Graph, S: set) -> float:
    return sum(
        G[u][v].get("weight", 1.0)
        for u, v in G.edges()
        if (u in S) != (v in S)
    )


def _node_gain(G: nx.Graph, S: set, v: int) -> float:
    """Change in cut value if v is moved to the other side."""
    gain = 0.0
    in_S = v in S
    for u in G.neighbors(v):
        w = G[v][u].get("weight", 1.0)
        if (u in S) == in_S:
            gain += w
        else:
            gain -= w
    return gain


def greedy_maxcut(G: nx.Graph, seed: int = 0, max_iter: int = 1000) -> tuple[set, float]:
    rng = np.random.RandomState(seed)
    nodes = list(G.nodes())

    S = set()
    for v in nodes:
        if rng.random() < 0.5:
            S.add(v)
    if len(S) == 0:
        S.add(nodes[0])

    for _ in range(max_iter):
        improved = False
        rng.shuffle(nodes)
        for v in nodes:
            gain = _node_gain(G, S, v)
            if gain > 1e-10:
                if v in S:
                    S.remove(v)
                else:
                    S.add(v)
                improved = True
        if not improved:
            break

    return S, _cut_value(G, S)
