"""Exact maximum cut by full enumeration, for small graphs.

Used to turn relative comparisons ("solver X beat the other solvers I
ran") into absolute ones ("solver X found the true optimum"). Only
tractable for small n; callers should restrict to n <= 20.
"""

import numpy as np


def exact_maxcut(G):
    """Exact maximum cut value of `G`, or nan if n > 20.

    Enumerates all 2^n partitions. Vectorised over the 2^n masks so the
    cost is one pass over an (2^n, |E|) boolean matrix per instance.
    """
    n = G.number_of_nodes()
    if n > 20:
        return float("nan")
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v], d.get("weight", 1.0))
             for u, v, d in G.edges(data=True)]
    if not edges:
        return 0.0
    w = np.array([e[2] for e in edges], dtype=float)
    us = np.array([e[0] for e in edges], dtype=np.uint32)
    vs = np.array([e[1] for e in edges], dtype=np.uint32)
    masks = np.arange(1 << n, dtype=np.uint32)
    bu = ((masks[:, None] >> us[None, :]) & 1).astype(bool)
    bv = ((masks[:, None] >> vs[None, :]) & 1).astype(bool)
    cuts = (bu != bv) @ w
    return float(cuts.max())
