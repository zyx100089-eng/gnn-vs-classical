"""Exact maximum cut by full enumeration, for small graphs.

Used to turn relative comparisons ("solver X beat the other solvers I
ran") into absolute ones ("solver X found the true optimum"). Only
tractable for small n; callers should restrict to n <= 20.
"""

import numpy as np


def exact_maxcut_partition(G):
    """Exact maximum cut of `G` as ``(partition, value)``; n <= 20 only.

    Enumerates all 2^n partitions, vectorised over the masks so the cost
    is one pass over an (2^n, |E|) boolean matrix per instance.
    """
    n = G.number_of_nodes()
    if n > 20:
        raise ValueError(f"exact_maxcut_partition is limited to n <= 20 (got {n})")
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v], d.get("weight", 1.0))
             for u, v, d in G.edges(data=True)]
    if not edges:
        return set(), 0.0
    w = np.array([e[2] for e in edges], dtype=float)
    us = np.array([e[0] for e in edges], dtype=np.uint32)
    vs = np.array([e[1] for e in edges], dtype=np.uint32)
    masks = np.arange(1 << n, dtype=np.uint32)
    bu = ((masks[:, None] >> us[None, :]) & 1).astype(bool)
    bv = ((masks[:, None] >> vs[None, :]) & 1).astype(bool)
    cuts = (bu != bv) @ w
    best = int(cuts.argmax())
    S = {nodes[i] for i in range(n) if (best >> i) & 1}
    # Canonicalise the global sign: a cut and its complement are the same
    # cut, so always return the orientation with the smallest node outside
    # S. Without this the returned partition flips arbitrarily with the
    # integer-mask ordering (node 0 was inside S ~54% of the time).
    if nodes[0] in S:
        S = set(nodes) - S
    return S, float(cuts[best])


def exact_maxcut(G):
    """Exact maximum cut value of `G`, or nan if n > 20."""
    if G.number_of_nodes() > 20:
        return float("nan")
    return exact_maxcut_partition(G)[1]

