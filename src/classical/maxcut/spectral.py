"""
Spectral relaxation for Max-Cut.

Uses the Fiedler vector (eigenvector of the second-smallest eigenvalue
of the Laplacian) to partition the graph. Then refines with local search.

Not as strong as Goemans-Williamson but much faster and still based on
the spectral structure of the graph Laplacian.
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh


def _cut_value(G: nx.Graph, S: set) -> float:
    return sum(
        G[u][v].get("weight", 1.0)
        for u, v in G.edges()
        if (u in S) != (v in S)
    )


def spectral_maxcut(G: nx.Graph, refine: bool = True) -> tuple[set, float]:
    """
    Partition using the largest eigenvector of the Laplacian.

    For Max-Cut, the relevant eigenvector is the one corresponding to
    the LARGEST eigenvalue of L (not the Fiedler vector).
    max_cut >= N/4 * lambda_max(L), achieved approximately by
    thresholding the max eigenvector at 0.
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    L = nx.laplacian_matrix(G).astype(float)

    if n <= 500:
        eigenvalues, eigenvectors = np.linalg.eigh(L.toarray())
        v_max = eigenvectors[:, -1]
    else:
        eigenvalues, eigenvectors = eigsh(L, k=1, which="LM")
        v_max = eigenvectors[:, 0]

    S = {nodes[i] for i in range(n) if v_max[i] >= 0}
    if len(S) == 0:
        S.add(nodes[0])
    elif len(S) == n:
        S.remove(nodes[0])

    if refine:
        improved = True
        while improved:
            improved = False
            for v in nodes:
                in_S = v in S
                gain = 0.0
                for u in G.neighbors(v):
                    w = G[v][u].get("weight", 1.0)
                    if (u in S) == in_S:
                        gain += w
                    else:
                        gain -= w
                if gain > 1e-10:
                    if in_S:
                        S.remove(v)
                    else:
                        S.add(v)
                    improved = True

    return S, _cut_value(G, S)
