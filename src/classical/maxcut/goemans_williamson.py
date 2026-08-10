"""
Goemans-Williamson algorithm for Max-Cut (1995).

The best known polynomial-time approximation:  >= 0.878 * OPT
(optimal under the Unique Games Conjecture).

Algorithm:
1. Formulate Max-Cut as an integer program: max (1/4) sum w_ij(1 - x_i*x_j), x_i in {-1,+1}
2. Relax to SDP: replace scalars x_i with unit vectors v_i in R^n
   max (1/4) sum w_ij(1 - v_i . v_j)  s.t. ||v_i|| = 1
   Equivalently: max (1/4) <W_L, X>  s.t. diag(X) = 1, X >= 0 (PSD)
3. Solve the SDP using CVXPY
4. Extract vectors via Cholesky decomposition of X
5. Random hyperplane rounding: sample r ~ N(0, I), set x_i = sign(v_i . r)
6. Take the best of multiple roundings

The linear algebra is deep:
- X is a positive semidefinite Gram matrix
- The SDP relaxes the rank-1 constraint (X = xx^T) to X >= 0
- The 0.878 factor comes from arccos: E[cut] = sum w_ij * arccos(v_i.v_j) / pi
"""

import networkx as nx
import numpy as np
import cvxpy as cp


def _cut_value(G: nx.Graph, S: set) -> float:
    return sum(
        G[u][v].get("weight", 1.0)
        for u, v in G.edges()
        if (u in S) != (v in S)
    )


def goemans_williamson(G: nx.Graph, n_roundings: int = 100,
                       seed: int = 0, max_n: int = 300) -> tuple[set, float]:
    """
    Full Goemans-Williamson with SDP + random hyperplane rounding.

    For large graphs (n > max_n), falls back to spectral relaxation
    because the SDP becomes too slow.

    Args:
        G: input graph
        n_roundings: number of random hyperplane roundings to try
        seed: random seed
        max_n: maximum graph size for SDP (larger graphs use fallback)

    Returns:
        S: partition (set of nodes)
        cut_value: weight of edges crossing the cut
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}

    if n > max_n:
        import warnings
        warnings.warn(
            f"Graph has {n} nodes > max_n={max_n}; falling back to spectral. "
            "GW results for this graph are actually spectral.",
            stacklevel=2,
        )
        from .spectral import spectral_maxcut
        return spectral_maxcut(G, refine=True)

    # Build weight matrix
    W = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        w = data.get("weight", 1.0)
        W[i, j] = w
        W[j, i] = w

    # SDP: maximize (1/4) * <L_w, X> subject to diag(X) = 1, X PSD
    X = cp.Variable((n, n), symmetric=True)

    L_w = np.diag(W.sum(axis=1)) - W

    objective = cp.Maximize(0.25 * cp.trace(L_w @ X))
    constraints = [
        X >> 0,  # PSD
        cp.diag(X) == 1,  # unit vectors
    ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
    except cp.SolverError:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=10000, eps=1e-6)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        import warnings
        warnings.warn(
            f"SDP solver status={prob.status}; falling back to spectral. "
            "GW results for this graph are actually spectral.",
            stacklevel=2,
        )
        from .spectral import spectral_maxcut
        return spectral_maxcut(G, refine=True)

    X_val = X.value
    X_val = (X_val + X_val.T) / 2

    # Project onto PSD cone (clip negative eigenvalues)
    eigenvalues, eigenvectors = np.linalg.eigh(X_val)
    eigenvalues = np.maximum(eigenvalues, 0)
    X_val = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Cholesky-like decomposition to get vectors
    eigenvalues_sqrt = np.sqrt(eigenvalues + 1e-10)
    V = eigenvectors * eigenvalues_sqrt[np.newaxis, :]  # n x n, each row is v_i

    # Random hyperplane rounding (multiple attempts, take best)
    rng = np.random.RandomState(seed)
    best_S = None
    best_cut = -1

    for _ in range(n_roundings):
        r = rng.randn(n)
        r = r / np.linalg.norm(r)

        signs = V @ r
        S = {nodes[i] for i in range(n) if signs[i] >= 0}
        if len(S) == 0:
            S.add(nodes[0])
        elif len(S) == n:
            S.remove(nodes[0])

        cut = _cut_value(G, S)
        if cut > best_cut:
            best_cut = cut
            best_S = S

    return best_S, best_cut
