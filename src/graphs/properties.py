"""
Compute graph structural and spectral properties.

These features are used as inputs to the failure prediction model:
given a graph's properties, predict whether GNN or classical will win.
"""

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def compute_properties(G: nx.Graph) -> dict:
    """Compute a comprehensive feature vector for a graph."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]

    props = {
        "n": n,
        "m": m,
        "density": 2 * m / (n * (n - 1)) if n > 1 else 0,
        "avg_degree": np.mean(degrees),
        "max_degree": max(degrees),
        "min_degree": min(degrees),
        "degree_std": np.std(degrees),
    }

    props["clustering_coeff"] = nx.average_clustering(G)

    if nx.is_connected(G):
        props["diameter"] = nx.diameter(G) if n <= 500 else nx.approximation.diameter(G)
        props["avg_path_length"] = nx.average_shortest_path_length(G) if n <= 200 else -1
    else:
        props["diameter"] = -1
        props["avg_path_length"] = -1

    L = nx.laplacian_matrix(G).astype(float)
    if n <= 500:
        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L.toarray())))
    else:
        try:
            eigenvalues_small = eigsh(L, k=min(6, n - 2), which="SM", return_eigenvectors=False)
            eigenvalues_large = eigsh(L, k=min(3, n - 2), which="LM", return_eigenvectors=False)
            eigenvalues = np.sort(np.concatenate([eigenvalues_small, eigenvalues_large]))
        except Exception:
            eigenvalues = np.array([0.0])

    props["spectral_gap"] = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    props["lambda_max_laplacian"] = float(eigenvalues[-1]) if len(eigenvalues) > 0 else 0.0
    props["algebraic_connectivity"] = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0

    A = nx.adjacency_matrix(G).astype(float)
    if n <= 500:
        adj_eigs = np.sort(np.real(np.linalg.eigvalsh(A.toarray())))
        props["spectral_radius"] = float(adj_eigs[-1])
        props["lambda_min_adj"] = float(adj_eigs[0])
    else:
        try:
            sr = eigsh(A, k=1, which="LM", return_eigenvectors=False)
            props["spectral_radius"] = float(sr[0])
            sm = eigsh(A, k=1, which="SA", return_eigenvectors=False)
            props["lambda_min_adj"] = float(sm[0])
        except Exception:
            props["spectral_radius"] = 0.0
            props["lambda_min_adj"] = 0.0

    props["n_components"] = nx.number_connected_components(G)

    return props


def maxcut_spectral_bound(G: nx.Graph) -> float:
    """Lower bound on max-cut from spectral theory: max_cut >= N/4 * lambda_max(L)."""
    n = G.number_of_nodes()
    L = nx.laplacian_matrix(G).astype(float)
    if n <= 500:
        eigenvalues = np.linalg.eigvalsh(L.toarray())
    else:
        eigenvalues = eigsh(L, k=1, which="LM", return_eigenvectors=False)
    lambda_max = float(np.max(eigenvalues))
    return n / 4.0 * lambda_max
