"""
Christofides-Serdyukov algorithm for metric TSP (1976).

The best known polynomial-time approximation: <= 1.5 * OPT.

Algorithm:
1. Compute MST of the complete graph
2. Find vertices with odd degree in the MST
3. Compute minimum-weight perfect matching on odd-degree vertices
4. Combine MST + matching to get an Eulerian multigraph
5. Find an Eulerian tour
6. Shortcut repeated vertices to get a Hamiltonian tour

The 1.5 guarantee comes from: MST <= OPT (MST is a lower bound minus one edge),
and min matching on odd vertices <= 0.5 * OPT (by a parity argument).
"""

import numpy as np
import networkx as nx


def christofides_tsp(dist_matrix: np.ndarray) -> tuple[list, float]:
    """
    Args:
        dist_matrix: (n, n) symmetric distance matrix (metric)

    Returns:
        tour: list of city indices
        length: total tour length
    """
    n = len(dist_matrix)

    # Step 1: Build complete graph and compute MST
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=dist_matrix[i][j])

    mst = nx.minimum_spanning_tree(G)

    # Step 2: Find odd-degree vertices
    odd_vertices = [v for v in mst.nodes() if mst.degree(v) % 2 == 1]

    # Step 3: Minimum-weight perfect matching on odd-degree vertices
    # Build complete subgraph on odd vertices
    odd_graph = nx.Graph()
    for i, u in enumerate(odd_vertices):
        for j, v in enumerate(odd_vertices):
            if i < j:
                odd_graph.add_edge(u, v, weight=dist_matrix[u][v])

    matching = nx.min_weight_matching(odd_graph)

    # Step 4: Combine MST + matching into multigraph
    multigraph = nx.MultiGraph(mst)
    for u, v in matching:
        multigraph.add_edge(u, v, weight=dist_matrix[u][v])

    # Step 5: Find Eulerian circuit
    euler_circuit = list(nx.eulerian_circuit(multigraph, source=0))

    # Step 6: Shortcut to Hamiltonian tour
    visited = set()
    tour = []
    for u, v in euler_circuit:
        if u not in visited:
            tour.append(u)
            visited.add(u)
    if len(tour) < n:
        for v in range(n):
            if v not in visited:
                tour.append(v)

    length = sum(dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))
    return tour, length
