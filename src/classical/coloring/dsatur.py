"""
DSatur algorithm for graph coloring (Brelaz, 1979).

DSatur = Degree of Saturation. At each step, color the vertex with
the highest saturation degree (number of distinct colors among neighbors).
Ties broken by highest uncolored degree.

DSatur is optimal for bipartite graphs and often near-optimal in practice.
Much better than simple greedy on structured graphs.
"""

import networkx as nx


def dsatur_coloring(G: nx.Graph) -> tuple[dict, int]:
    """
    DSatur graph coloring.

    Returns:
        coloring: dict mapping node -> color (0-indexed)
        n_colors: number of colors used
    """
    n = G.number_of_nodes()
    coloring = {}
    saturation = {v: set() for v in G.nodes()}

    for _ in range(n):
        # Pick uncolored vertex with highest saturation, break ties by degree
        best_v = None
        best_sat = -1
        best_deg = -1
        for v in G.nodes():
            if v in coloring:
                continue
            sat = len(saturation[v])
            deg = G.degree(v)
            if sat > best_sat or (sat == best_sat and deg > best_deg):
                best_v = v
                best_sat = sat
                best_deg = deg

        # Assign smallest available color
        neighbor_colors = {coloring[u] for u in G.neighbors(best_v) if u in coloring}
        color = 0
        while color in neighbor_colors:
            color += 1
        coloring[best_v] = color

        # Update saturation of neighbors
        for u in G.neighbors(best_v):
            if u not in coloring:
                saturation[u].add(color)

    n_colors = max(coloring.values()) + 1 if coloring else 0
    return coloring, n_colors


def verify_coloring(G: nx.Graph, coloring: dict) -> bool:
    """Check that no adjacent vertices share a color."""
    for u, v in G.edges():
        if coloring.get(u) == coloring.get(v):
            return False
    return True
