"""
Graph generators for 6 families used in the GNN vs Classical comparison.

Each generator returns a networkx Graph with optional edge weights.
All graphs are undirected and connected (we retry if disconnected).
"""

import networkx as nx
import numpy as np
from typing import Optional


def _ensure_connected(G: nx.Graph, max_retries: int = 100) -> nx.Graph:
    """If G is disconnected, connect components with minimum-weight edges."""
    if nx.is_connected(G):
        return G
    components = list(nx.connected_components(G))
    for i in range(1, len(components)):
        u = list(components[i - 1])[0]
        v = list(components[i])[0]
        G.add_edge(u, v, weight=1.0)
    return G


def erdos_renyi(n: int, p: float = 0.15, seed: Optional[int] = None,
                weighted: bool = False) -> nx.Graph:
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    G = _ensure_connected(G)
    if weighted:
        rng = np.random.RandomState(seed)
        for u, v in G.edges():
            G[u][v]["weight"] = rng.uniform(0.1, 1.0)
    else:
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0
    return G


def planted_partition(n: int, k: int = 3, p_in: float = 0.5,
                      p_out: float = 0.05, seed: Optional[int] = None) -> nx.Graph:
    """Stochastic block model with k equal-sized clusters."""
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    G = _ensure_connected(G)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    G.graph.pop("partition", None)
    return G


def random_regular(n: int, d: int = 3, seed: Optional[int] = None) -> nx.Graph:
    if d >= n:
        d = n - 1 if n % 2 == 0 or (n - 1) % 2 == 0 else n - 2
    if (n * d) % 2 != 0:
        d = d - 1 if d > 2 else d + 1
    G = nx.random_regular_graph(d, n, seed=seed)
    G = _ensure_connected(G)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


def barabasi_albert(n: int, m: int = 3, seed: Optional[int] = None) -> nx.Graph:
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


def watts_strogatz(n: int, k: int = 6, p_rewire: float = 0.1,
                   seed: Optional[int] = None) -> nx.Graph:
    G = nx.watts_strogatz_graph(n, k, p_rewire, seed=seed)
    G = _ensure_connected(G)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


def complete_bipartite(n: int) -> nx.Graph:
    """K_{n/2, n/2} — useful for verification (known optimal max-cut = n^2/4)."""
    half = n // 2
    G = nx.complete_bipartite_graph(half, n - half)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


GRAPH_FAMILIES = {
    "erdos_renyi": erdos_renyi,
    "planted_partition": planted_partition,
    "random_regular": random_regular,
    "barabasi_albert": barabasi_albert,
    "watts_strogatz": watts_strogatz,
}


def generate_instance(family: str, n: int, seed: int = 0, **kwargs) -> nx.Graph:
    """Generate a single graph instance."""
    gen = GRAPH_FAMILIES[family]
    G = gen(n, seed=seed, **kwargs)
    G.graph["family"] = family
    G.graph["n"] = n
    G.graph["seed"] = seed
    return G


def generate_batch(family: str, n: int, count: int = 100,
                   base_seed: int = 0, **kwargs) -> list[nx.Graph]:
    """Generate a batch of graph instances."""
    return [generate_instance(family, n, seed=base_seed + i, **kwargs)
            for i in range(count)]
