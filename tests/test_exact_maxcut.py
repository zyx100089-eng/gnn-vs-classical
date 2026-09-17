"""Tests for the exact Max-Cut solver (brute force, n <= 20)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx

from src.classical.maxcut.exact import exact_maxcut


def test_complete_graph():
    # K_n: the best cut separates floor(n/2) from ceil(n/2) vertices.
    for n in range(2, 9):
        G = nx.complete_graph(n)
        assert exact_maxcut(G) == (n // 2) * (n - n // 2)


def test_odd_cycle():
    # C_5: max cut is n - 1 = 4 (all but one edge).
    G = nx.cycle_graph(5)
    assert exact_maxcut(G) == 4


def test_complete_bipartite():
    # K_{a,b}: the natural bipartition cuts every edge.
    for a, b in [(1, 1), (2, 3), (4, 4), (3, 7)]:
        G = nx.complete_bipartite_graph(a, b)
        assert exact_maxcut(G) == a * b


def test_edgeless_graph():
    G = nx.empty_graph(6)
    assert exact_maxcut(G) == 0


def test_matches_bruteforce_enumeration():
    # Independent, deliberately naive enumeration for n = 14.
    import itertools
    import random
    rng = random.Random(0)
    for _ in range(8):
        n = 14
        G = nx.gnp_random_graph(n, 0.4, seed=rng.randrange(10**6))
        nodes = list(G.nodes())
        best = 0
        for subset in itertools.product([0, 1], repeat=n):
            side = {nodes[i] for i, s in enumerate(subset) if s}
            cut = sum(1 for u, v in G.edges() if (u in side) != (v in side))
            best = max(best, cut)
        assert exact_maxcut(G) == best


def test_too_large_returns_nan():
    import math
    G = nx.path_graph(21)
    assert math.isnan(exact_maxcut(G))


def test_weighted():
    # Two parallel paths of weights 1 and 3; best cut takes the heavier.
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(1, 2, weight=3.0)
    assert exact_maxcut(G) == 4.0
