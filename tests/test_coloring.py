"""Tests for graph coloring algorithms."""

import networkx as nx
import pytest

from src.classical.coloring.greedy import greedy_coloring, welsh_powell_coloring
from src.classical.coloring.dsatur import dsatur_coloring, verify_coloring


class TestGreedyColoring:
    def test_bipartite_two_colors(self):
        G = nx.complete_bipartite_graph(5, 5)
        _, n_colors = welsh_powell_coloring(G)
        assert n_colors == 2

    def test_complete_graph(self):
        G = nx.complete_graph(10)
        coloring, n_colors = greedy_coloring(G)
        assert n_colors == 10
        assert verify_coloring(G, coloring)

    def test_valid_coloring(self):
        G = nx.erdos_renyi_graph(50, 0.3, seed=42)
        coloring, _ = greedy_coloring(G)
        assert verify_coloring(G, coloring)


class TestWelshPowell:
    def test_valid(self):
        G = nx.erdos_renyi_graph(50, 0.3, seed=42)
        coloring, _ = welsh_powell_coloring(G)
        assert verify_coloring(G, coloring)

    def test_at_most_delta_plus_one(self):
        G = nx.erdos_renyi_graph(50, 0.3, seed=42)
        _, n_colors = welsh_powell_coloring(G)
        delta = max(d for _, d in G.degree())
        assert n_colors <= delta + 1


class TestDSatur:
    def test_bipartite_optimal(self):
        G = nx.complete_bipartite_graph(5, 5)
        coloring, n_colors = dsatur_coloring(G)
        assert n_colors == 2
        assert verify_coloring(G, coloring)

    def test_cycle_graph(self):
        G = nx.cycle_graph(6)  # Even cycle -> 2-colorable
        coloring, n_colors = dsatur_coloring(G)
        assert n_colors == 2
        assert verify_coloring(G, coloring)

    def test_odd_cycle(self):
        G = nx.cycle_graph(5)  # Odd cycle -> 3 colors
        coloring, n_colors = dsatur_coloring(G)
        assert n_colors == 3
        assert verify_coloring(G, coloring)

    def test_beats_greedy(self):
        """DSatur should use <= greedy colors on average."""
        G = nx.erdos_renyi_graph(100, 0.2, seed=42)
        _, dsatur_c = dsatur_coloring(G)
        _, greedy_c = greedy_coloring(G)
        assert dsatur_c <= greedy_c
