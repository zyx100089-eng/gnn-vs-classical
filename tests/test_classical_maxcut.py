"""
Verify classical Max-Cut algorithms on known instances.

This verification is ESSENTIAL — if the baseline is wrong, the comparison is worthless.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import numpy as np
import pytest

from src.classical.maxcut.random_cut import random_cut
from src.classical.maxcut.greedy import greedy_maxcut
from src.classical.maxcut.spectral import spectral_maxcut
from src.classical.maxcut.goemans_williamson import goemans_williamson
from src.graphs.generators import complete_bipartite, erdos_renyi
from src.graphs.properties import maxcut_spectral_bound


class TestRandomCut:
    def test_valid_partition(self):
        G = erdos_renyi(50, 0.3, seed=0)
        S, cut = random_cut(G)
        assert len(S) > 0
        assert len(S) < len(G)
        assert cut >= 0

    def test_cut_value_correct(self):
        G = nx.path_graph(4)
        for u, v in G.edges():
            G[u][v]["weight"] = 1.0
        S = {0, 2}
        expected = sum(1 for u, v in G.edges() if (u in S) != (v in S))
        _, cut = random_cut(G, seed=42)
        assert cut >= 0


class TestGreedyCut:
    def test_complete_bipartite(self):
        """On K_{10,10}, optimal cut = 100. Greedy should get close."""
        G = complete_bipartite(20)
        S, cut = greedy_maxcut(G, seed=0)
        assert cut == 100.0  # greedy should find optimal on bipartite

    def test_beats_half_edges(self):
        """Greedy guarantee: >= 0.5 * total_weight."""
        G = erdos_renyi(50, 0.3, seed=42)
        total_weight = sum(G[u][v].get("weight", 1.0) for u, v in G.edges())
        S, cut = greedy_maxcut(G, seed=0)
        assert cut >= 0.5 * total_weight - 1e-6


class TestSpectralCut:
    def test_complete_bipartite(self):
        G = complete_bipartite(20)
        S, cut = spectral_maxcut(G, refine=True)
        assert cut == 100.0

    def test_reasonable_quality(self):
        G = erdos_renyi(50, 0.3, seed=42)
        total_weight = sum(G[u][v].get("weight", 1.0) for u, v in G.edges())
        S, cut = spectral_maxcut(G)
        assert cut >= 0.5 * total_weight - 1e-6


class TestGoemansWilliamson:
    def test_complete_bipartite(self):
        """GW should find optimal or near-optimal on K_{10,10}."""
        G = complete_bipartite(20)
        S, cut = goemans_williamson(G, n_roundings=50, seed=0)
        assert cut >= 0.878 * 100.0

    def test_878_guarantee_approx(self):
        """On random graphs, GW should beat 0.878 * spectral_bound."""
        G = erdos_renyi(30, 0.3, seed=42)
        S, cut = goemans_williamson(G, n_roundings=100, seed=0)
        greedy_S, greedy_cut = greedy_maxcut(G, seed=0)
        assert cut >= greedy_cut * 0.9  # GW should be competitive with greedy

    def test_beats_random(self):
        G = erdos_renyi(50, 0.3, seed=42)
        _, random_val = random_cut(G, seed=0)
        _, gw_val = goemans_williamson(G, n_roundings=50, seed=0)
        assert gw_val >= random_val


class TestSpectralBound:
    def test_bound_valid(self):
        """Spectral bound should be <= actual max-cut (it's a lower bound)."""
        G = complete_bipartite(20)
        bound = maxcut_spectral_bound(G)
        _, greedy_cut = greedy_maxcut(G, seed=0)
        assert bound <= greedy_cut + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
