"""Tests for TSP algorithms."""

import numpy as np
import pytest

from src.classical.tsp.nearest_neighbor import nearest_neighbor_tsp
from src.classical.tsp.christofides import christofides_tsp
from src.classical.tsp.two_opt import two_opt_improve


def square_instance():
    """4 cities at corners of unit square. Optimal tour = 4.0."""
    dist = np.array([
        [0, 1, np.sqrt(2), 1],
        [1, 0, 1, np.sqrt(2)],
        [np.sqrt(2), 1, 0, 1],
        [1, np.sqrt(2), 1, 0],
    ])
    return dist


class TestNearestNeighbor:
    def test_visits_all(self):
        dist = square_instance()
        tour, length = nearest_neighbor_tsp(dist)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}

    def test_tour_length_positive(self):
        dist = square_instance()
        tour, length = nearest_neighbor_tsp(dist)
        assert length > 0

    def test_finds_optimal_on_square(self):
        dist = square_instance()
        tour, length = nearest_neighbor_tsp(dist, start=0)
        assert abs(length - 4.0) < 1e-6


class TestChristofides:
    def test_visits_all(self):
        dist = square_instance()
        tour, length = christofides_tsp(dist)
        assert len(tour) == 4
        assert set(tour) == {0, 1, 2, 3}

    def test_guarantee_on_square(self):
        dist = square_instance()
        tour, length = christofides_tsp(dist)
        optimal = 4.0
        assert length <= 1.5 * optimal + 1e-6

    def test_larger_instance(self):
        rng = np.random.RandomState(42)
        coords = rng.rand(20, 2)
        dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
        tour, length = christofides_tsp(dist)
        assert len(tour) == 20
        assert len(set(tour)) == 20
        assert length > 0


class TestTwoOpt:
    def test_improves_or_equals(self):
        rng = np.random.RandomState(42)
        coords = rng.rand(15, 2)
        dist = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(axis=-1))
        nn_tour, nn_len = nearest_neighbor_tsp(dist)
        opt_tour, opt_len = two_opt_improve(nn_tour, dist)
        assert opt_len <= nn_len + 1e-6

    def test_valid_tour(self):
        dist = square_instance()
        tour, _ = nearest_neighbor_tsp(dist)
        opt_tour, _ = two_opt_improve(tour, dist)
        assert len(opt_tour) == 4
        assert set(opt_tour) == {0, 1, 2, 3}
