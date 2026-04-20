"""Tests for CongestionGame.

Every method is cross-validated against either:
  - hand-computed values on Pigou/Braess networks, or
  - brute-force enumeration on the equivalent normal-form game.

This module must pass before we can trust the polynomial expectation
oracle in downstream JLB tests.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from against_hope.congestion_game import CongestionGame, pigou_game, braess_game
from against_hope.compact_game import NormalFormCompactGame
from against_hope.fixtures.ground_truth import expected_utility_brute_force


# ---------------------------------------------------------------------------
# Game construction sanity
# ---------------------------------------------------------------------------


class TestPigou:
    def test_construction(self):
        g = pigou_game(n_agents=3, a=0.0, b=1.0, const_route_cost=1.0)
        assert g.spec.n == 3
        assert g.spec.action_counts == (2, 2, 2)
        assert g.spec.M == 8
        assert g.n_edges == 2

    def test_utility_hand_computed_all_on_edge0(self):
        """If all 3 agents take edge 0 (costing l), each gets -3."""
        g = pigou_game(n_agents=3)
        profile = (0, 0, 0)
        for i in range(3):
            assert g.utility(i, profile) == -3.0  # load = 3, cost = 3

    def test_utility_hand_computed_mixed(self):
        """Two on edge 0 (load 2, each cost 2); one on edge 1 (constant cost 1)."""
        g = pigou_game(n_agents=3, a=0.0, b=1.0, const_route_cost=1.0)
        profile = (0, 0, 1)
        assert g.utility(0, profile) == -2.0
        assert g.utility(1, profile) == -2.0
        assert g.utility(2, profile) == -1.0

    def test_normal_form_roundtrip(self):
        """Converting to NormalFormGame preserves all utilities."""
        g = pigou_game(n_agents=3)
        nf = g.as_normal_form()
        assert nf.spec.action_counts == g.spec.action_counts
        # Sample some profiles
        for profile in product(*[range(2)] * 3):
            for i in range(3):
                assert np.isclose(nf.utilities[i][profile], g.utility(i, profile)), (
                    f"mismatch at {profile}, agent {i}"
                )


class TestBraess:
    def test_construction(self):
        g = braess_game(n_agents=2, free_shortcut=True)
        assert g.spec.n == 2
        assert g.spec.action_counts == (3, 3)
        assert g.n_edges == 5

    def test_no_shortcut_has_two_routes(self):
        g = braess_game(n_agents=2, free_shortcut=False)
        assert g.spec.action_counts == (2, 2)

    def test_utility_equilibrium_flow(self):
        """Without the shortcut, Braess network with 2 agents has both
        agents choosing different routes; each gets cost 1 (constant edge) + 1 (load 1)."""
        g = braess_game(n_agents=2, free_shortcut=False)
        # Agent 0 on S-A-T (edges 0, 2), agent 1 on S-B-T (edges 1, 3):
        #   edge 0: load 1, cost 1
        #   edge 2: load 1, cost 1
        #   edge 1: load 1, cost 1
        #   edge 3: load 1, cost 1
        # Agent 0 total cost = 1 + 1 = 2
        # Agent 1 total cost = 1 + 1 = 2
        profile = (0, 1)
        assert g.utility(0, profile) == -2.0
        assert g.utility(1, profile) == -2.0


# ---------------------------------------------------------------------------
# Polynomial expectation oracle: compare to brute-force enumeration
# ---------------------------------------------------------------------------


class TestPolynomialExpectationCongestion:
    @pytest.mark.parametrize(
        "game_fn,label",
        [
            (lambda: pigou_game(n_agents=2), "Pigou-2"),
            (lambda: pigou_game(n_agents=3), "Pigou-3"),
            (lambda: pigou_game(n_agents=4), "Pigou-4"),
            (lambda: braess_game(n_agents=2, free_shortcut=False), "Braess-2-noshort"),
            (lambda: braess_game(n_agents=3, free_shortcut=True), "Braess-3-full"),
        ],
    )
    def test_expectation_matches_brute_force(self, game_fn, label):
        """CongestionGame.polynomial_expectation must match brute-force
        enumeration of the equivalent normal-form game under every product
        distribution we throw at it."""
        g = game_fn()
        nf = g.as_normal_form()

        rng = np.random.default_rng(hash(label) % (2**31))
        for trial in range(10):
            # Random product distribution
            prod = [rng.dirichlet(np.ones(m)) for m in g.spec.action_counts]
            for i in range(g.spec.n):
                for a_i in range(g.spec.action_counts[i]):
                    # Expectation via compact oracle
                    ours = g.polynomial_expectation(i, a_i, prod)
                    # Brute force: replace agent i with point mass, enumerate
                    prod_cond = [p.copy() for p in prod]
                    prod_cond[i] = np.zeros(g.spec.action_counts[i])
                    prod_cond[i][a_i] = 1.0
                    bf = expected_utility_brute_force(nf, i, prod_cond)
                    assert np.isclose(ours, bf, atol=1e-12), (
                        f"{label} trial {trial} agent {i} action {a_i}: "
                        f"oracle={ours}, brute-force={bf}"
                    )

    def test_deterministic_agrees_with_utility(self):
        """Point-mass product distribution gives deterministic utility."""
        g = pigou_game(n_agents=3)
        for profile in product(*[range(2)] * 3):
            prod = [np.zeros(2) for _ in range(3)]
            for i, a_i in enumerate(profile):
                prod[i][a_i] = 1.0
            for i in range(3):
                u = g.polynomial_expectation(i, profile[i], prod)
                assert np.isclose(u, g.utility(i, profile), atol=1e-12), (
                    f"profile {profile}, agent {i}"
                )


# ---------------------------------------------------------------------------
# edge_use_probability invariants
# ---------------------------------------------------------------------------


class TestEdgeUseProbability:
    def test_all_probability_accounted_for(self):
        """For each agent, sum over edges of P[edge used] >= 1 in general
        (an edge can be used by multiple routes).  At minimum, each non-empty
        route contributes probability, so per edge the quantity is in [0, 1]."""
        g = pigou_game(n_agents=3)
        x = [np.array([0.3, 0.7]), np.array([1.0, 0.0]), np.array([0.5, 0.5])]
        for i in range(3):
            for e in range(g.n_edges):
                p = g.edge_use_probability(i, x, e)
                assert 0.0 <= p <= 1.0 + 1e-14

    def test_point_mass(self):
        """If x_i is a point mass on a route, the edge use probability is 1 for
        edges on that route and 0 otherwise."""
        g = pigou_game(n_agents=1)
        # Agent 0 plays route 0 (edge 0) with probability 1.
        x = [np.array([1.0, 0.0])]
        assert g.edge_use_probability(0, x, 0) == 1.0
        assert g.edge_use_probability(0, x, 1) == 0.0
