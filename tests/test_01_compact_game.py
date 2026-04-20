"""Tests for the compact-game abstraction.

The NormalFormCompactGame wrapper is the ground-truth implementation of
the two oracles (utility, polynomial_expectation); its values are checked
against hand-computed or brute-force values.
"""

from __future__ import annotations

import numpy as np
import pytest

from against_hope.compact_game import NormalFormCompactGame
from against_hope.fixtures.benchmark_games import (
    prisoners_dilemma,
    matching_pennies,
    shapley_game,
)
from against_hope.fixtures.ground_truth import expected_utility_brute_force


# ---------------------------------------------------------------------------
# Utility lookup
# ---------------------------------------------------------------------------


class TestUtilityLookup:
    def test_pd_utility(self):
        g = NormalFormCompactGame(prisoners_dilemma())
        assert g.utility(0, (0, 0)) == 3.0  # (C, C)
        assert g.utility(0, (1, 0)) == 5.0  # (D, C)
        assert g.utility(1, (1, 0)) == 0.0  # from P2's perspective


# ---------------------------------------------------------------------------
# Polynomial expectation correctness (ground truth vs brute force)
# ---------------------------------------------------------------------------


class TestPolynomialExpectation:
    @pytest.mark.parametrize(
        "game_fn,label",
        [
            (prisoners_dilemma, "PD"),
            (matching_pennies, "MP"),
            (shapley_game, "Shapley"),
        ],
    )
    def test_deterministic_product(self, game_fn, label):
        """Point-mass distributions give exact deterministic utility."""
        game = game_fn()
        cg = NormalFormCompactGame(game)
        n = game.spec.n
        m_each = game.spec.action_counts
        # Put all mass on (0, 0, ..., 0).
        prod_dist = [np.zeros(m_each[i]) for i in range(n)]
        for i in range(n):
            prod_dist[i][0] = 1.0
        for i in range(n):
            u_i = cg.polynomial_expectation(i, a_i=0, product_dist=prod_dist)
            expected = float(game.utilities[i][(0,) * n])
            assert u_i == expected, f"{label}: agent {i}: got {u_i}, expected {expected}"

    @pytest.mark.parametrize(
        "game_fn,label",
        [
            (prisoners_dilemma, "PD"),
            (matching_pennies, "MP"),
            (shapley_game, "Shapley"),
        ],
    )
    def test_against_brute_force(self, game_fn, label):
        """Compare polynomial_expectation to brute-force enumeration."""
        game = game_fn()
        cg = NormalFormCompactGame(game)
        rng = np.random.default_rng(42)
        n = game.spec.n
        m = game.spec.action_counts

        for trial in range(20):
            # Random product distribution
            prod = [rng.dirichlet(np.ones(m[i])) for i in range(n)]
            for i in range(n):
                for a_i in range(m[i]):
                    # Our polynomial expectation is E_{a_{-i}}[u_i(a_i, a_{-i})]
                    ours = cg.polynomial_expectation(i, a_i, prod)

                    # Brute force: replace prod[i] with point mass on a_i, then compute E
                    prod_cond = [p.copy() for p in prod]
                    prod_cond[i] = np.zeros(m[i])
                    prod_cond[i][a_i] = 1.0
                    bf = expected_utility_brute_force(game, i, prod_cond)

                    assert np.isclose(ours, bf, atol=1e-12), (
                        f"{label} trial {trial} agent {i} action {a_i}: "
                        f"ours={ours}, brute-force={bf}"
                    )

    def test_zero_marginal_fallback(self):
        """If x_i(a_i) = 0, we still compute the right expectation via fallback."""
        game = prisoners_dilemma()
        cg = NormalFormCompactGame(game)
        # Agent 0 never plays 0 but we still ask for E[u_0 | a_0 = 0]
        prod = [
            np.array([0.0, 1.0]),  # agent 0 always plays 1
            np.array([0.3, 0.7]),  # agent 1 mixed
        ]
        # Expected: E_{a_1}[u_0(0, a_1)] = 0.3 * 3 + 0.7 * 0 = 0.9
        u = cg.polynomial_expectation(agent=0, a_i=0, product_dist=prod)
        assert np.isclose(u, 0.9, atol=1e-12)


class TestUtilityDiffExpectation:
    def test_self_diff_zero(self):
        game = prisoners_dilemma()
        cg = NormalFormCompactGame(game)
        prod = [np.array([0.4, 0.6]), np.array([0.2, 0.8])]
        for i in range(2):
            for a in range(2):
                d = cg.utility_diff_expectation(i, a, a, prod)
                assert d == 0.0

    def test_antisymmetry(self):
        game = prisoners_dilemma()
        cg = NormalFormCompactGame(game)
        prod = [np.array([0.4, 0.6]), np.array([0.2, 0.8])]
        d1 = cg.utility_diff_expectation(agent=0, a_i=0, a_prime_i=1, product_dist=prod)
        d2 = cg.utility_diff_expectation(agent=0, a_i=1, a_prime_i=0, product_dist=prod)
        assert np.isclose(d1, -d2, atol=1e-12)
