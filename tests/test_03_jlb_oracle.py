"""Tests for the JLB separation oracle.

The oracle has two high-level responsibilities whose correctness must
be verified independently:

1. **Papadimitriou-Roughgarden Lemma 3.1**: for any y >= 0, the
   constructed product distribution x satisfies E_x[f(a)] = 0, where
   f(a) = (U^T y)_a.

2. **JLB 2011 correction**: the pure profile a* selected by the
   method of conditional expectations satisfies f(a*) >= 0, providing
   a valid cut with bounded encoding size.
"""

from __future__ import annotations

import numpy as np
import pytest

from against_hope.compact_game import NormalFormCompactGame
from against_hope.jlb_oracle import (
    build_markov_chain,
    stationary_distribution,
    build_product_distribution,
    evaluate_f_pure,
    expected_f_under_product,
    sample_pure_profile,
    cut_from_profile,
    jlb_separation_oracle,
    y_dim,
    y_block,
    pack_y,
)
from against_hope.fixtures.benchmark_games import (
    prisoners_dilemma,
    matching_pennies,
    shapley_game,
    chicken_game,
)


# ---------------------------------------------------------------------------
# Markov chain structure
# ---------------------------------------------------------------------------


class TestMarkovChain:
    def test_row_stochastic(self):
        """Every row of P_i sums to 1."""
        game = shapley_game()
        N = y_dim(game.spec)
        rng = np.random.default_rng(0)
        for _ in range(20):
            y = rng.uniform(0, 1, size=N)
            for i in range(game.spec.n):
                P = build_markov_chain(y, game.spec, i)
                row_sums = P.sum(axis=1)
                np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_zero_row_stays(self):
        """A row with zero y entries keeps mass at itself."""
        game = prisoners_dilemma()
        N = y_dim(game.spec)
        y = np.zeros(N)
        P = build_markov_chain(y, game.spec, 0)
        # With all zeros, every row stays put
        np.testing.assert_array_equal(P, np.eye(2))

    def test_nonnegative(self):
        """All entries of P_i are non-negative."""
        game = shapley_game()
        N = y_dim(game.spec)
        rng = np.random.default_rng(1)
        for _ in range(10):
            y = rng.uniform(0, 1, size=N)
            for i in range(game.spec.n):
                P = build_markov_chain(y, game.spec, i)
                assert np.all(P >= 0.0)


class TestStationaryDistribution:
    def test_fixed_point(self):
        """pi @ P = pi exactly, sum(pi) = 1."""
        game = shapley_game()
        rng = np.random.default_rng(2)
        for _ in range(20):
            y = rng.uniform(0.1, 1.0, size=y_dim(game.spec))
            for i in range(game.spec.n):
                P = build_markov_chain(y, game.spec, i)
                pi = stationary_distribution(P)
                np.testing.assert_allclose(pi @ P, pi, atol=1e-10)
                assert np.isclose(pi.sum(), 1.0, atol=1e-12)
                assert np.all(pi >= -1e-14)

    def test_identity(self):
        """For P = I, any distribution is stationary; we return uniform."""
        P = np.eye(3)
        pi = stationary_distribution(P)
        assert np.isclose(pi.sum(), 1.0, atol=1e-12)
        assert np.all(pi >= 0)


# ---------------------------------------------------------------------------
# Central Papadimitriou-Roughgarden invariant: E_x[f] = 0
# ---------------------------------------------------------------------------


class TestPRLemma31:
    """For any y >= 0, E_{a ~ x}[f(a)] = 0 where f = U^T y and x is
    the product distribution from the Markov chains."""

    @pytest.mark.parametrize(
        "game_fn,label,n_trials",
        [
            (prisoners_dilemma, "PD", 50),
            (matching_pennies, "MP", 50),
            (chicken_game, "Chicken", 50),
            (shapley_game, "Shapley", 50),
        ],
    )
    def test_expected_f_is_zero(self, game_fn, label, n_trials):
        game = NormalFormCompactGame(game_fn())
        rng = np.random.default_rng(hash(label) % (2**31))
        for t in range(n_trials):
            y = rng.uniform(0, 1, size=y_dim(game.spec))
            prod = build_product_distribution(y, game.spec)
            e = expected_f_under_product(game, y, prod)
            assert abs(e) < 1e-8, f"{label} trial {t}: E_x[f] = {e}"

    def test_zero_y_gives_zero_f(self):
        """y = 0 makes f identically zero; any product distribution works."""
        game = NormalFormCompactGame(prisoners_dilemma())
        y = np.zeros(y_dim(game.spec))
        prod = build_product_distribution(y, game.spec)
        e = expected_f_under_product(game, y, prod)
        assert abs(e) < 1e-14


# ---------------------------------------------------------------------------
# JLB 2011 correction: cut is valid (f(a*) >= 0)
# ---------------------------------------------------------------------------


class TestJLBCutValidity:
    """Every pure profile returned must satisfy f(a*) >= 0."""

    @pytest.mark.parametrize(
        "game_fn,label,n_trials",
        [
            (prisoners_dilemma, "PD", 100),
            (matching_pennies, "MP", 100),
            (chicken_game, "Chicken", 100),
            (shapley_game, "Shapley", 100),
        ],
    )
    def test_cut_is_valid(self, game_fn, label, n_trials):
        game = NormalFormCompactGame(game_fn())
        rng = np.random.default_rng((hash(label) + 1) % (2**31))
        for t in range(n_trials):
            y = rng.uniform(0, 1, size=y_dim(game.spec))
            result = jlb_separation_oracle(game, y)
            # cut^T y = f(a*) must be >= 0 (up to tiny float noise)
            assert result.f_value >= -1e-10, (
                f"{label} trial {t}: f(a*) = {result.f_value} < 0"
            )

    def test_cut_matches_evaluate_f(self):
        """f_value returned by the oracle equals evaluate_f_pure(a*)."""
        game = NormalFormCompactGame(shapley_game())
        rng = np.random.default_rng(7)
        for _ in range(20):
            y = rng.uniform(0, 1, size=y_dim(game.spec))
            result = jlb_separation_oracle(game, y)
            f_direct = evaluate_f_pure(game, y, result.profile)
            assert np.isclose(result.f_value, f_direct, atol=1e-12)

    def test_profile_in_support(self):
        """Selected pure profile lies in the support of the product distribution."""
        game = NormalFormCompactGame(shapley_game())
        rng = np.random.default_rng(8)
        for _ in range(20):
            y = rng.uniform(0.5, 1.0, size=y_dim(game.spec))
            result = jlb_separation_oracle(game, y)
            for i, a_i in enumerate(result.profile):
                p = float(result.product_dist[i][a_i])
                assert p > 0, f"profile agent {i} action {a_i} has zero probability"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_y_same_profile(self):
        """Calling the oracle twice with the same y yields the same profile."""
        game = NormalFormCompactGame(shapley_game())
        rng = np.random.default_rng(100)
        y = rng.uniform(0, 1, size=y_dim(game.spec))
        r1 = jlb_separation_oracle(game, y)
        r2 = jlb_separation_oracle(game, y)
        assert r1.profile == r2.profile
        np.testing.assert_allclose(r1.cut, r2.cut, atol=1e-14)


# ---------------------------------------------------------------------------
# Cut construction
# ---------------------------------------------------------------------------


class TestCutConstruction:
    def test_cut_dimension(self):
        game = NormalFormCompactGame(prisoners_dilemma())
        q = cut_from_profile(game, (0, 0))
        assert q.shape == (y_dim(game.spec),)

    def test_cut_zero_on_other_agent_actions(self):
        """Entry (i, a_i, a'_i) of q is zero if profile[i] != a_i."""
        game = NormalFormCompactGame(prisoners_dilemma())
        profile = (0, 1)
        q = cut_from_profile(game, profile)
        # Agent 0 plays 0, so the block for agent 0, row a_i=1 must be 0.
        Q0 = y_block(q, game.spec, 0)
        assert np.all(Q0[1, :] == 0)
        # Agent 1 plays 1, so the block for agent 1, row a_i=0 must be 0.
        Q1 = y_block(q, game.spec, 1)
        assert np.all(Q1[0, :] == 0)

    def test_cut_hand_value(self):
        """Hand-compute the cut for (D, D) in Prisoner's Dilemma.

        u_1 = [[3, 0], [5, 1]]:   at profile (1, 1),
            Q0[1, 0] = u_1(1, 1) - u_1(0, 1) = 1 - 0 = 1
        u_2 = [[3, 5], [0, 1]]:   at profile (1, 1),
            Q1[1, 0] = u_2(1, 1) - u_2(1, 0) = 1 - 0 = 1
        """
        game = NormalFormCompactGame(prisoners_dilemma())
        q = cut_from_profile(game, (1, 1))
        Q0 = y_block(q, game.spec, 0)
        Q1 = y_block(q, game.spec, 1)
        assert Q0[1, 0] == 1.0
        assert Q1[1, 0] == 1.0

    def test_cut_hand_value_mp(self):
        """Hand-compute the cut for (0, 0) in Matching Pennies.

        u_1 = [[1, -1], [-1, 1]]:  at profile (0, 0),
            Q0[0, 1] = u_1(0, 0) - u_1(1, 0) = 1 - (-1) = 2
        u_2 = [[-1, 1], [1, -1]]:  at profile (0, 0),
            Q1[0, 1] = u_2(0, 0) - u_2(0, 1) = -1 - 1 = -2
        """
        game = NormalFormCompactGame(matching_pennies())
        q = cut_from_profile(game, (0, 0))
        Q0 = y_block(q, game.spec, 0)
        Q1 = y_block(q, game.spec, 1)
        assert Q0[0, 1] == 2.0
        assert Q1[0, 1] == -2.0
