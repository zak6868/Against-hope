"""Benchmark games for JLB validation.

All games here are constructed with integer utilities so that CE computation
can be verified in exact rational arithmetic.  Each game is documented with
its known CE (where easy to write down) and structural properties used in
tests.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from against_hope.game import NormalFormGame


def prisoners_dilemma() -> NormalFormGame:
    """Classical prisoner's dilemma (2 players, 2 actions each).

    Actions: 0 = Cooperate, 1 = Defect.
    Payoff matrices (row = player 1, col = player 2):
        u_1[a_1, a_2] = payoff to player 1
        u_2[a_1, a_2] = payoff to player 2

    Unique CE = Nash equilibrium = (Defect, Defect) = (1, 1).
    """
    # u_1[row, col] where row = player 1's action, col = player 2's action.
    u_1 = np.array([[3, 0],
                    [5, 1]], dtype=np.float64)
    u_2 = np.array([[3, 5],
                    [0, 1]], dtype=np.float64)
    return NormalFormGame((2, 2), [u_1, u_2])


def matching_pennies() -> NormalFormGame:
    """Zero-sum matching pennies (2 players, 2 actions each).

    Unique Nash (and CE) = uniform (1/2, 1/2) for each player.
    """
    u_1 = np.array([[1, -1],
                    [-1, 1]], dtype=np.float64)
    u_2 = -u_1
    return NormalFormGame((2, 2), [u_1, u_2])


def shapley_game() -> NormalFormGame:
    """Shapley's classical 3x3 game.

    MWU dynamics cycle on this game; converging to CE is non-trivial.
    """
    u_1 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]], dtype=np.float64)
    u_2 = np.array([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]], dtype=np.float64)
    return NormalFormGame((3, 3), [u_1, u_2])


def chicken_game() -> NormalFormGame:
    """Chicken / Hawk-Dove (2 players, 2 actions each).

    Actions: 0 = Swerve, 1 = Straight.
    Two pure NE: (Swerve, Straight) and (Straight, Swerve).
    CE polytope is non-trivial.
    """
    u_1 = np.array([[3, 2],
                    [4, 0]], dtype=np.float64)
    u_2 = np.array([[3, 4],
                    [2, 0]], dtype=np.float64)
    return NormalFormGame((2, 2), [u_1, u_2])


def theorem5_counterexample_u() -> NormalFormGame:
    """Game u from Theorem 5 of the main paper (4x2, player 1 utilities u_1).

    See recommend_unknown_games/tests/test_all.py TestTheorem5Counterexample.
    Non-equivalent to v but BR-indistinguishable from it.
    """
    u_1 = np.array([[0, 8], [3, 6.5], [5, 4.5], [8, 0]], dtype=np.float64)
    u_2 = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.3, 0.7]], dtype=np.float64)
    return NormalFormGame((4, 2), [u_1, u_2])


def theorem5_counterexample_v() -> NormalFormGame:
    """Game v from Theorem 5 of the main paper (4x2, player 1 utilities v_1).

    Shares player 2 utilities with game u but differs in player 1 utilities.
    """
    v_1 = np.array([[0, 8], [2, 7], [6, 3], [8, 0]], dtype=np.float64)
    u_2 = np.array([[1, 0], [0, 1], [0.5, 0.5], [0.3, 0.7]], dtype=np.float64)
    return NormalFormGame((4, 2), [v_1, u_2])


def all_benchmark_games() -> dict[str, NormalFormGame]:
    """Return a name -> game mapping for parametrized tests."""
    return {
        "prisoners_dilemma": prisoners_dilemma(),
        "matching_pennies": matching_pennies(),
        "shapley": shapley_game(),
        "chicken": chicken_game(),
        "theorem5_u": theorem5_counterexample_u(),
        "theorem5_v": theorem5_counterexample_v(),
    }


__all__ = [
    "prisoners_dilemma",
    "matching_pennies",
    "shapley_game",
    "chicken_game",
    "theorem5_counterexample_u",
    "theorem5_counterexample_v",
    "all_benchmark_games",
]
