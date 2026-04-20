"""Verify JLB outputs a genuine CE — max constraint violation ≤ float tol.

A JLB run that claims to have "found a CE" must actually produce a
mixture that satisfies every single CE constraint of the game up to
float precision.  This test closes the loop: take the output, evaluate
every (agent, a_i, a'_i) constraint, assert no strict violation.
"""

from __future__ import annotations

import numpy as np
import pytest

from against_hope import (
    CongestionGame,
    pigou_game,
    braess_game,
    sioux_falls_game,
    graph_coloring_game,
    best_shot_public_goods,
    ring_majority_game,
    grid_adjacency,
    cycle_adjacency,
    run_jlb_ellipsoid,
    verify_ce,
)


CONGESTION_CASES = [
    ("pigou_5", lambda: pigou_game(n_agents=5)),
    ("pigou_20", lambda: pigou_game(n_agents=20)),
    ("braess_5", lambda: braess_game(n_agents=5, free_shortcut=True)),
    ("sfgrid3_10", lambda: sioux_falls_game(n_agents=10, grid_size=3, k_routes=3)),
]


GRAPHICAL_CASES = [
    (
        "coloring_grid3x3",
        lambda: graph_coloring_game(grid_adjacency(3, 3), n_colors=3),
    ),
    (
        "best_shot_cycle10",
        lambda: best_shot_public_goods(cycle_adjacency(10)),
    ),
    ("ring_majority_10", lambda: ring_majority_game(n=10, m=2)),
]


@pytest.mark.parametrize("label,game_fn", CONGESTION_CASES)
def test_ce_exact_on_congestion(label: str, game_fn) -> None:
    """JLB-computed mixture must be an exact CE on congestion games."""
    g = game_fn()
    res = run_jlb_ellipsoid(
        g,
        max_iterations=100_000,
        initial_radius=10.0,
        reconstruct_every=50,
    )
    assert res.reconstructed, f"{label}: JLB did not reconstruct a CE"
    v = verify_ce(g, res.profiles, res.alpha, tol=1e-6)
    assert v.is_ce, (
        f"{label}: CE constraint violated. "
        f"max_violation={v.max_violation:.3e} "
        f"at (agent={v.worst_violation[0]}, "
        f"a_i={v.worst_violation[1]}, a'_i={v.worst_violation[2]}). "
        f"#violated = {v.n_violated}"
    )


@pytest.mark.parametrize("label,game_fn", GRAPHICAL_CASES)
def test_ce_exact_on_graphical(label: str, game_fn) -> None:
    """JLB-computed mixture must be an exact CE on graphical games."""
    g = game_fn()
    res = run_jlb_ellipsoid(
        g,
        max_iterations=100_000,
        initial_radius=10.0,
        reconstruct_every=50,
    )
    assert res.reconstructed, f"{label}: JLB did not reconstruct a CE"
    v = verify_ce(g, res.profiles, res.alpha, tol=1e-6)
    assert v.is_ce, (
        f"{label}: CE constraint violated. "
        f"max_violation={v.max_violation:.3e} "
        f"at (agent={v.worst_violation[0]}, "
        f"a_i={v.worst_violation[1]}, a'_i={v.worst_violation[2]})."
    )


def test_verify_ce_catches_non_ce() -> None:
    """Sanity: a deliberately-bad mixture should be flagged as non-CE."""
    g = pigou_game(n_agents=3)
    # Recommend "everyone uses edge 0" (the high-cost congested edge).
    # Each agent would rather deviate to edge 1.
    bad_profile = (0, 0, 0)
    v = verify_ce(g, [bad_profile], np.array([1.0]), tol=1e-8)
    assert not v.is_ce, "expected bad mixture to fail CE check"
    assert v.max_violation > 0
