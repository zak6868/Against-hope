"""jlb — Exact correlated equilibrium for succinct games.

A reference implementation of Jiang & Leyton-Brown (2011)'s polynomial-
time CE algorithm for compactly-represented games.

Typical usage:

    from against_hope import pigou_game, run_jlb_ellipsoid

    g = pigou_game(n_agents=50)        # 2^50 ≈ 10^15 joint profiles
    result = run_jlb_ellipsoid(g, max_iterations=100_000)
    # result.profiles, result.alpha: sparse mixture CE

For custom succinct games, subclass `CompactGame` and implement
`utility(agent, profile)` and `polynomial_expectation(agent, a_i,
product_dist)`.  JLB will do the rest.
"""

from .types import GameSpec
from .game import NormalFormGame
from .compact_game import CompactGame, NormalFormCompactGame
from .congestion_game import CongestionGame, pigou_game, braess_game, sioux_falls_game
from .graphical_games import (
    GraphicalGame,
    graph_coloring_game,
    best_shot_public_goods,
    ring_majority_game,
    grid_adjacency,
    cycle_adjacency,
)
from .jlb_oracle import (
    jlb_separation_oracle,
    sample_pure_profile,
    build_product_distribution,
    cut_from_profile,
    OracleResult,
    y_dim,
)
from .jlb_ellipsoid import run_jlb_ellipsoid, EllipsoidJLBResult
from .jlb_driver import run_jlb, reconstruct_ce_mixture, JLBRunResult
from .verify import verify_ce, CEVerificationResult

__all__ = [
    "GameSpec",
    "NormalFormGame",
    "CompactGame",
    "NormalFormCompactGame",
    "CongestionGame",
    "pigou_game",
    "braess_game",
    "sioux_falls_game",
    "GraphicalGame",
    "graph_coloring_game",
    "best_shot_public_goods",
    "ring_majority_game",
    "grid_adjacency",
    "cycle_adjacency",
    "jlb_separation_oracle",
    "sample_pure_profile",
    "build_product_distribution",
    "cut_from_profile",
    "OracleResult",
    "y_dim",
    "run_jlb_ellipsoid",
    "EllipsoidJLBResult",
    "run_jlb",
    "reconstruct_ce_mixture",
    "JLBRunResult",
    "verify_ce",
    "CEVerificationResult",
]
