"""Graphical games — classical succinct representation beyond congestion.

A graphical game (Kearns, Littman, Singh 2001) has agents on a graph
G = (V, E) where each agent's utility depends only on their OWN action
and the actions of their neighbors.  The game is described by one local
utility table per agent of size `m × m^{deg(i)+1}` — polynomial in the
graph size when the maximum degree is bounded.

Pure Nash is PPAD-hard for graphical games in general (Kearns 2007), so
no "Nash shortcut" exists.  But the polynomial expectation property
holds — each agent's expected utility can be computed as a sum over
their neighbors' joint distribution — so JLB applies directly.

This module provides a generic GraphicalGame class and three classical
games as factories:

    - `graph_coloring_game`      : each node chooses a color; reward
                                   increases with the number of neighbors
                                   with a different color (interference /
                                   frequency-assignment).

    - `best_shot_public_goods`   : each node decides to contribute or
                                   free-ride on a public good; reward
                                   depends on whether *any* neighbor
                                   contributes.

    - `ring_majority_game`       : n agents on a cycle; reward = 1 if
                                   your action matches the majority of
                                   your two neighbours, else 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from numpy.typing import NDArray

from .compact_game import CompactGame
from .types import GameSpec


@dataclass
class GraphicalGame(CompactGame):
    """Graphical game on an arbitrary directed graph of dependencies.

    Attributes
    ----------
    n
        Number of agents.
    m
        Number of actions per agent (uniform).
    neighbours
        neighbours[i] is a tuple of agent indices that agent i's utility
        depends on (apart from i itself).  May include i; we treat the
        "own action" axis separately for clarity.
    local_utilities
        local_utilities[i] is an ndarray whose shape is
        (m,) + tuple(m for _ in neighbours[i])
        and `local_utilities[i][a_i, a_{neighbours[i][0]}, …]` gives
        `u_i(a_i, a_{N(i)})`.
    """

    n: int
    m: int
    neighbours: list[tuple[int, ...]]
    local_utilities: list[NDArray[np.float64]]
    spec: GameSpec = field(init=False)

    def __post_init__(self) -> None:
        self.spec = GameSpec.from_action_counts(tuple([self.m] * self.n))
        for i, (nbr, U) in enumerate(
            zip(self.neighbours, self.local_utilities)
        ):
            expected_shape = (self.m,) + tuple(self.m for _ in nbr)
            assert U.shape == expected_shape, (
                f"agent {i}: local utility shape {U.shape} != {expected_shape}"
            )

    # ----- abstract methods -----

    def utility(self, agent: int, profile: tuple[int, ...]) -> float:
        a_i = profile[agent]
        nbr_actions = tuple(profile[j] for j in self.neighbours[agent])
        return float(self.local_utilities[agent][(a_i,) + nbr_actions])

    def polynomial_expectation(
        self,
        agent: int,
        a_i: int,
        product_dist: list[NDArray[np.float64]],
    ) -> float:
        """E_{a_N(i) ~ x_N(i)}[u_i(a_i, a_N(i))].

        Computed by enumerating the neighbours' joint actions — cheap
        because |N(i)| is bounded.
        """
        nbr = self.neighbours[agent]
        U = self.local_utilities[agent]
        total = 0.0
        for nbr_actions in product(*[range(self.m) for _ in nbr]):
            p = 1.0
            for k, j in enumerate(nbr):
                p *= float(product_dist[j][nbr_actions[k]])
                if p == 0.0:
                    break
            if p == 0.0:
                continue
            total += p * float(U[(a_i,) + nbr_actions])
        return total


# ----------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------


def graph_coloring_game(
    adjacency: list[tuple[int, ...]],
    n_colors: int = 3,
    collision_cost: float = 1.0,
) -> GraphicalGame:
    """Graph-coloring interference game.

    Each agent picks one of `n_colors`.  Their utility is the number of
    neighbours who picked a DIFFERENT color — so they prefer to avoid
    collisions.  This is the canonical frequency-assignment / channel-
    selection game.

    `adjacency[i]` is the tuple of neighbours of i (undirected — should
    be symmetric).
    """
    n = len(adjacency)
    m = n_colors
    neighbours = [tuple(adjacency[i]) for i in range(n)]
    local_utilities: list[NDArray[np.float64]] = []
    for i in range(n):
        deg = len(neighbours[i])
        U = np.zeros((m,) + tuple(m for _ in range(deg)), dtype=np.float64)
        for a_i in range(m):
            for nbr_actions in product(*[range(m) for _ in range(deg)]):
                reward = sum(
                    1.0 for aj in nbr_actions if aj != a_i
                ) - collision_cost * sum(
                    1.0 for aj in nbr_actions if aj == a_i
                )
                U[(a_i,) + nbr_actions] = reward
        local_utilities.append(U)
    return GraphicalGame(
        n=n, m=m,
        neighbours=neighbours,
        local_utilities=local_utilities,
    )


def best_shot_public_goods(
    adjacency: list[tuple[int, ...]],
    contribution_cost: float = 0.5,
    public_benefit: float = 1.0,
) -> GraphicalGame:
    """Best-shot public goods on a graph.

    Each agent chooses to contribute (1) or free-ride (0).
    - If at least one neighbour (including self) contributes, every
      agent in that neighbourhood gets `public_benefit`.
    - Contributing costs `contribution_cost`.

    Classic free-riding dilemma — who installs the fire alarm in the
    neighbourhood?  Pure Nash = some independent set contributes, the
    rest free-ride.  CE can correlate contributions fairly.
    """
    n = len(adjacency)
    m = 2
    neighbours = [tuple(adjacency[i]) for i in range(n)]
    local_utilities: list[NDArray[np.float64]] = []
    for i in range(n):
        deg = len(neighbours[i])
        U = np.zeros((m,) + tuple(m for _ in range(deg)), dtype=np.float64)
        for a_i in range(m):
            for nbr_actions in product(*[range(m) for _ in range(deg)]):
                any_contributes = a_i == 1 or any(a == 1 for a in nbr_actions)
                reward = (public_benefit if any_contributes else 0.0) - (
                    contribution_cost if a_i == 1 else 0.0
                )
                U[(a_i,) + nbr_actions] = reward
        local_utilities.append(U)
    return GraphicalGame(
        n=n, m=m,
        neighbours=neighbours,
        local_utilities=local_utilities,
    )


def ring_majority_game(n: int, m: int = 2) -> GraphicalGame:
    """n agents on a cycle; each wants to match the *majority* of their
    two cycle neighbours (including self).

    Used as a simple social-learning / opinion-dynamics benchmark.  Has
    many Nashes (e.g., everyone playing the same action) and a rich CE
    polytope.
    """
    neighbours = [((i - 1) % n, (i + 1) % n) for i in range(n)]
    local_utilities: list[NDArray[np.float64]] = []
    for i in range(n):
        U = np.zeros((m, m, m), dtype=np.float64)
        for a_i in range(m):
            for a_left in range(m):
                for a_right in range(m):
                    counts = np.bincount(
                        [a_i, a_left, a_right], minlength=m
                    )
                    maj = int(np.argmax(counts))
                    U[a_i, a_left, a_right] = 1.0 if a_i == maj else 0.0
        local_utilities.append(U)
    return GraphicalGame(
        n=n, m=m,
        neighbours=neighbours,
        local_utilities=local_utilities,
    )


# ----------------------------------------------------------------------
# Graph helpers (no networkx dependency)
# ----------------------------------------------------------------------


def grid_adjacency(rows: int, cols: int) -> list[tuple[int, ...]]:
    """4-connected grid graph adjacency."""
    n = rows * cols
    adj: list[list[int]] = [[] for _ in range(n)]
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    adj[i].append(nr * cols + nc)
    return [tuple(a) for a in adj]


def cycle_adjacency(n: int) -> list[tuple[int, ...]]:
    return [((i - 1) % n, (i + 1) % n) for i in range(n)]


__all__ = [
    "GraphicalGame",
    "graph_coloring_game",
    "best_shot_public_goods",
    "ring_majority_game",
    "grid_adjacency",
    "cycle_adjacency",
]
