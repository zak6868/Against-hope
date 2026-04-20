"""Ground-truth brute-force helpers for small-game tests.

Only what JLB tests actually need.  The original project has a heavier
`brute_force_ce` via a normal-form CE solver — omitted here since we
don't ship the moderator's CE solver (it would defeat the purpose of
the JLB stand-alone).
"""

from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from against_hope.game import NormalFormGame


def enumerate_profiles(
    spec_action_counts: tuple[int, ...],
) -> Iterable[tuple[int, ...]]:
    """Iterate over all joint pure profiles.  Feasible only for small M."""
    for profile in product(*[range(m) for m in spec_action_counts]):
        yield profile


def expected_utility_brute_force(
    game: NormalFormGame,
    agent: int,
    x_product: list[NDArray[np.float64]],
) -> float:
    """E_{a ~ x_product}[u_agent(a)] by enumerating all profiles."""
    assert len(x_product) == game.spec.n
    total = 0.0
    for profile in enumerate_profiles(game.spec.action_counts):
        prob = 1.0
        for i, a_i in enumerate(profile):
            prob *= float(x_product[i][a_i])
        if prob == 0.0:
            continue
        total += prob * float(game.utilities[agent][profile])
    return total


def ce_constraints_dense(game: NormalFormGame) -> NDArray[np.float64]:
    """Return the full CE constraint matrix (rows indexed by
    (i, a_i, a'_i), columns by flat profile index)."""
    spec = game.spec
    M = spec.M
    rows: list[NDArray[np.float64]] = []
    action_counts = spec.action_counts
    for i in range(spec.n):
        m_i = action_counts[i]
        for a_i in range(m_i):
            for a_prime in range(m_i):
                if a_prime == a_i:
                    continue
                row = np.zeros(M, dtype=np.float64)
                for profile in enumerate_profiles(action_counts):
                    if profile[i] != a_i:
                        continue
                    profile_dev = list(profile)
                    profile_dev[i] = a_prime
                    flat = int(np.ravel_multi_index(profile, action_counts))
                    u_here = float(game.utilities[i][profile])
                    u_dev = float(game.utilities[i][tuple(profile_dev)])
                    row[flat] = u_dev - u_here
                rows.append(row)
    return np.stack(rows, axis=0)


def verify_ce_dense(
    game: NormalFormGame,
    x: NDArray[np.float64],
    tol: float = 1e-10,
) -> tuple[bool, float]:
    """(is_ce, max_violation) for a dense mixture x on joint profiles."""
    U = ce_constraints_dense(game)
    violations = U @ x
    max_v = float(np.max(violations))
    return bool(max_v <= tol), max_v


__all__ = [
    "enumerate_profiles",
    "expected_utility_brute_force",
    "ce_constraints_dense",
    "verify_ce_dense",
]
