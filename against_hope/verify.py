"""CE-correctness verification.

Checks every CE constraint literally for a candidate mixture
{(profile_k, alpha_k)} on an abstract `CompactGame`.  For each agent i,
recommended action a_i, and alternative a'_i, computes

    gain(i, a_i, a'_i) = Σ_k α_k · 1[profile_k[i] = a_i]
                        · [ u_i(a'_i, a_{-i}^k) - u_i(a_i, a_{-i}^k) ]

and asserts every entry ≤ 0 (no agent prefers deviating).  Returns the
max violation across all constraints — a true CE has max ≤ 0 up to
float precision.

Complexity: O(Σ_i m_i² · |support|) utility evaluations.  Uses the
game's `utility(agent, profile)` — no assumption about the compact
representation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .compact_game import CompactGame


@dataclass
class CEVerificationResult:
    is_ce: bool
    max_violation: float                 # max over constraints of gain
    worst_violation: tuple[int, int, int] # (agent, a_i, a'_i)
    n_violated: int                       # count of constraints with gain > tol


def verify_ce(
    game: CompactGame,
    profiles: list[tuple[int, ...]],
    alpha: NDArray[np.float64],
    tol: float = 1e-8,
) -> CEVerificationResult:
    """Return whether the mixture is a CE (max constraint violation ≤ tol)."""
    spec = game.spec
    alpha = np.asarray(alpha, dtype=np.float64)
    assert len(profiles) == len(alpha)
    max_v = -np.inf
    worst = (-1, -1, -1)
    n_violated = 0

    for i in range(spec.n):
        m_i = spec.action_counts[i]
        for a_i in range(m_i):
            for a_prime in range(m_i):
                if a_prime == a_i:
                    continue
                gain = 0.0
                for profile, w in zip(profiles, alpha):
                    if w <= 0 or profile[i] != a_i:
                        continue
                    profile_dev = list(profile)
                    profile_dev[i] = a_prime
                    u_here = game.utility(i, tuple(profile))
                    u_dev = game.utility(i, tuple(profile_dev))
                    gain += float(w) * (u_dev - u_here)
                if gain > max_v:
                    max_v = gain
                    worst = (i, a_i, a_prime)
                if gain > tol:
                    n_violated += 1

    return CEVerificationResult(
        is_ce=max_v <= tol,
        max_violation=max_v,
        worst_violation=worst,
        n_violated=n_violated,
    )


__all__ = ["verify_ce", "CEVerificationResult"]
