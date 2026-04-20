"""GameSpec: immutable specification of game dimensions.

Used throughout JLB to pass around (n, action_counts, M, N) without
passing the whole game object.  Deliberately small and dependency-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GameSpec:
    """Immutable specification of game dimensions.

    Attributes
    ----------
    n
        Number of agents.
    action_counts
        Tuple (m_1, ..., m_n) of per-agent action counts.
    M
        Total number of joint profiles = prod(m_i).  NEVER materialized
        as storage — computed here only so caller code can reason about
        scale.
    d
        Tuple (d_1, ..., d_n) with d_i = M / m_i (= |A_{-i}|).
    N
        sum_i (m_i - 1) * d_i.  This is the dimension of the full dual-
        utility vector; JLB operates on the compact y ∈ R^{Σ m_i^2}
        layer, not on R^N.
    """

    n: int
    action_counts: tuple[int, ...]
    M: int
    d: tuple[int, ...]
    N: int

    @staticmethod
    def from_action_counts(action_counts: tuple[int, ...]) -> "GameSpec":
        n = len(action_counts)
        M = math.prod(action_counts)
        d = tuple(M // m_i for m_i in action_counts)
        N = sum((m_i - 1) * d_i for m_i, d_i in zip(action_counts, d))
        return GameSpec(n=n, action_counts=action_counts, M=M, d=d, N=N)


__all__ = ["GameSpec"]
