"""Abstract compact-game interface for the JLB algorithm.

Papadimitriou and Roughgarden (2008) identified two properties a game
representation must satisfy for their compact CE algorithm to run in
polynomial time in the representation size:

  1. Polynomial type: n, max_i m_i are polynomial in the representation.
  2. Polynomial expectation property: for every player i and every product
     distribution x = (x_1, ..., x_n), E_{a ~ x}[u_i(a)] is computable in
     polynomial time.

This module defines a minimal abstract class `CompactGame` exposing those
two oracles (plus standard accessors) and a concrete wrapper
`NormalFormCompactGame` around our existing `NormalFormGame` so that JLB
can be validated on small games for which we have ground truth.

For congestion games we will later add another concrete subclass that
implements `polynomial_expectation` via edge-load moments, avoiding any
enumeration of joint profiles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product

import numpy as np
from numpy.typing import NDArray

from .game import NormalFormGame
from .types import GameSpec


class CompactGame(ABC):
    """Abstract game with polynomial expectation oracle.

    Subclasses must expose:
      - spec: GameSpec (n, action_counts, M, d, N)
      - polynomial_expectation(agent, a_i, product_dist) -> float
      - utility(agent, profile) -> float

    All other derived quantities are computed generically in this base
    class.
    """

    spec: GameSpec

    @abstractmethod
    def utility(self, agent: int, profile: tuple[int, ...]) -> float:
        """Utility u_i(a) for a pure profile."""

    @abstractmethod
    def polynomial_expectation(
        self,
        agent: int,
        a_i: int,
        product_dist: list[NDArray[np.float64]],
    ) -> float:
        """E_{a_{-i} ~ x_{-i}}[u_i(a_i, a_{-i})] under a product distribution.

        Note: conditions agent's own action to be `a_i`; expectation is
        over the other agents only.
        """

    # -----------------------------------------------------------------
    # Generic helpers built on top of the two abstract methods
    # -----------------------------------------------------------------

    def utility_diff_expectation(
        self,
        agent: int,
        a_i: int,
        a_prime_i: int,
        product_dist: list[NDArray[np.float64]],
    ) -> float:
        """E_{a_{-i} ~ x_{-i}}[u_i(a_i, a_{-i}) - u_i(a_prime_i, a_{-i})]."""
        return self.polynomial_expectation(
            agent, a_i, product_dist
        ) - self.polynomial_expectation(agent, a_prime_i, product_dist)

    def apply_to_product(
        self,
        agent: int,
        a_i: int,
        a_prime_i: int,
        product_dist: list[NDArray[np.float64]],
    ) -> float:
        """Return the CE constraint value for (i, a_i, a_prime_i) under a product dist.

        Specifically, if x is a product distribution with agent i restricted
        to playing a_i with probability 1 (via conditioning), this returns
            sum_{a_{-i}} x_{-i}(a_{-i}) * [u_i(a_i, a_{-i}) - u_i(a_prime_i, a_{-i})]
        """
        return self.utility_diff_expectation(agent, a_i, a_prime_i, product_dist)


class NormalFormCompactGame(CompactGame):
    """Wraps a NormalFormGame as a CompactGame.

    The polynomial expectation is computed by direct enumeration --- this
    is O(M) per call and only feasible for small games.  Used as ground
    truth for JLB validation on small games.
    """

    def __init__(self, game: NormalFormGame):
        self._game = game
        self.spec = game.spec

    def utility(self, agent: int, profile: tuple[int, ...]) -> float:
        return float(self._game.utilities[agent][profile])

    def polynomial_expectation(
        self,
        agent: int,
        a_i: int,
        product_dist: list[NDArray[np.float64]],
    ) -> float:
        spec = self.spec
        # Enumerate all a_{-i}
        total = 0.0
        other_ranges = [range(m) for j, m in enumerate(spec.action_counts) if j != agent]
        # Vectorized version: compute conditional distribution over (a_1,...,a_n)
        # with agent i fixed to a_i, and contract with the utility slice.
        x_multi = np.ones(spec.action_counts)
        for i, x_i in enumerate(product_dist):
            shape = [1] * spec.n
            shape[i] = spec.action_counts[i]
            x_multi = x_multi * np.asarray(x_i, dtype=np.float64).reshape(shape)
        # Sum over a_{-i} with agent's action fixed to a_i.
        # Take the slice along agent axis at index a_i, then sum over all remaining axes.
        cond = np.take(x_multi, a_i, axis=agent)  # shape = (m_1, ..., m_{i-1}, m_{i+1}, ..., m_n)
        # Corresponding utility slice
        u_slice = np.take(self._game.utilities[agent], a_i, axis=agent)
        # Marginal over a_{-i}: sum of x(a_i, a_{-i}) times u_i(a_i, a_{-i}).
        # Note: np.take(x_multi, a_i, axis=agent) is x_i(a_i) * prod_{j != i} x_j(a_j)
        # which is P[a_i] * P[a_{-i}].  To get E_{a_{-i} ~ x_{-i}}[u_i(a_i, a_{-i})]
        # we must divide by P[a_i].  But P[a_i] = x_i(a_i); if that is zero the
        # expectation is undefined (we return 0 by convention).
        p_i = float(product_dist[agent][a_i])
        if p_i <= 0:
            # Fall back: bypass the fixed a_i entirely and compute
            # sum_{a_{-i}} prod_{j != i} x_j(a_j) * u_i(a_i, a_{-i}).
            total = 0.0
            it_shape = [spec.action_counts[j] for j in range(spec.n) if j != agent]
            for indices in product(*[range(k) for k in it_shape]):
                profile = list(indices)
                profile.insert(agent, a_i)
                profile = tuple(profile)
                p = 1.0
                for j, a_j in enumerate(profile):
                    if j == agent:
                        continue
                    p *= float(product_dist[j][a_j])
                total += p * float(self._game.utilities[agent][profile])
            return total

        return float(np.sum(cond * u_slice) / p_i)


__all__ = ["CompactGame", "NormalFormCompactGame"]
