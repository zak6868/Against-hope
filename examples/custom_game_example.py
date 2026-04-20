"""Subclass CompactGame for your own succinct game.

Here: a "graphical game" where each agent only cares about their
immediate neighbour on a ring. With n agents and m=3 actions each, the
normal-form has 3^n profiles but the game is described by n local
utility matrices of size 3×3 — succinct.
"""

from __future__ import annotations

import numpy as np

from against_hope import CompactGame, GameSpec, run_jlb_ellipsoid


class RingGraphicalGame(CompactGame):
    """n agents on a ring. u_i(a) depends only on a_i and a_{i+1 mod n}.

    Each agent has m actions. Given a pair-utility matrix U_pair of
    shape (m, m), agent i's utility is U_pair[a_i, a_{i+1 mod n}].
    """

    def __init__(self, n: int, m: int, pair_utility: np.ndarray) -> None:
        assert pair_utility.shape == (m, m)
        self.n = n
        self.m = m
        self.U_pair = pair_utility.astype(np.float64, copy=True)
        self.spec = GameSpec.from_action_counts(tuple([m] * n))

    def utility(self, agent: int, profile: tuple[int, ...]) -> float:
        a_i = profile[agent]
        a_next = profile[(agent + 1) % self.n]
        return float(self.U_pair[a_i, a_next])

    def polynomial_expectation(
        self,
        agent: int,
        a_i: int,
        product_dist: list[np.ndarray],
    ) -> float:
        """E_{a_{-i}}[u_i(a_i, a_{-i})] = Σ_j x_{i+1}[j] · U_pair[a_i, j].

        Only agent (i+1) mod n matters for agent i — the classical
        graphical-game compactness.
        """
        x_next = product_dist[(agent + 1) % self.n]
        return float((x_next * self.U_pair[a_i, :]).sum())


def main() -> None:
    # Coordination game: both play 0 or both play 1 wins.
    pair_utility = np.array([
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 3.0],
    ])

    for n in [5, 10, 30]:
        g = RingGraphicalGame(n=n, m=3, pair_utility=pair_utility)
        print(f"\nRing graphical game n={n}: M = 3^{n} = {3**n:.1e} profiles")
        res = run_jlb_ellipsoid(
            g,
            max_iterations=20_000,
            initial_radius=10.0,
            reconstruct_every=50,
        )
        if res.reconstructed:
            print(f"  ✓ CE found. Support: {len(res.profiles)}")
        else:
            print(f"  ✗ No CE in {res.timings['iterations']} iterations.")


if __name__ == "__main__":
    main()
