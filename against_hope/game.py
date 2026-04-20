"""Minimal NormalFormGame wrapper.

Only what JLB needs: a tensor-backed utility container.  Used by
`compact_game.NormalFormCompactGame` (wrap an explicit game as a
CompactGame for ground-truth validation on tiny instances) and by
`congestion_game.CongestionGame.as_normal_form()` (same, for tests).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .types import GameSpec


class NormalFormGame:
    """A finite normal-form game stored as explicit utility tensors.

    Only feasible for small games (memory scales as n · prod(mᵢ)).  JLB
    itself never materializes this — it's only used in tests / cross-
    validation.
    """

    def __init__(
        self,
        action_counts: tuple[int, ...],
        utilities: list[NDArray[np.float64]] | None = None,
    ) -> None:
        self.spec = GameSpec.from_action_counts(action_counts)
        if utilities is not None:
            assert len(utilities) == self.spec.n
            self.utilities = [
                np.asarray(u, dtype=np.float64) for u in utilities
            ]
            for i, u in enumerate(self.utilities):
                assert u.shape == self.spec.action_counts, (
                    f"Agent {i}: expected shape {self.spec.action_counts}, "
                    f"got {u.shape}"
                )
        else:
            self.utilities = [
                np.zeros(self.spec.action_counts, dtype=np.float64)
                for _ in range(self.spec.n)
            ]


__all__ = ["NormalFormGame"]
