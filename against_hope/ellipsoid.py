"""Ellipsoid method for LP feasibility with a black-box separation oracle.

An ellipsoid in R^N is parameterized by a center c and a positive-definite
shape matrix P:
    E(c, P) = { y in R^N : (y - c)^T P^{-1} (y - c) <= 1 }

Given a half-space cut  g^T y <= h  (which the current center c violates,
i.e., g^T c > h), the deep-cut ellipsoid update produces the smallest
ellipsoid containing E(c, P) intersect { y : g^T y <= h }.

Update rule (Grotschel-Lovasz-Schrijver 1988, standard deep-cut form):
    Let  s = sqrt(g^T P g)  (a positive scalar).
    Let  alpha = (g^T c - h) / s   (depth of the cut; must be > 0).
    If  alpha >= 1:   the cut excludes all of E; LP is infeasible.
    Else:
        tau    = (1 + N*alpha) / (N + 1)
        sigma  = 2 * (1 + N*alpha) / ((N + 1) * (1 + alpha))
        delta  = (N^2 / (N^2 - 1)) * (1 - alpha^2)
        c_new  = c - (tau / s) * P g
        P_new  = delta * (P - (sigma / s^2) * (P g) (P g)^T)

Volume shrinks by factor  exp(-alpha^2 / (2N) - 1/(2N+2))  roughly, so
after O(N^2 log(Vol0 / Volmin)) iterations the ellipsoid is smaller than
the minimum feasible volume (if feasible) => certified infeasible.

For JLB on dual LP (D):  { y >= 0 : U^T y <= -1 } is known infeasible,
so the ellipsoid will always shrink to certify this.

Usage
-----
Instantiate Ellipsoid(N, R) for a ball of radius R at origin; call
apply_cut(g, h) each iteration.  Use ellipsoid_infeasible_loop() to
wrap the standard infeasibility-certification loop around a user-
provided separation oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class Ellipsoid:
    """A full-dimensional ellipsoid E(c, P) in R^N."""

    N: int
    c: NDArray[np.float64]       # center, shape (N,)
    P: NDArray[np.float64]       # shape matrix, shape (N, N), PSD
    # log-volume relative to initial (for convergence tracking).
    # log_vol_factor is updated each iteration so we can certify
    # infeasibility without computing det(P) directly.
    log_vol_ratio: float = 0.0   # log(Vol_current / Vol_initial)

    @staticmethod
    def ball(N: int, radius: float, center: Optional[NDArray[np.float64]] = None) -> "Ellipsoid":
        """Initialize as a ball of given radius (optionally offset)."""
        c = np.zeros(N) if center is None else np.asarray(center, dtype=np.float64).copy()
        P = (radius ** 2) * np.eye(N)
        return Ellipsoid(N=N, c=c, P=P, log_vol_ratio=0.0)

    def apply_cut(self, g: NDArray[np.float64], h: float) -> str:
        """Apply the half-space cut  g^T y <= h  using the deep-cut update.

        The current center self.c is assumed to violate the cut (g^T c > h).
        If the violation alpha >= 1 the LP is infeasible.

        Returns a status string:
          "cut"        : ellipsoid updated.
          "infeasible" : the cut excludes the entire ellipsoid.
          "null_cut"   : g is effectively zero (shouldn't happen).
        """
        g = np.asarray(g, dtype=np.float64)
        Pg = self.P @ g                       # (N,)
        gtPg = float(g @ Pg)
        if gtPg <= 1e-30:
            return "null_cut"
        s = np.sqrt(gtPg)
        alpha = (float(g @ self.c) - h) / s
        if alpha >= 1.0:
            return "infeasible"
        if alpha <= -1.0 + 1e-15:
            # The cut is not violated by any point in the ellipsoid;
            # no update needed.
            return "redundant"

        N = self.N
        tau = (1.0 + N * alpha) / (N + 1.0)
        sigma = 2.0 * (1.0 + N * alpha) / ((N + 1.0) * (1.0 + alpha))
        delta = (N * N) / (N * N - 1.0) * (1.0 - alpha * alpha)

        self.c = self.c - (tau / s) * Pg
        # P_new = delta * (P - (sigma / gtPg) * Pg Pg^T)
        self.P = delta * (self.P - (sigma / gtPg) * np.outer(Pg, Pg))
        # Symmetrize for numerical stability
        self.P = 0.5 * (self.P + self.P.T)

        # Volume ratio update.
        # For the deep-cut update P_new = delta * (P - (sigma/gtPg) * Pg Pg^T):
        #   det(P_new) = delta^N * det(P) * (1 - sigma)
        # and volume(E_new) / volume(E_old) = sqrt(det(P_new) / det(P))
        #                                   = delta^(N/2) * sqrt(1 - sigma)
        # So log(vol_new / vol_old) = (N/2) * log(delta) + (1/2) * log(1 - sigma).
        if sigma < 1.0 and delta > 0.0:
            self.log_vol_ratio += 0.5 * (N * np.log(delta) + np.log(1.0 - sigma))
        else:
            self.log_vol_ratio += -1e6  # effectively zero volume

        return "cut"


# ---------------------------------------------------------------------------
# Infeasibility-certification driver
# ---------------------------------------------------------------------------


@dataclass
class EllipsoidRun:
    """Result of running the ellipsoid method."""

    cuts: list[tuple[NDArray[np.float64], float]] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = ""      # "infeasible_certified" / "max_iter" / "redundant_loop"
    final_ellipsoid: Optional[Ellipsoid] = None


def ellipsoid_infeasible_loop(
    N: int,
    separation_oracle: Callable[[NDArray[np.float64]], Optional[tuple[NDArray[np.float64], float]]],
    initial_radius: float = 1.0,
    max_iterations: int = 10_000,
    min_log_vol_ratio: float = -500.0,
    initial_center: Optional[NDArray[np.float64]] = None,
    verbose: bool = False,
) -> EllipsoidRun:
    """Run the ellipsoid method for LP infeasibility certification.

    Parameters
    ----------
    N : int
        Dimension of the LP variable space.
    separation_oracle : callable
        Given a candidate y (center), returns either:
          - None if y is feasible (never happens for known-infeasible LPs), or
          - (g, h) such that g^T y <= h is a valid LP constraint violated by y.
    initial_radius : float
        Radius of initial ball.
    max_iterations : int
        Maximum ellipsoid iterations.
    min_log_vol_ratio : float
        Stop when ellipsoid volume drops below exp(min_log_vol_ratio) times
        initial volume.  This certifies infeasibility if the LP is known to
        admit a feasible ball of radius > sqrt(exp(min_log_vol_ratio)).
    initial_center : optional
        Initial ellipsoid center.  Defaults to the origin.
    verbose : bool
    """
    E = Ellipsoid.ball(N, initial_radius, center=initial_center)
    run = EllipsoidRun(final_ellipsoid=E)

    consecutive_redundant = 0
    for t in range(max_iterations):
        run.iterations = t + 1
        result = separation_oracle(E.c)
        if result is None:
            run.stopped_reason = "feasible_center"
            return run

        g, h = result
        run.cuts.append((np.asarray(g, dtype=np.float64).copy(), float(h)))

        status = E.apply_cut(np.asarray(g, dtype=np.float64), float(h))

        if verbose and (t < 5 or t % max(1, max_iterations // 20) == 0):
            print(f"  iter {t+1:5d}  log_vol_ratio={E.log_vol_ratio:+.2f}  "
                  f"status={status}", flush=True)

        if status == "infeasible":
            run.stopped_reason = "infeasible_certified"
            return run
        if status == "null_cut":
            run.stopped_reason = "null_cut"
            return run
        if status == "redundant":
            consecutive_redundant += 1
            if consecutive_redundant > 10:
                run.stopped_reason = "redundant_loop"
                return run
        else:
            consecutive_redundant = 0

        if E.log_vol_ratio < min_log_vol_ratio:
            run.stopped_reason = "volume_threshold"
            return run

    run.stopped_reason = "max_iter"
    return run


__all__ = ["Ellipsoid", "EllipsoidRun", "ellipsoid_infeasible_loop"]
