"""JLB 2011 pipeline with the ellipsoid method as outer loop.

This replaces `jlb_driver.run_jlb`'s "random-y" outer loop with a
rigorous ellipsoid-based infeasibility certification:

  1. Start with an ellipsoid enclosing the unit box in y-space.
  2. At each iteration, query the JLB separation oracle at the center.
     - If the center has a negative coordinate y_i < 0, use the axis-aligned
       cut y_i >= 0 (i.e., g = -e_i, h = 0) as the separating hyperplane.
     - Otherwise, the JLB oracle returns a pure profile a* satisfying
       (U_{a*})^T y >= 0 > -1.  The cut is g = U_{a*}, h = -1.
  3. Update the ellipsoid with the deep-cut rule.
  4. Stop when the ellipsoid's log-volume ratio drops below the threshold
     (certifying infeasibility of the dual LP), or when we've collected
     enough pure-profile cuts to reconstruct a CE.

The collected pure profiles are then fed to the standard CE reconstruction
LP (same as in jlb_driver).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .compact_game import CompactGame
from .ellipsoid import Ellipsoid, EllipsoidRun, ellipsoid_infeasible_loop
from .jlb_driver import reconstruct_ce_mixture
from .jlb_oracle import (
    OracleResult,
    cut_from_profile,
    evaluate_f_pure,
    jlb_separation_oracle,
    y_dim,
)


@dataclass
class EllipsoidJLBResult:
    profiles: list[tuple[int, ...]] = field(default_factory=list)
    alpha: Optional[NDArray[np.float64]] = None
    ellipsoid_run: Optional[EllipsoidRun] = None
    reconstructed: bool = False
    stopped_reason: str = ""
    timings: dict = field(default_factory=dict)


def run_jlb_ellipsoid(
    game: CompactGame,
    max_iterations: int = 2000,
    initial_radius: float = 10.0,
    min_log_vol_ratio: float = -300.0,
    reconstruct_every: int = 20,
    reconstruct_backend: str = "auto",
    seed: int = 0,
    verbose: bool = False,
) -> EllipsoidJLBResult:
    """Run JLB with an ellipsoid outer loop.

    Parameters
    ----------
    game : CompactGame
    max_iterations : int
        Max ellipsoid iterations.
    initial_radius : float
        Radius of the initial ball in y-space.  Dual constraints have
        encoding roughly log(u_max), so a modest radius suffices for
        most practical games; for exact JLB termination one would use
        the theoretical R = u^{5N^3}, which is astronomical in practice.
    min_log_vol_ratio : float
        Stop when log(Vol_current / Vol_initial) < this threshold.
    reconstruct_every : int
        Attempt CE reconstruction every this many profile-cuts collected.
    verbose : bool
    """
    spec = game.spec
    N = y_dim(spec)
    rng = np.random.default_rng(seed)

    # Starting center must scale with 1/sqrt(N) so the first JLB cut's
    # deep-cut parameter α = (g^T c - h)/sqrt(g^T P g) stays O(1).
    # ||g||_1 ~ N·u_max and ||g||_2 ~ sqrt(N)·u_max, so a fixed-magnitude
    # center would give α ~ sqrt(N)/R — unbounded as N grows (fatal at
    # n=100, y_dim=900: ellipsoid reports infeasibility in 1 iteration
    # before collecting enough profiles to reconstruct a CE).
    # We shrink the ε factor so gentle cuts emerge even at large N.
    eps = 0.1 / max(np.sqrt(N), 1.0)
    c0 = np.full(N, eps * initial_radius, dtype=np.float64)

    profiles: list[tuple[int, ...]] = []
    profile_set: set[tuple[int, ...]] = set()
    alpha: Optional[NDArray[np.float64]] = None
    reconstructed = False
    stopped_reason = "max_iter"

    t_oracle = 0.0
    t_ellipsoid = 0.0
    t_reconstruct = 0.0

    ellipsoid = Ellipsoid.ball(N, initial_radius, center=c0)

    consecutive_redundant = 0
    infeasible_attempts = 0  # How many times an "infeasible" status fired without CE found

    for t in range(max_iterations):
        y = ellipsoid.c

        # Priority 1: y_i >= 0 constraint
        neg_idx = int(np.argmin(y))
        used_nonnegativity_cut = False
        if y[neg_idx] < -1e-12:
            g = np.zeros(N)
            g[neg_idx] = -1.0
            h = 0.0  # -y_i <= 0  <=> y_i >= 0
            used_nonnegativity_cut = True
        else:
            # Priority 2: JLB oracle cut
            t0 = perf_counter()
            res = jlb_separation_oracle(game, np.maximum(y, 0.0), rng=rng)
            t_oracle += perf_counter() - t0
            profile = res.profile
            if profile not in profile_set:
                profiles.append(profile)
                profile_set.add(profile)
            g = res.cut                # U_{a*}
            h = -1.0                   # (U_{a*})^T y <= -1 is the cut
            # Normalize (g, h) together so the cut has bounded magnitude
            # regardless of utility scale. This keeps the ellipsoid's deep-cut
            # parameter α bounded and prevents single-iteration infeasibility
            # when ||U_{a*}|| >> initial ball radius (a common failure mode
            # at large n).
            g_norm = float(np.linalg.norm(g))
            if g_norm > 1e-15:
                g = g / g_norm
                h = h / g_norm

        t0 = perf_counter()
        status = ellipsoid.apply_cut(g, h)
        t_ellipsoid += perf_counter() - t0

        if verbose and (t < 5 or (t + 1) % max(1, max_iterations // 30) == 0):
            print(f"  iter {t+1:5d}  #prof={len(profiles):4d}  "
                  f"log_vol={ellipsoid.log_vol_ratio:+.2f}  "
                  f"status={status}"
                  f"{' (y>=0 cut)' if used_nonnegativity_cut else ''}", flush=True)

        if status == "infeasible":
            # Try reconstruction with current profiles.  If it succeeds we're
            # done.  Otherwise we don't yet have enough JLB cuts; perturb the
            # center back into the interior of the orthant and continue.
            t0 = perf_counter()
            alpha = reconstruct_ce_mixture(game, profiles, backend=reconstruct_backend)
            t_reconstruct += perf_counter() - t0
            if alpha is not None:
                reconstructed = True
                stopped_reason = "ce_reconstructed_on_infeasible"
                break
            # Not enough cuts yet: reset the center to a random interior point
            # of the positive orthant to elicit a different profile from the
            # oracle.  We reset P as well (otherwise the ellipsoid stays tiny
            # and the same center would dominate).
            infeasible_attempts += 1
            if infeasible_attempts > 50:
                stopped_reason = "infeasible_cert_without_enough_cuts"
                break
            shrink = 0.5 ** infeasible_attempts  # progressively smaller radius
            new_radius = max(initial_radius * shrink, 0.1)
            ellipsoid.c = rng.uniform(0.05, 1.0, size=N) * initial_radius
            ellipsoid.P = (new_radius ** 2) * np.eye(N)
            continue
        if status == "null_cut":
            stopped_reason = "null_cut"
            break
        if status == "redundant":
            consecutive_redundant += 1
            if consecutive_redundant > 20:
                stopped_reason = "redundant_loop"
                break
        else:
            consecutive_redundant = 0

        if ellipsoid.log_vol_ratio < min_log_vol_ratio:
            stopped_reason = "volume_threshold"
            break

        # Every reconstruct_every new profiles, try to reconstruct CE.
        if (len(profiles) > 0
            and len(profiles) % reconstruct_every == 0
            and len(profiles) != getattr(run_jlb_ellipsoid, "_last_recon", -1)):
            t0 = perf_counter()
            alpha = reconstruct_ce_mixture(game, profiles, backend=reconstruct_backend)
            t_reconstruct += perf_counter() - t0
            run_jlb_ellipsoid._last_recon = len(profiles)
            if alpha is not None:
                reconstructed = True
                stopped_reason = "ce_reconstructed"
                break

    # Final reconstruction attempt if not yet done
    if not reconstructed and len(profiles) > 0:
        t0 = perf_counter()
        alpha = reconstruct_ce_mixture(game, profiles, backend=reconstruct_backend)
        t_reconstruct += perf_counter() - t0
        if alpha is not None:
            reconstructed = True
            if stopped_reason == "max_iter":
                stopped_reason = "ce_reconstructed_final"

    return EllipsoidJLBResult(
        profiles=profiles,
        alpha=alpha,
        reconstructed=reconstructed,
        stopped_reason=stopped_reason,
        timings={
            "oracle_s": t_oracle,
            "ellipsoid_s": t_ellipsoid,
            "reconstruct_s": t_reconstruct,
            "iterations": t + 1,
            "final_log_vol_ratio": ellipsoid.log_vol_ratio,
        },
    )


__all__ = ["EllipsoidJLBResult", "run_jlb_ellipsoid"]
