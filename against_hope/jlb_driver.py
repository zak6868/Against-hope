"""End-to-end JLB pipeline driver and CE extraction.

Given a compact game, the JLB algorithm proceeds as follows:

  1. Run a cutting-plane method on the dual LP (D):  U^T y <= -1, y >= 0.
     The dual is infeasible (CE exists), so the method terminates after
     a polynomial number of iterations and produces a collection of
     pure-profile cuts s(1), ..., s(L).

  2. Solve the (polynomial-size) primal LP (P''):
         find x' in R^L,  U' x' >= 0,  x' >= 0,  sum x' = 1
     where the columns of U' are U_{s(1)}, ..., U_{s(L)}.  A feasible
     x' gives a CE as a mixture over {s(1), ..., s(L)}.

This module implements both steps for validation on small games, using
floating-point arithmetic.  The outer loop here is NOT yet the rigorous
exact-arithmetic ellipsoid; it is a simple "random-y + solve primal"
driver that suffices to:

  - Verify the full JLB logic on small games (Prisoner's Dilemma,
    Matching Pennies, Shapley).
  - Produce a baseline implementation that can later be swapped for a
    SCIP-backed rigorous version.

Once this end-to-end pipeline is validated, we will replace the outer
loop with SCIP 10 in exact mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from .compact_game import CompactGame
from .jlb_oracle import (
    OracleResult,
    cut_from_profile,
    evaluate_f_pure,
    jlb_separation_oracle,
    y_dim,
)


# ---------------------------------------------------------------------------
# CE reconstruction from a list of pure profiles
# ---------------------------------------------------------------------------


def build_U_prime(
    game: CompactGame, profiles: list[tuple[int, ...]]
) -> NDArray[np.float64]:
    """Assemble the N x L matrix U' whose columns are U_{s(k)}.

    Each column is a CE-constraint column (signs match the convention
    in jlb_oracle.cut_from_profile: entries are u_i(a_i, a_{-i}) -
    u_i(a'_i, a_{-i})).
    """
    L = len(profiles)
    N = y_dim(game.spec)
    U_prime = np.zeros((N, L), dtype=np.float64)
    for k, s in enumerate(profiles):
        U_prime[:, k] = cut_from_profile(game, s)
    return U_prime


def reconstruct_ce_mixture(
    game: CompactGame,
    profiles: list[tuple[int, ...]],
    backend: str = "auto",
    tol: float = 1e-9,
) -> Optional[NDArray[np.float64]]:
    """Solve for CE mixture weights over the given pure profiles.

    We look for alpha in R^L with alpha >= 0, sum alpha = 1, and U' alpha >= 0
    where U' is as built by build_U_prime.  If the LP is feasible, returns
    alpha; otherwise None.

    Uses Gurobi when available, scipy.linprog as a fallback.
    """
    L = len(profiles)
    if L == 0:
        return None

    U_prime = build_U_prime(game, profiles)
    # Drop all-zero rows (trivial constraints where the profile's coord doesn't match)
    # — scipy struggles with them, Gurobi is fine.  We keep them for now.

    if backend == "auto":
        try:
            import gurobipy  # noqa: F401
            backend = "gurobi"
        except ImportError:
            backend = "scipy"

    if backend == "gurobi":
        return _solve_gurobi(U_prime, L, tol)
    return _solve_scipy(U_prime, L, tol)


def _solve_gurobi(
    U_prime: NDArray[np.float64], L: int, tol: float
) -> Optional[NDArray[np.float64]]:
    import gurobipy as gp
    model = gp.Model()
    model.Params.OutputFlag = 0
    alpha = model.addMVar(L, lb=0.0, name="alpha")
    # U' alpha >= 0
    model.addMConstr(U_prime, alpha, ">", np.zeros(U_prime.shape[0]))
    # sum alpha = 1
    model.addConstr(alpha.sum() == 1.0)
    # objective: minimize sum(alpha * alpha) to pick an interior-ish solution
    model.setObjective(alpha @ alpha, gp.GRB.MINIMIZE)
    model.optimize()
    if model.status == gp.GRB.OPTIMAL:
        sol = np.array(alpha.X, dtype=np.float64)
        sol = np.maximum(sol, 0.0)
        s = sol.sum()
        if s > 0:
            sol = sol / s
        return sol
    return None


def _solve_scipy(
    U_prime: NDArray[np.float64], L: int, tol: float
) -> Optional[NDArray[np.float64]]:
    from scipy.optimize import linprog
    # feasibility LP:
    #   alpha >= 0,  sum alpha = 1,  U' alpha >= 0
    # rewrite U' alpha >= 0  as  -U' alpha <= 0
    c = np.zeros(L)
    A_ub = -U_prime
    b_ub = np.zeros(U_prime.shape[0])
    A_eq = np.ones((1, L))
    b_eq = np.array([1.0])
    res = linprog(
        c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=[(0, None)] * L, method="highs",
    )
    if res.success:
        sol = np.maximum(res.x, 0.0)
        s = sol.sum()
        if s > 0:
            sol = sol / s
        return sol
    return None


# ---------------------------------------------------------------------------
# Simple outer driver
# ---------------------------------------------------------------------------


@dataclass
class JLBRunResult:
    """Bundle the output of the JLB driver."""

    profiles: list[tuple[int, ...]]            # pure profiles collected
    alpha: Optional[NDArray[np.float64]]       # mixture weights, None if no CE found
    iterations: int                            # oracle calls made
    stopped_reason: str                        # "found_ce" / "max_iter" / "error"
    diagnostics: dict = field(default_factory=dict)


def random_y_schedule(
    spec, n_iter: int, seed: int = 0, scale: float = 1.0
) -> Callable[[int], NDArray[np.float64]]:
    """Return a function t -> y (random y vectors).

    Useful as a starting outer-loop strategy for validation on small games.
    """
    rng = np.random.default_rng(seed)
    N = y_dim(spec)
    pool = rng.uniform(0.0, scale, size=(n_iter, N))

    def sched(t: int) -> NDArray[np.float64]:
        return pool[t]

    return sched


def structured_y_schedule(
    spec, n_iter: int, seed: int = 0
) -> Callable[[int], NDArray[np.float64]]:
    """Outer-loop schedule that mixes canonical "push-to-action-k" y's with
    random perturbations.

    A "push-to-k" y has Y_p[j, k] = 1 for all j != k, driving each player's
    Markov chain toward the absorbing state k.  The oracle then produces the
    pure Nash candidate (k, k, ..., k).  Cycling through all possible target
    actions k gives the oracle a chance to propose every "all-same" profile,
    which in many structured games (congestion, Pigou, etc.) are natural
    Nash candidates.
    """
    rng = np.random.default_rng(seed)
    N = y_dim(spec)
    max_m = max(spec.action_counts)
    # Pre-compute push-to-k y vectors for each k in 0..max_m-1
    push_vecs = []
    for k in range(max_m):
        y = np.zeros(N)
        offset = 0
        for i, m_i in enumerate(spec.action_counts):
            if k < m_i:
                for j in range(m_i):
                    if j != k:
                        y[offset + j * m_i + k] = 1.0
            offset += m_i * m_i
        push_vecs.append(y)

    random_pool = rng.uniform(0.0, 1.0, size=(n_iter, N))

    def sched(t: int) -> NDArray[np.float64]:
        # Rotate through canonical pushes in the first `max_m` iterations,
        # then a mix of push + noise, then pure random.
        if t < max_m:
            return push_vecs[t].copy()
        if t < 3 * max_m:
            base = push_vecs[t % max_m]
            noise = rng.uniform(0, 0.2, size=N)
            return base + noise
        return random_pool[t]

    return sched


def run_jlb(
    game: CompactGame,
    max_iterations: int = 200,
    y_schedule: Optional[Callable[[int], NDArray[np.float64]]] = None,
    reconstruct_every: int = 10,
    verbose: bool = False,
    seed: int = 0,
) -> JLBRunResult:
    """Run the JLB algorithm end-to-end.

    Parameters
    ----------
    game : CompactGame
    max_iterations : int
        Maximum number of oracle calls.
    y_schedule : callable t -> y, or None
        If None, uses a random y schedule (for validation on small games).
        In the final rigorous version this will be the ellipsoid-generated
        sequence of query points.
    reconstruct_every : int
        Try to solve the CE reconstruction LP every this many iterations.
        As soon as a feasible CE is found, stop.
    verbose : bool
    seed : int
    """
    spec = game.spec
    if y_schedule is None:
        y_schedule = structured_y_schedule(spec, max_iterations, seed=seed)

    profiles: list[tuple[int, ...]] = []
    profiles_set: set[tuple[int, ...]] = set()  # avoid duplicates

    alpha: Optional[NDArray[np.float64]] = None
    stopped = "max_iter"

    for t in range(max_iterations):
        y = y_schedule(t)
        # Make y non-negative (in case the schedule doesn't guarantee it)
        y = np.maximum(y, 0.0)
        res = jlb_separation_oracle(game, y)
        if res.profile not in profiles_set:
            profiles.append(res.profile)
            profiles_set.add(res.profile)

        if verbose and (t + 1) % max(1, reconstruct_every) == 0:
            print(f"iter {t+1:4d}  #profiles={len(profiles):3d}  "
                  f"f(a*)={res.f_value:.3e}  E_x[f]={res.expected_f:.3e}")

        # Attempt CE reconstruction every `reconstruct_every` iterations
        if (t + 1) % reconstruct_every == 0:
            alpha = reconstruct_ce_mixture(game, profiles)
            if alpha is not None:
                stopped = "found_ce"
                break

    return JLBRunResult(
        profiles=profiles,
        alpha=alpha,
        iterations=t + 1,
        stopped_reason=stopped,
        diagnostics={"distinct_profiles": len(profiles)},
    )


def _solve_gurobi_with_farkas(
    U_prime: NDArray[np.float64], L: int, tol: float = 1e-9,
) -> tuple[Optional[NDArray[np.float64]], Optional[NDArray[np.float64]]]:
    """Solve the reconstruction LP. On infeasibility, also extract a Farkas
    direction y >= 0 that certifies infeasibility (y · U_{a^k} has one
    sign for all collected profiles).  Returns (alpha, farkas_y); one is
    always None.
    """
    import gurobipy as gp

    model = gp.Model()
    model.Params.OutputFlag = 0
    model.Params.InfUnbdInfo = 1
    # Linear LP: minimize a trivial linear objective on α.  (Gurobi only
    # provides Farkas duals for linear programs, not QPs.)
    alpha = model.addMVar(L, lb=0.0, name="alpha")
    ce_cons = model.addMConstr(
        U_prime, alpha, ">", np.zeros(U_prime.shape[0])
    )
    model.addConstr(alpha.sum() == 1.0)
    model.setObjective(alpha.sum(), gp.GRB.MINIMIZE)  # trivial linear obj
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        sol = np.maximum(np.array(alpha.X, dtype=np.float64), 0.0)
        s = sol.sum()
        if s > 0:
            sol = sol / s
        return sol, None

    if model.status == gp.GRB.INFEASIBLE:
        try:
            fd = np.array(
                [c.FarkasDual for c in ce_cons.tolist()],
                dtype=np.float64,
            )
        except AttributeError:
            fd = np.array(ce_cons.getAttr("FarkasDual"), dtype=np.float64)
        # Take |y| to get a non-negative probe direction (sign of Farkas
        # depends on Gurobi's internal normalization).
        y = np.abs(fd)
        yn = float(np.linalg.norm(y))
        if yn > 0:
            y = y / yn
        return None, y

    return None, None


@dataclass
class ColumnGenResult:
    profiles: list[tuple[int, ...]]
    alpha: Optional[NDArray[np.float64]]
    iterations: int
    stopped_reason: str
    diagnostics: dict = field(default_factory=dict)


def run_jlb_column_generation(
    game: CompactGame,
    max_iterations: int = 300,
    seed: int = 0,
    verbose: bool = False,
) -> ColumnGenResult:
    """JLB outer loop via LP-guided column generation.

    At each step:
      1. Solve the reconstruction LP over the profiles collected so far.
      2. If feasible → done (return α).
      3. Else → extract Farkas dual direction y. Use y as probe for the
         PR oracle → new pure profile a* (theoretically non-redundant).
      4. Add a* and loop.

    Seed with canonical "all-k" profiles plus one random-y draw.  Converges
    in at most O(L) iterations where L is the CE support size (often O(d)
    for structured games).
    """
    spec = game.spec
    rng = np.random.default_rng(seed)

    profiles: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def _add(p: tuple[int, ...]) -> bool:
        t = tuple(int(x) for x in p)
        if t in seen:
            return False
        profiles.append(t)
        seen.add(t)
        return True

    # Seed with "all agents play k" profiles and one random-y probe.
    max_m = max(spec.action_counts)
    for k in range(max_m):
        _add(tuple(min(k, spec.action_counts[i] - 1) for i in range(spec.n)))
    N = y_dim(spec)
    y0 = rng.uniform(0.1, 1.0, size=N)
    _add(jlb_separation_oracle(game, y0, rng=rng).profile)

    alpha: Optional[NDArray[np.float64]] = None
    stopped = "max_iter"
    iters_done = 0

    for t in range(max_iterations):
        iters_done = t + 1
        U_prime = build_U_prime(game, profiles)
        alpha, farkas_y = _solve_gurobi_with_farkas(U_prime, len(profiles))
        if alpha is not None:
            stopped = "found_ce"
            break
        if farkas_y is None:
            stopped = "lp_error"
            break
        res = jlb_separation_oracle(game, farkas_y, rng=rng)
        if not _add(res.profile):
            # Duplicate — perturb and retry once.
            y_perturbed = farkas_y + rng.uniform(0.0, 0.1, size=N)
            res = jlb_separation_oracle(game, y_perturbed, rng=rng)
            if not _add(res.profile):
                stopped = "stuck_duplicate"
                break
        if verbose and iters_done % 10 == 0:
            print(f"  cg-iter {iters_done}: profiles={len(profiles)}", flush=True)

    return ColumnGenResult(
        profiles=profiles,
        alpha=alpha,
        iterations=iters_done,
        stopped_reason=stopped,
    )


__all__ = [
    "build_U_prime",
    "reconstruct_ce_mixture",
    "random_y_schedule",
    "run_jlb",
    "run_jlb_column_generation",
    "ColumnGenResult",
    "JLBRunResult",
]
