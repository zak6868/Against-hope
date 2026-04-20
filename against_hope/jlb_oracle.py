"""JLB 2011 separation oracle for the CE dual LP.

Given a compact game and a candidate dual vector y >= 0, the oracle:

  1. Builds the per-agent Markov chain P_i defined by
         P_i[a, a'] = y_{(i, a, a')} / sum_{a''} y_{(i, a, a'')}
     and computes its stationary distribution x_i.  (Papadimitriou &
     Roughgarden Lemma 3.1 guarantees that the resulting product
     distribution x = prod_i x_i satisfies x^T U^T y = 0.)
  2. Uses the method of conditional expectations to pick a pure profile
     a* such that (U^T y)_{a*} >= 0.  This is the JLB 2011 correction:
     returning a pure profile instead of the product distribution gives
     a cut with bounded encoding size.

Signs and notation follow Papadimitriou--Roughgarden and
Jiang--Leyton-Brown:
  * y is indexed by triples (i, a_i, a'_i).
  * The constraint matrix U has
        U[(i, a_i, a'_i), a] = u_i(a_i, a_{-i}) - u_i(a'_i, a_{-i})
     when a's i-th coordinate is a_i, and 0 otherwise.
  * A correlated equilibrium x satisfies U x >= 0.  The dual we want
    to certify infeasible is { y >= 0 : U^T y <= 0 }.
  * f(a) := (U^T y)_a.  The oracle returns an a* with f(a*) >= 0.

The indexing convention:
  * Flat index of y for triple (i, a, a') uses the offset layout
        y_off(i) + a * m_i + a'
    where y_off(i) = sum_{j<i} m_j^2.  Diagonal entries (a == a') are
    included but always carry weight 0 in CE constraints; we preserve
    them to keep the arithmetic clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .compact_game import CompactGame


def y_dim(spec) -> int:
    """Return the dimension of the dual vector y."""
    return int(sum(m * m for m in spec.action_counts))


def y_offset(spec, agent: int) -> int:
    """Starting index of agent's block in the flat y vector."""
    offset = 0
    for i in range(agent):
        offset += spec.action_counts[i] ** 2
    return offset


def y_block(y: NDArray[np.float64], spec, agent: int) -> NDArray[np.float64]:
    """Return the m_i x m_i block Y_i where Y_i[a, a'] = y_{(i, a, a')}."""
    m = spec.action_counts[agent]
    off = y_offset(spec, agent)
    return y[off : off + m * m].reshape(m, m)


def pack_y(spec, per_agent_blocks: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Inverse of y_block: assemble a flat y from per-agent m_i x m_i matrices."""
    parts = []
    for i, blk in enumerate(per_agent_blocks):
        m = spec.action_counts[i]
        assert blk.shape == (m, m)
        parts.append(np.asarray(blk, dtype=np.float64).reshape(m * m))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Step 1: per-agent Markov chain + stationary distribution
# ---------------------------------------------------------------------------


def build_markov_chain(
    y: NDArray[np.float64], spec, agent: int, eps: float = 0.0
) -> NDArray[np.float64]:
    """Build the m x m row-stochastic Markov chain P_i.

        P_i[a, a'] = y_{(i, a, a')} / sum_{a''} y_{(i, a, a'')}

    Rows with zero total are kept stationary (P_i[a, a] = 1).

    If eps > 0, a small uniform perturbation is mixed in to guarantee a
    unique stationary distribution (useful in degenerate cases; the
    oracle then computes a slightly off x but this is only problematic
    when exact arithmetic is required -- for initial debugging we pass
    eps = 0).
    """
    m = spec.action_counts[agent]
    Y = y_block(y, spec, agent).astype(np.float64, copy=True)
    # Row sums
    row_sums = Y.sum(axis=1)
    P = np.zeros((m, m), dtype=np.float64)
    for a in range(m):
        if row_sums[a] > 0:
            P[a, :] = Y[a, :] / row_sums[a]
        else:
            # no outgoing mass -- stay put
            P[a, a] = 1.0
    if eps > 0:
        P = (1 - eps) * P + eps * (np.ones((m, m)) / m)
    return P


def stationary_distribution(P: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return a stationary distribution pi of the row-stochastic matrix P.

    Solves pi @ P = pi and sum(pi) = 1 via null-space of (P - I)^T.
    Returns the solution with all non-negative entries summing to 1.
    If there are multiple stationary distributions (chain is reducible)
    we return the one from the null-space solver normalized to sum to 1.
    """
    m = P.shape[0]
    # We want pi such that (P - I)^T pi = 0 and 1^T pi = 1.
    # Replace the last column of (P - I)^T with ones, rhs = (0, ..., 0, 1).
    A = (P - np.eye(m)).T.copy()
    A[-1, :] = 1.0
    b = np.zeros(m)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fall back: null space via SVD
        vals, vecs = np.linalg.eig(P.T)
        # find eigenvalue closest to 1
        idx = int(np.argmin(np.abs(vals - 1.0)))
        pi = np.real(vecs[:, idx])
        s = pi.sum()
        if s == 0:
            pi = np.ones(m) / m
        else:
            pi = pi / s
    # Clip small negatives from numerical noise
    pi = np.maximum(pi, 0.0)
    s = pi.sum()
    if s > 0:
        pi = pi / s
    else:
        pi = np.ones(m) / m
    return pi


def build_product_distribution(
    y: NDArray[np.float64], spec, eps: float = 0.0
) -> list[NDArray[np.float64]]:
    """Return per-agent product distributions x_1, ..., x_n from y.

    Using the notation Y_p[a, a'] = y_{(p, a, a')} and R_a = row-a sum of Y_p,
    the product distribution constructed from y is:

      1. T = row-normalize(Y_p)                  (row-stochastic Markov chain)
      2. v = stationary distribution of T       (v^T T = v^T)
      3. xp[a] = v[a] / R_a    when R_a > 0
      4. xp[a] = v[a]          when R_a == 0    (row is zero; v[a] must also be 0)
      5. Normalize xp to sum to 1.

    This is the Papadimitriou-Roughgarden Lemma 3.1 construction; see the
    derivation in the module docstring.  The key identity is that
        xp[k] * R_k = (xp^T Y_p)[k]    for all k
    which is exactly what E_{a ~ x}[f(a)] = 0 requires.
    """
    result = []
    for i in range(spec.n):
        Y_i = y_block(y, spec, i)
        m = spec.action_counts[i]
        R = Y_i.sum(axis=1)  # row sums, length m

        T = build_markov_chain(y, spec, i, eps=eps)
        v = stationary_distribution(T)

        xp = np.zeros(m)
        for a in range(m):
            if R[a] > 0:
                xp[a] = v[a] / R[a]
            else:
                # Row is zero: Markov chain has absorbing state at a.
                # v[a] could be any non-negative value.  Either way
                # xp[a] must satisfy R_a * xp[a] = 0, which is auto.
                # We pick xp[a] = v[a] (the stationary mass stays there).
                xp[a] = v[a]
        s = xp.sum()
        if s > 0:
            xp = xp / s
        else:
            xp = np.ones(m) / m
        result.append(xp)
    return result


# ---------------------------------------------------------------------------
# Step 2: f(a) = (U^T y)_a and its conditional expectations
# ---------------------------------------------------------------------------


def evaluate_f_pure(
    game: CompactGame,
    y: NDArray[np.float64],
    profile: tuple[int, ...],
) -> float:
    """Compute f(a) = (U^T y)_a for a pure profile a.

    f(a) = sum_i sum_{a'_i != a_i} y_{(i, a_i, a'_i)}
               * [u_i(a_i, a_{-i}) - u_i(a'_i, a_{-i})]
    """
    spec = game.spec
    total = 0.0
    for i in range(spec.n):
        a_i = profile[i]
        Y_i = y_block(y, spec, i)
        # Switch a_i to a'_i; profile - i part unchanged
        u_here = game.utility(i, profile)
        for a_prime in range(spec.action_counts[i]):
            if a_prime == a_i:
                continue
            w = float(Y_i[a_i, a_prime])
            if w == 0.0:
                continue
            prof_dev = list(profile)
            prof_dev[i] = a_prime
            u_dev = game.utility(i, tuple(prof_dev))
            total += w * (u_here - u_dev)
    return total


def expected_f_under_product(
    game: CompactGame,
    y: NDArray[np.float64],
    product_dist: list[NDArray[np.float64]],
) -> float:
    """E_{a ~ product_dist}[ f(a) ] = E[ (U^T y)_a ].

    This is 0 for any product distribution obtained from Lemma 3.1.
    Useful as a correctness invariant test.

    Fast path: if the game exposes `edge_use_probabilities` +
    `polynomial_expectation_fast` (e.g. CongestionGame), we precompute
    the edge-use state once and evaluate each (i, a_i, a') term in
    O(|route|) instead of the generic O(n · |E|).  For congestion games
    this turns the sample_pure_profile loop from O(n³ m³ |E|) into
    O(n² m³ |route|).
    """
    spec = game.spec
    fast = hasattr(game, "edge_use_probabilities") and hasattr(
        game, "polynomial_expectation_fast"
    )
    if fast:
        P, S = game.edge_use_probabilities(product_dist)
        # Cache expected utilities per (i, a): O(n · m · |route|)
        exp_u: list[list[float]] = [
            [
                game.polynomial_expectation_fast(i, a, P, S)
                for a in range(spec.action_counts[i])
            ]
            for i in range(spec.n)
        ]

    total = 0.0
    for i in range(spec.n):
        Y_i = y_block(y, spec, i)
        for a_i in range(spec.action_counts[i]):
            p_i = float(product_dist[i][a_i])
            if p_i <= 0:
                continue
            for a_prime in range(spec.action_counts[i]):
                if a_prime == a_i:
                    continue
                w = float(Y_i[a_i, a_prime])
                if w == 0.0:
                    continue
                if fast:
                    d = exp_u[i][a_i] - exp_u[i][a_prime]
                else:
                    d = game.utility_diff_expectation(
                        i, a_i, a_prime, product_dist
                    )
                total += p_i * w * d
    return total


def conditional_expected_f(
    game: CompactGame,
    y: NDArray[np.float64],
    product_dist: list[NDArray[np.float64]],
    fixed: dict[int, int],
) -> float:
    """Compute E[ f(a) | a_j = fixed[j] for j in fixed, a_k ~ x_k elsewhere ].

    Used by the method of conditional expectations to greedily select a
    pure profile with f(a*) >= 0.
    """
    # Build the distribution where the "fixed" agents are point masses.
    spec = game.spec
    cond_dist = []
    for i in range(spec.n):
        if i in fixed:
            pm = np.zeros(spec.action_counts[i])
            pm[fixed[i]] = 1.0
            cond_dist.append(pm)
        else:
            cond_dist.append(product_dist[i])
    return expected_f_under_product(game, y, cond_dist)


# ---------------------------------------------------------------------------
# Step 3: pure profile via conditional expectations
# ---------------------------------------------------------------------------


def sample_pure_profile(
    game: CompactGame,
    y: NDArray[np.float64],
    product_dist: list[NDArray[np.float64]],
    tol: float = -1e-12,
    rng: Optional[np.random.Generator] = None,
) -> tuple[int, ...]:
    """Return a pure profile a* with f(a*) >= 0 via conditional expectations.

    Method of conditional expectations: for each agent i in turn, fix a_i
    to ANY action whose conditional E[f | a_1..a_i fixed] is at least the
    running "parent" threshold.  Since the starting threshold is 0
    (E_x[f] = 0 by PR Lemma 3.1), maintaining the invariant guarantees
    f(a*) >= 0 at the end.

    To generate diverse pure profiles across iterations of the outer
    ellipsoid — essential for the reconstruction LP to have enough
    witnesses at large n — we randomize among all actions that meet the
    threshold (rather than always taking the argmax).  `rng` controls the
    randomization; pass a persistent Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Fast specialized path for games exposing a vectorized sampler
    # (e.g., CongestionGame.sample_pure_profile_fast).
    if hasattr(game, "sample_pure_profile_fast"):
        profile = game.sample_pure_profile_fast(y, product_dist, rng=rng)
        val = evaluate_f_pure(game, y, profile)
        assert val >= tol, (
            f"JLB invariant violated (fast path): f(a*) = {val}."
        )
        return profile

    spec = game.spec
    fixed: dict[int, int] = {}
    threshold = 0.0  # PR Lemma 3.1: E_x[f] = 0, so we only need ≥ 0
    for i in range(spec.n):
        candidates: list[tuple[int, float]] = []
        for a_i in range(spec.action_counts[i]):
            if product_dist[i][a_i] <= 0:
                continue
            fixed_try = dict(fixed)
            fixed_try[i] = a_i
            val = conditional_expected_f(game, y, product_dist, fixed_try)
            candidates.append((a_i, val))

        if not candidates:
            # No support action: fall back to argmax of x_i (degenerate)
            best_action = int(np.argmax(product_dist[i]))
            fixed[i] = best_action
            continue

        # Keep only actions whose conditional expectation is ≥ threshold
        # (within a small numerical slack).  There is always at least one
        # by PR Lemma 3.1 applied to the conditional distribution.
        max_val = max(v for _, v in candidates)
        eligible = [a for (a, v) in candidates if v >= threshold - 1e-10]
        if not eligible:
            # Numerical case: threshold unattainable due to float error.
            # Fall back to argmax.
            best_action = max(candidates, key=lambda av: av[1])[0]
            fixed[i] = best_action
            threshold = max_val
            continue

        # Randomize among eligible actions for profile diversity.
        choice = eligible[int(rng.integers(0, len(eligible)))]
        # Advance the threshold to the chosen action's conditional value.
        for a, v in candidates:
            if a == choice:
                threshold = v
                break
        fixed[i] = choice

    profile = tuple(fixed[i] for i in range(spec.n))
    val = evaluate_f_pure(game, y, profile)
    assert val >= tol, (
        f"JLB invariant violated: f(a*) = {val}, expected >= 0."
    )
    return profile


# ---------------------------------------------------------------------------
# Step 4: build the cut (column of U at a*) in flat y-space
# ---------------------------------------------------------------------------


def cut_from_profile(
    game: CompactGame, profile: tuple[int, ...]
) -> NDArray[np.float64]:
    """Return the column vector U_{.,a*} in R^{y_dim} for the pure profile a*.

    The entry at index (i, a_i, a'_i) is:
        u_i(a_i, a_{-i}) - u_i(a'_i, a_{-i})    if profile[i] == a_i
        0                                       otherwise.
    """
    spec = game.spec
    N = y_dim(spec)
    q = np.zeros(N, dtype=np.float64)
    for i in range(spec.n):
        a_i_star = profile[i]
        off = y_offset(spec, i)
        m_i = spec.action_counts[i]
        u_here = game.utility(i, profile)
        for a_prime in range(m_i):
            if a_prime == a_i_star:
                continue
            prof_dev = list(profile)
            prof_dev[i] = a_prime
            u_dev = game.utility(i, tuple(prof_dev))
            q[off + a_i_star * m_i + a_prime] = u_here - u_dev
    return q


# ---------------------------------------------------------------------------
# Public API: the oracle as a single function
# ---------------------------------------------------------------------------


@dataclass
class OracleResult:
    """Bundle the diagnostic outputs of one oracle call."""

    profile: tuple[int, ...]                  # pure profile a*
    product_dist: list[NDArray[np.float64]]   # per-agent stationary distributions
    cut: NDArray[np.float64]                  # column U_{.,a*} in flat y-space
    f_value: float                            # f(a*) >= 0, the cut's "slack"
    expected_f: float                         # should be ~0 (PR Lemma 3.1)


def jlb_separation_oracle(
    game: CompactGame,
    y: NDArray[np.float64],
    markov_eps: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> OracleResult:
    """Run one JLB 2011 oracle query for the dual candidate y >= 0.

    Parameters
    ----------
    game : CompactGame
        The game (exposes utility() and polynomial_expectation()).
    y : array of shape (y_dim(spec),)
        Candidate dual vector, must be entrywise non-negative.
    markov_eps : float, default 0
        Optional uniform perturbation mixed into the per-agent Markov
        chain to break degeneracy.  Usually leave at 0 for exact tests;
        set > 0 if the stationary distribution is ill-defined.

    Returns
    -------
    OracleResult with profile, product distribution, cut, f(a*),
    and E_x[f] (a correctness probe: should be ~0).
    """
    if np.any(y < 0):
        raise ValueError("y must be entrywise non-negative")

    product_dist = build_product_distribution(y, game.spec, eps=markov_eps)
    expected_f = expected_f_under_product(game, y, product_dist)
    profile = sample_pure_profile(game, y, product_dist, rng=rng)
    cut = cut_from_profile(game, profile)
    f_value = float(cut @ y)
    return OracleResult(
        profile=profile,
        product_dist=product_dist,
        cut=cut,
        f_value=f_value,
        expected_f=expected_f,
    )


__all__ = [
    "y_dim",
    "y_offset",
    "y_block",
    "pack_y",
    "build_markov_chain",
    "stationary_distribution",
    "build_product_distribution",
    "evaluate_f_pure",
    "expected_f_under_product",
    "conditional_expected_f",
    "sample_pure_profile",
    "cut_from_profile",
    "jlb_separation_oracle",
    "OracleResult",
]
