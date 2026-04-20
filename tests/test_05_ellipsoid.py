"""Tests for the standalone ellipsoid method.

Before testing it as the JLB outer loop, validate the ellipsoid
mechanics on simple, hand-verifiable LP feasibility problems.
"""

from __future__ import annotations

import numpy as np
import pytest

from against_hope.ellipsoid import Ellipsoid, ellipsoid_infeasible_loop


class TestEllipsoidBasics:
    def test_initial_ball(self):
        E = Ellipsoid.ball(N=3, radius=5.0)
        assert E.N == 3
        np.testing.assert_array_equal(E.c, np.zeros(3))
        np.testing.assert_array_equal(E.P, 25.0 * np.eye(3))
        assert E.log_vol_ratio == 0.0

    def test_cut_central(self):
        """Central cut (alpha=0): volume shrinks by the classical factor

            new_vol / old_vol = delta^(N/2) * sqrt(1 - sigma)

        where for alpha=0:  delta = N^2/(N^2-1),  sigma = 2/(N+1),
        giving  (N/(N+1)) * (N/sqrt(N^2-1))^{N-1}.
        """
        N = 4
        E = Ellipsoid.ball(N=N, radius=1.0)
        g = np.zeros(N); g[0] = 1.0
        status = E.apply_cut(g, h=0.0)
        assert status == "cut"

        # Expected log-volume-ratio for a central cut:
        sigma = 2.0 / (N + 1.0)
        delta = (N * N) / (N * N - 1.0)
        expected = 0.5 * (N * np.log(delta) + np.log(1.0 - sigma))
        assert np.isclose(E.log_vol_ratio, expected, atol=1e-12), (
            f"log_vol_ratio={E.log_vol_ratio}, expected {expected}"
        )
        # Specifically: for N=4, sigma = 2/5, delta = 16/15
        # Expected = 0.5 * (4 * log(16/15) + log(3/5)) ~ -0.127
        assert -0.15 < E.log_vol_ratio < -0.10

    def test_cut_deep_exact(self):
        """A cut with alpha very close to 1 should be nearly degenerate,
        and alpha >= 1 means the entire ellipsoid is outside the halfspace."""
        E = Ellipsoid.ball(N=2, radius=1.0)
        # g = e_1, h = -1.5.  At center 0, g^T c = 0 > -1.5, alpha = 1.5 / 1 = 1.5 > 1.
        g = np.array([1.0, 0.0])
        status = E.apply_cut(g, h=-1.5)
        assert status == "infeasible"

    def test_cut_redundant(self):
        """A cut that doesn't violate the center should be redundant."""
        E = Ellipsoid.ball(N=2, radius=1.0)
        # g = e_1, h = 10.  At center 0, g^T c = 0 < 10, so center is feasible,
        # alpha = (0 - 10) / 1 = -10 < -1.
        g = np.array([1.0, 0.0])
        status = E.apply_cut(g, h=10.0)
        assert status == "redundant"


class TestEllipsoidInfeasibilityCertification:
    def test_infeasible_LP_simple(self):
        """LP:  y_0 <= -1,  y_0 >= 1   (infeasible in R^1, extended to R^2).

        Oracle returns whichever constraint y violates.
        """
        calls = {"count": 0}

        def oracle(y):
            calls["count"] += 1
            if y[0] > -1:
                # y_0 <= -1 violated
                return np.array([1.0, 0.0]), -1.0
            elif y[0] < 1:
                # y_0 >= 1 violated -> -y_0 <= -1
                return np.array([-1.0, 0.0]), -1.0
            return None

        run = ellipsoid_infeasible_loop(
            N=2, separation_oracle=oracle, initial_radius=10.0,
            max_iterations=200, min_log_vol_ratio=-50.0,
        )
        assert run.stopped_reason in (
            "infeasible_certified", "volume_threshold", "redundant_loop"
        )
        assert run.iterations <= 200


class TestEllipsoidVolume:
    def test_volume_monotone_decreasing(self):
        """Volume should monotonically decrease under any non-redundant cut."""
        N = 4
        E = Ellipsoid.ball(N=N, radius=1.0)
        last_log = E.log_vol_ratio
        for i in range(N):
            g = np.zeros(N); g[i] = 1.0
            E.apply_cut(g, h=-0.01)  # mild non-trivial cut
            assert E.log_vol_ratio < last_log, (
                f"volume did not decrease at iter {i}: "
                f"log_vol={E.log_vol_ratio} >= previous {last_log}"
            )
            last_log = E.log_vol_ratio

    def test_shape_remains_PSD(self):
        """Under repeated cuts, P must remain positive semi-definite."""
        N = 5
        E = Ellipsoid.ball(N=N, radius=1.0)
        rng = np.random.default_rng(0)
        for _ in range(30):
            g = rng.standard_normal(N)
            h = float(g @ E.c) - 0.01  # small violation
            E.apply_cut(g, h=h)
            # Check eigenvalues of P
            w = np.linalg.eigvalsh(E.P)
            assert np.min(w) >= -1e-8, f"P lost PSD: min eig = {np.min(w)}"
