"""Compute a CE of Pigou n=50 — a game normal-form can't even represent.

At n=50 the joint profile space has 2^50 ≈ 10^15 profiles.  Standard
CE solvers that enumerate profiles (nashpy, gambit, etc.) cannot
allocate the tensor.  JLB solves it in seconds.
"""

from __future__ import annotations

import time

from against_hope import pigou_game, run_jlb_ellipsoid


def main() -> None:
    for n in [10, 20, 50]:
        g = pigou_game(n_agents=n)
        print(f"\nPigou n={n}: M = 2^{n} ≈ {2**n:.1e} profiles")
        t0 = time.time()
        res = run_jlb_ellipsoid(
            g,
            max_iterations=100_000,
            initial_radius=10.0,
            reconstruct_every=100,
        )
        dt = time.time() - t0
        if res.reconstructed:
            supp = int((res.alpha > 1e-6).sum())
            print(f"  ✓ CE found  t={dt:.2f}s  iters={res.timings['iterations']}"
                  f"  support={supp}")
        else:
            print(f"  ✗ max_iter  t={dt:.2f}s")


if __name__ == "__main__":
    main()
