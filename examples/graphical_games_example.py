"""Exact CE for three classical graphical games.

Graphical games are succinct: each agent's utility depends only on their
local neighbourhood.  Pure Nash is PPAD-hard in general (no shortcut
like Rosenthal's potential for congestion), so JLB is genuinely needed
for CE.

Three games, all solved in seconds even though the dense normal-form
would have m^n profiles:

    1. Graph coloring (5x5 grid, 3 colors)      M = 3^25 ≈ 8e11
    2. Best-shot public goods on 3-regular 20   M = 2^20 ≈ 10^6
    3. Ring majority game n=30                   M = 2^30 ≈ 10^9
"""

from __future__ import annotations

import time

from against_hope import (
    graph_coloring_game,
    best_shot_public_goods,
    ring_majority_game,
    grid_adjacency,
    cycle_adjacency,
    run_jlb_ellipsoid,
)


def report(label: str, M: int, fn) -> None:
    print(f"\n=== {label}  (M = {M:.2e} profiles) ===")
    t0 = time.time()
    res = fn()
    dt = time.time() - t0
    if res.reconstructed:
        supp = int((res.alpha > 1e-6).sum())
        print(f"  ✓ CE found  t={dt:.2f}s  iters={res.timings['iterations']}"
              f"  support={supp}  (among {len(res.profiles)} pure profiles)")
        # Print top 5 profile probabilities
        import numpy as np
        order = np.argsort(-res.alpha)
        print("  Top-5 profiles by probability:")
        for k in order[:5]:
            if res.alpha[k] < 1e-6:
                continue
            p = res.profiles[k]
            p_str = "".join(str(a) for a in p[:min(len(p), 30)])
            if len(p) > 30:
                p_str += "…"
            print(f"    α={res.alpha[k]:.3f}  profile={p_str}")
    else:
        print(f"  ✗ not reconstructed  t={dt:.2f}s  stop={res.stopped_reason}")


def main() -> None:
    # --- 1. Graph coloring on a 5×5 grid ---
    adj = grid_adjacency(5, 5)
    g = graph_coloring_game(adj, n_colors=3, collision_cost=1.0)
    report(
        f"Graph coloring  (5×5 grid, {len(adj)} nodes, 3 colors)",
        3 ** 25,
        lambda: run_jlb_ellipsoid(g, max_iterations=20_000,
                                    initial_radius=10.0,
                                    reconstruct_every=50),
    )

    # --- 2. Best-shot public goods on a 3-regular graph ---
    # Use a cycle-with-chords to get degree 3.
    n = 20
    cyc = cycle_adjacency(n)
    adj2 = [list(cyc[i]) + [(i + n // 2) % n] for i in range(n)]
    adj2 = [tuple(a) for a in adj2]
    g2 = best_shot_public_goods(adj2, contribution_cost=0.4, public_benefit=1.0)
    report(
        f"Best-shot public goods  (n={n}, degree-3 graph)",
        2 ** n,
        lambda: run_jlb_ellipsoid(g2, max_iterations=10_000,
                                     initial_radius=10.0,
                                     reconstruct_every=50),
    )

    # --- 3. Ring majority game n=30 ---
    g3 = ring_majority_game(n=30, m=2)
    report(
        "Ring majority  (n=30 on a cycle)",
        2 ** 30,
        lambda: run_jlb_ellipsoid(g3, max_iterations=20_000,
                                     initial_radius=10.0,
                                     reconstruct_every=50),
    )


if __name__ == "__main__":
    main()
