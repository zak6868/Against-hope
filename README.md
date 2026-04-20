# against_hope — exact correlated equilibria for succinct games

`against_hope` is a practical, open-source implementation of the
**Jiang & Leyton-Brown (2011)** polynomial-time algorithm for computing
**exact correlated equilibria** of compactly-represented games —
without enumerating the exponentially-sized normal form.

> To our knowledge, this is the first publicly-available implementation
> of JLB 2011, and the first exact-CE solver for polynomial-type
> succinct games that bypasses normal-form enumeration.

---

## Why this exists

Most CE solvers (`nashpy`, `gambit`, `CorrelatedEquilibria.jl`, …)
work on the **normal form**: one scalar per joint-action profile.  A
game with `n` agents and `m` actions each has `m^n` profiles.  Already
at `n = 50, m = 3` that's `3⁵⁰ ≈ 10²⁴` — you can't allocate the
tensor, let alone solve the LP.

Many interesting games are **succinct**: their description is
polynomial in `n` even though the normal form is exponential —
congestion, graphical, anonymous, action-graph, … For such games,
Jiang & Leyton-Brown showed that CE is still computable in **time
polynomial in the succinct description**.  

---

## 60-second quickstart

```bash
git clone <repo>
cd against_hope
pip install -e .
```

```python
from against_hope import pigou_game, run_jlb_ellipsoid, verify_ce

# 50 agents choosing between 2 parallel edges
# (normal form would have 2^50 ≈ 10^15 profiles — unusable).
g = pigou_game(n_agents=50)

result = run_jlb_ellipsoid(g, max_iterations=100_000)
assert result.reconstructed

# Verify the output is a genuine CE (every incentive constraint satisfied)
v = verify_ce(g, result.profiles, result.alpha)
assert v.is_ce, f"max violation = {v.max_violation}"

print(f"CE support size: {(result.alpha > 1e-6).sum()}")
```


---

## Step-by-step usage

### 1. Pick or build a game

**Built-in congestion games:**
```python
from against_hope import pigou_game, braess_game, sioux_falls_game

g = pigou_game(n_agents=100)
g = braess_game(n_agents=50, free_shortcut=True)
g = sioux_falls_game(n_agents=200, grid_size=3, k_routes=3)
```

**Built-in graphical games:**
```python
from against_hope import (
    graph_coloring_game, best_shot_public_goods,
    ring_majority_game, grid_adjacency, cycle_adjacency,
)

g = graph_coloring_game(grid_adjacency(5, 5), n_colors=3)
g = best_shot_public_goods(cycle_adjacency(20))
g = ring_majority_game(n=30, m=2)
```

**Your own succinct game** — subclass `CompactGame`:
```python
from against_hope import CompactGame, GameSpec

class MyGame(CompactGame):
    def __init__(self, ...):
        self.spec = GameSpec.from_action_counts((m1, m2, ..., mn))
        ...
    def utility(self, agent: int, profile: tuple[int, ...]) -> float:
        """u_i(a) for a pure profile — must run in poly(succinct-rep-size)."""
        ...
    def polynomial_expectation(self, agent, a_i, product_dist) -> float:
        """E_{a_{-i} ~ x_{-i}}[u_i(a_i, a_{-i})] in poly time.

        Required for PR/JLB to apply.  See `congestion_game.py` and
        `graphical_games.py` for worked examples.
        """
        ...
```

### 2. Solve for an exact CE

```python
from against_hope import run_jlb_ellipsoid

result = run_jlb_ellipsoid(
    game,
    max_iterations=100_000,     # safety ceiling; early-stops on convergence
    initial_radius=10.0,        # dual-space starting ball; conservative default
    reconstruct_every=50,       # try LP reconstruction every K new cuts
    seed=0,
)
```

Returns:
- `result.profiles` — list of pure joint profiles (the CE support).
- `result.alpha` — their convex-combination weights (sum to 1, ≥ 0).
- `result.reconstructed` — True iff a CE was found within the budget.
- `result.timings["iterations"]` — how many inner iterations were needed.

### 3. Verify correctness

```python
from against_hope import verify_ce

v = verify_ce(game, result.profiles, result.alpha, tol=1e-6)
if not v.is_ce:
    print(f"worst constraint (agent={v.worst_violation[0]}, "
          f"a={v.worst_violation[1]}, a'={v.worst_violation[2]})"
          f" gain = {v.max_violation:.3e}")
```

### 4. Use the CE

Sample recommendations for each agent:
```python
import numpy as np
rng = np.random.default_rng(0)
k = rng.choice(len(result.profiles), p=result.alpha)
recommended = result.profiles[k]     # one joint profile from the mixture
```

Under the CE guarantee, no agent wants to deviate from their
recommendation given the conditional distribution.

---

## Algorithm — what's happening inside

The pipeline implements Jiang & Leyton-Brown 2011, which in turn
builds on Papadimitriou & Roughgarden 2008.  Three layers:

### Layer 1 — CE feasibility as an LP

A correlated equilibrium is a probability distribution `x` over joint
action profiles satisfying, for every agent `i`, action `a_i`, and
alternative `a'_i`:

```
Σ_{a_{-i}}  x(a_i, a_{-i}) · [ u_i(a_i, a_{-i}) - u_i(a'_i, a_{-i}) ]  ≥  0
```

Stacking these into a matrix inequality gives the **primal LP**:

```
    find x ∈ Δ(A)  such that  U x  ≥  0       (primal: CE always exists)
```

where `U` is the CE-constraint matrix with rows indexed by triples
`(i, a_i, a'_i)` and columns indexed by joint profiles `a`.

For succinct games `x` has `m^n` entries — unusable.  Fortunately the
number of constraints (rows of `U`) is only `Σᵢ mᵢ² = O(n · m²)`,
polynomial.

### Layer 2 — Dual LP and its infeasibility

By LP duality, the **dual LP** asks:

```
    find y ≥ 0, y ≠ 0  with  Uᵀ y ≤ 0
```

For any finite game with a CE (always, for finite games), this dual is
**infeasible**.  Certifying infeasibility is how we'll get our CE
witness.

Enter the **separation oracle** of Papadimitriou & Roughgarden
(Lemma 3.1), corrected by Jiang & Leyton-Brown (2011):

> Given any `y ≥ 0`, in polynomial time (in the succinct game
> description) return a pure profile `a*` such that `(Uᵀ y)_{a*} ≥ 0`.

Existence of such an `a*` is guaranteed for every `y ≥ 0`, because
`E_{a ~ π(y)}[Uᵀy]_a = 0` for a carefully-constructed product
distribution `π(y)` (PR Lemma 3.1).  One of the profiles in the
support of `π(y)` must satisfy the inequality.  JLB's correction
ensures we can *deterministically extract* such a pure profile, not
just a distribution.

### Layer 3 — Ellipsoid method on the dual

We run the **deep-cut ellipsoid** in `y`-space:

1. Start with an ellipsoid containing the (always infeasible) feasible
   region `{ y : Uᵀy ≤ 0, y ≥ 0, y ≠ 0 }`.
2. Query the PR/JLB oracle at the current ellipsoid center.
3. The returned profile gives a cut `(Uᵀy)_{a*} ≥ 0` that the current
   center violates (otherwise infeasibility would already be proved).
4. Shrink the ellipsoid along this cut.
5. Repeat until the ellipsoid collapses (infeasibility certified) or
   the CE reconstruction LP becomes feasible (see Layer 4).

Each iteration collects one pure profile `a*`.  After enough cuts, the
set of profiles collected is rich enough to span the CE polytope.

### Layer 4 — CE reconstruction

The collected profiles `a^(1), a^(2), …, a^(L)` are candidate CE
support points.  We solve the **small LP**:

```
    find α ≥ 0,  Σ α = 1,  U'α ≥ 0
```

where `U'` has the `L` columns of `U` corresponding to the collected
profiles.  `U'` is `O(n · m²) × L` — polynomial in size.

If feasible, `α` gives the CE mixture weights over `{a^(k)}`.
Otherwise we keep collecting profiles.

### Complexity

Polynomial in the succinct representation size, with three nested
costs:

| Step | Cost |
|---|---|
| Per-oracle call (PR/JLB sampling) | `O(n · m² · C_poly_expect)` |
| Per-ellipsoid iteration | above + `O(N²)` for the rank-1 ellipsoid update |
| Iterations to infeasibility | `O(N² · log(R/ε))` in theory; often `O(N)` in practice |
| Final reconstruction LP | `O(poly(L, N))` via any LP solver |

where `N = Σᵢ mᵢ²` is the dual-space dimension and
`C_poly_expect` is the cost of one `polynomial_expectation` call on
your game.

---

## Scaling, measured

On Sioux-Falls-style grid games (3×3 grid, 3 routes per agent):

| n   | y_dim (N) | per-iter | iters | wall time |
|-----|-----------|----------|-------|-----------|
| 20  | 180       | 1 ms     | 1 141 | 0.3 s     |
| 50  | 450       | 1 ms     | 2 000 | 1 s       |
| 100 | 900       | 2.4 ms   | 39 000| 90 s      |

Normal form at n=100 would need ~10⁴⁷ profiles — physically
impossible.  `against_hope` handles it on a laptop.

---

## Examples

Three worked examples in `examples/`:

- `pigou_example.py` — congestion on two parallel edges, `n ∈ {10, 20, 50}`.
- `graphical_games_example.py` — graph coloring on a 5×5 grid, best-shot
  public goods on a 3-regular graph, ring majority with n=30.
- `custom_game_example.py` — subclass `CompactGame` for a ring graphical
  coordination game with `n = 30`.

Run any of them:
```bash
PYTHONPATH=. python examples/pigou_example.py
```

---

## Tests

61 unit + integration tests, all passing.  Includes:

- Per-module correctness (ellipsoid, PR/JLB oracle, CE extraction).
- Exact-CE verification on every sample output (`verify_ce`).
- Ground-truth brute-force comparison on small games.

```bash
pytest tests/
```

---

## Caveats (honest)

This is a *reference* implementation, not a certified-exact one.

| What | Impact |
|------|--------|
| **Float64, not gmpy2 rationals** | Output is exact modulo ≈ 10⁻¹² precision, not literal rational. |
| **Initial ellipsoid radius `R = 10`** | Paper requires `R = u^{5N³}` for strict theoretical certificate. We use a practical value. |
| **Cut normalization** | `(g, h) ← (g, h) / ‖g‖` per iteration — pragmatic, not in paper. |
| **LP reconstruction** | Returns a feasible α, not a basic-feasible solution. Support size bound `|supp| ≤ N+1` not guaranteed. |
| **Affine costs only** | For congestion games; polynomial costs (BPR) are an easy extension (moment recursion). |
| **No SCIP exact mode** | The paper's exact-arithmetic certification path is not wired up. |

Points (1) and (2) together prevent a strict "exact CE certificate"
claim in the paper-theoretical sense.  For most applications they are
irrelevant — the output is a valid CE in practice, verified by
`verify_ce`.

---

## Citation

```bibtex
@inproceedings{jiang2011polynomial,
  title     = {Polynomial-time computation of exact correlated equilibrium
               in compact games},
  author    = {Jiang, Albert Xin and Leyton-Brown, Kevin},
  booktitle = {Proceedings of the 12th ACM conference on Electronic commerce},
  year      = {2011},
}

@article{papadimitriou2008computing,
  title   = {Computing correlated equilibria in multi-player games},
  author  = {Papadimitriou, Christos H and Roughgarden, Tim},
  journal = {Journal of the ACM},
  year    = {2008},
}
```

## License

MIT.
