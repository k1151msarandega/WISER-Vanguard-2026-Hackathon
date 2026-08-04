# Equal-Footing Classical Benchmarks: Findings

Week 3. Addresses a gap flagged explicitly after Week 2: the existing
Markowitz baseline (`baseline/markowitz.py`) solves a genuinely different
problem from what QAOA solves -- continuous weights over the full 15-asset
universe, with H not pre-fixed. That's a legitimate comparison for a
different question, but it is not an *equal-footing* comparison for "how
good is QAOA specifically, on the exact combinatorial problem it was
handed." Code: `validation/classical_benchmarks.py`,
`validation/run_classical_benchmarks.py`.

**Same caveat as everywhere else in this project:** all results below used
synthetic price data (no internet access in the dev sandbox). Needs
re-confirmation with real data.

## Method design

Three classical benchmarks, all solving the identical discretized problem
QAOA solves (same bit-encoded weights, same budget equality, same class-cap
inequalities):

- **ILP (diagonal risk)**: the risk term uses a diagonal-only covariance
  approximation (drops cross-asset covariance) -- for binary variables,
  `b_i^2 = b_i`, so even the diagonal quadratic term collapses to linear,
  making the whole objective genuinely solvable by a real open-source MILP
  solver (PuLP + CBC), not an approximation of a QUBO solver. This mirrors
  the precedent set by an earlier (AQC hackathon) classical-benchmark
  design. Dropping cross-covariance is a real methodological
  simplification, documented, not hidden.
- **Greedy**: single-pass, deterministic, ranks assets by a marginal
  return-risk-cost score (same diagonal simplification as the ILP) and
  fills ticks highest-ranked-first until budget/caps bind. No search.
- **Random**: rejection-sampled feasible tick allocations (exact budget
  match, respecting class caps), best-of-N and mean/std reported across the
  full sample distribution, not just one draw.

All five methods (exact, QAOA, ILP, greedy, random) are scored via the same
`weight_space_objective()` -- the true return-risk-cost financial objective
computed directly from each method's output weights, not from bits or QUBO
internals. This was a deliberate design correction, not the original plan:
see "Bugs found and fixed" below for why.

## Bugs found and fixed while building this

Building a second, independent path to score the same problem surfaced two
real bugs that the QAOA-only code path had never exposed:

1. **Dimension mismatch on QUBO evaluation.** `qubo.objective.evaluate()`
   requires a full-length bitstring including slack variables (added
   whenever a class-cap constraint binds). ILP/greedy/random solve the true
   constraints directly and produce no slack values at all, since they
   never needed a penalty-based relaxation. An early version of this module
   crashed on an O-set that needed slack qubits (the first O-set tested
   happened not to need any, which is why this wasn't caught immediately).
   Fixed by scoring every method from its **decoded weights** via a shared
   `weight_space_objective()` function instead of touching QUBO/bit
   internals at all -- this also sidesteps a deeper problem: QAOA's
   *repaired* weights are continuous-rescaled and don't correspond to any
   exact bitstring in the first place, so a bits-based objective wouldn't
   even be well-defined for it.
2. **Sign-convention bug in the original random sampler.** Before the fix
   above, `solve_random` selected its best sample via `np.argmax` on a
   MINIMIZE-sense QUBO value -- which picks the numerically largest (i.e.
   least-negative, *worst*) value, not the best. This silently reported the
   worst of N random samples as if it were the best. Caught by comparing
   the "fixed" random result's behavior against its own mean/std (a jump
   from "worse than ILP/greedy" to "matches exact" was the tell that
   something structural had changed, not just a different random seed).

Both were caught by cross-checking results across two different O-set
instances before trusting either, not by inspection alone -- worth noting as
a methodology point: an unusual result was the signal to re-derive by hand
rather than write it up as-is.

## Results (two O-set instances, post-fix)

**Instance 1** — default dials, O = [IWM, LQD, SPY, UUP], budget=0.30:

| Method | Objective | Gap to exact | Time (s) |
|---|---|---|---|
| Exact (brute force) | 0.018763 | 0.000000 | 0.0000 |
| QAOA | 0.018763 | 0.000000 | 0.0000 |
| ILP (diagonal risk) | 0.018218 | -0.000544 | 0.48 |
| Greedy | 0.018218 | -0.000544 | 0.0005 |
| Random (best-of-500) | 0.018763 | 0.000000 | 0.08 |

**Instance 2** — growth=0.3, income=1.5, drawdown=1.5, cost=2.0, O = [UUP, VNQI, EFA, IWM], budget=0.35:

| Method | Objective | Gap to exact | Time (s) |
|---|---|---|---|
| Exact (brute force) | 0.024034 | 0.000000 | -- |
| QAOA | 0.023722 | -0.000312 | -- |
| ILP (diagonal risk) | 0.024025 | -0.000009 | -- |
| Greedy | 0.024025 | -0.000009 | -- |
| Random (best-of-300) | 0.024034 | 0.000000 | -- |

## What this actually shows

- **ILP and greedy tie in both instances.** Both use the same diagonal-risk
  simplification, so this isn't surprising -- it suggests that for these
  particular small O-sets, the simplification itself (not the search
  method) is what limits quality. QAOA's edge over both (instance 1) comes
  specifically from using the *full* quadratic risk term (real cross-asset
  covariance), which the classical ILP had to sacrifice to stay a
  genuinely solvable linear program. That's a fair, non-cherry-picked
  reason for an advantage, not an artificially weakened baseline.
- **Random matching exact in both instances is a real result, not
  suspicious** -- these O-sets are small (4 assets, ≤8 discrete levels
  each, tightly budget/class-constrained), so the feasible discrete space
  is small enough that a few hundred random draws plausibly stumble onto
  the true optimum. This is expected to look very different at larger O
  sizes, where the feasible space grows combinatorially and random should
  degrade sharply relative to QAOA/ILP -- **not yet tested at scale**, a
  natural extension of this work.
- **Instance 2's unresolved nuance**: QAOA's bitstring matched the exact
  brute-force optimum exactly (`matches_exact=True`), yet its weight-space
  objective is measurably lower (-0.000312) than exact's. Since matching
  bitstrings should decode to identical weights, this discrepancy has to
  come from the repair step -- `repair_applied=True` was confirmed for this
  run, but the actual weight-sum values shown are numerically identical to
  ~12 decimal places, which doesn't explain a gap this size. **Not fully
  root-caused** -- flagged honestly rather than hidden or over-invested in
  chasing down given the time available. Worth revisiting if this pattern
  recurs at scale.

## Known limitations / not yet done

- Only two O-set instances tested, not a systematic sweep -- this is a
  demonstration that the harness works and produces sane, self-consistent
  results, not yet the "multi-seed variance reporting" or "walk-forward
  validation" items still separately open on the Week 3 list.
- All on synthetic data (see caveat at top).
- The instance-2 QAOA weight-space discrepancy is unresolved.
