# Multi-Seed QAOA Variance: Findings

Week 3 item flagged in HANDOFF.md: module built (`validation/multi_seed_variance.py`,
`run_multi_seed_sweep.py`) and spot-tested at a small O-set size, but never
tested at a larger size -- "NOT yet tested at a larger size due to the
performance bug below eating the whole session before it got fixed
cleanly." That bug (see HANDOFF.md, and the fix in `qaoa_solver.py`'s
`_brute_force_exact`) is fixed; this is the larger-size test that was
blocked on it.

**Same caveat as elsewhere**: synthetic price data. Per the actual
challenge brief, synthetic data is explicitly permitted, so this is a
documented input choice, not an apology.

## Two sizes compared

### Small size (12 objective+slack qubits) -- prior result, referenced from HANDOFF.md

12 seeds, zero variance, all matched the exact brute-force optimum. This
was the result that motivated the original warm-start + multi-restart fix
(replacing single-seed random-init QAOA, which an earlier 8-seed spot check
had shown catastrophically failing 1/7 times) -- and at this size, the fix
appears to fully work: every seed lands on the exact answer.

### Larger size (18 objective+slack qubits) -- new result

Same instance used throughout `classical_benchmarks_findings.md`'s
Instance 1 (O = [IWM, LQD, SPY, UUP], `max_o_size=5` picks a 5th asset
[VNQI] relative to that doc's 4-asset O-set, giving 18 total qubits here
vs. 12 there) -- fixed QUBO, only the QAOA seed varies across runs.

**Settings reduced from production defaults**, same reason as the
walk-forward doc: `n_restarts=1, maxiter=15, shots=256` vs. the pipeline's
default `n_restarts=3, maxiter=60, shots=1024`. A single seed at these
*reduced* settings still took 60-140s; production settings would have made
a 10-seed sweep impractical given the session's tool-call-boundary
constraints (see walk-forward doc's method section for the same
constraint). This means the variance measured below is likely an
**overestimate** of production-setting variance -- fewer restarts means
less protection against a bad individual optimization run, which is
exactly the failure mode multi-restart was built to guard against in the
first place. This isn't a caveat to explain away; it likely means the real
picture at production settings is better than what's reported here, not
worse.

10 seeds (0-9):

| Seed | Objective | Gap to exact | Matched exact | Repair applied |
|---|---|---|---|---|
| 0 | 0.018741 | 0.000021 | No | No |
| 1 | 0.016916 | 0.001847 | No | No |
| 2 | 0.018741 | 0.000021 | No | No |
| 3 | 0.016916 | 0.001847 | No | No |
| 4 | 0.017953 | 0.000810 | No | No |
| 5 | 0.018741 | 0.000021 | No | No |
| 6 | 0.018762 | 0.000000 | No | Yes |
| 7 | 0.017503 | 0.001259 | No | No |
| 8 | 0.018741 | 0.000021 | No | No |
| 9 | 0.017679 | 0.001083 | No | No |

Exact (brute-force) objective: 0.018763.

**Aggregate**: mean objective 0.018069, std 0.000737, min 0.016916, max
0.018762. Mean gap to exact 0.000693, max gap 0.001847. **0/10 seeds
matched the exact bitstring** (vs. 12/12 at the smaller size). 1/10 needed
repair.

Only 6 distinct objective values across 10 seeds (0.016916, 0.017503,
0.017679, 0.017953, 0.018741, 0.018762) -- several seeds converge to the
exact same local optimum (seeds 0/2/5/8 all land on 0.018741; seeds 1/3 both
land on 0.016916), consistent with a COBYLA local search from a
warm-started but still-randomized starting point landing in a handful of
recurring basins rather than a continuum of distinct outcomes.

## What this actually shows

- **Variance is real at this size, and it wasn't at the smaller one.** This
  directly updates HANDOFF.md's open question: multi-restart QAOA's
  reliability is not a fixed property of the method -- it degrades as
  problem size grows, at least between 12 and 18 qubits. Zero seeds hitting
  the exact optimum at 18 qubits (vs. 12/12 at 12 qubits) is a clear,
  qualitative shift, not a marginal one.
- **This is very likely settings-related, not purely size-related** -- see
  the reduced-settings caveat above. `n_restarts=1` at 18 qubits is a
  meaningfully weaker search than `n_restarts=3` at 12 qubits; this result
  characterizes "QAOA at reduced settings, 18 qubits" specifically, not
  "QAOA at any settings, 18 qubits." Disentangling how much of the
  size-12-to-18 degradation is intrinsic (larger landscape, more local
  optima) vs. settings-driven (fewer restarts to escape them) is a natural
  next step, not done here given time constraints.
- **No seed produced a guardrail breach or a wildly bad outcome** -- unlike
  the original pre-warm-start catastrophic-failure pattern from Week 2
  (1/7 seeds landing ~60x worse than the others), every seed here lands
  within ~10% of the exact objective (worst case 0.016916 vs exact
  0.018763, a 9.8% relative gap). The warm-start bias is still doing real
  work at this size -- it's preventing catastrophe, just not guaranteeing
  exactness the way it did at the smaller size.
- **Connects to `classical_benchmarks_findings.md`'s unresolved Instance 2
  nuance** (QAOA matching exact's bitstring but scoring measurably
  differently). This sweep didn't reproduce that specific paradox (every
  non-matching seed here scored *worse* than exact, which is the expected
  direction, not the same anomaly), but it's the same general territory --
  worth keeping in mind that these two findings docs are looking at
  related-but-not-identical open questions about repair-step fidelity.

## Known limitations / not yet done

- Only one problem instance tested at the larger size, not a sweep across
  multiple O-set instances (unlike `classical_benchmarks_findings.md`'s two
  instances) -- risk that this specific instance's landscape isn't
  representative.
- Reduced settings throughout, as flagged -- a production-settings rerun
  (even at just 3-5 seeds) would meaningfully sharpen this finding.
- No test at sizes between 12 and 18, or above 18 -- the "12 qubits: perfect,
  18 qubits: real variance" comparison is two data points, not a curve.
  Given `mps_scaling_findings.md`'s own finding that circuit-depth cost
  scales sharply and independently of qubit count, and this project's new
  finding (walk-forward doc) that qubit count itself varies unpredictably
  with slack, a proper variance-vs-size curve would need to control for
  both qubit count and depth separately -- out of scope for the remaining
  time before submission.
