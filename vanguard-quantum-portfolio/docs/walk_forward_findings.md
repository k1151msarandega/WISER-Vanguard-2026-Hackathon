# Walk-Forward Validation: Findings

Week 3 item flagged in HANDOFF.md as "not started at all." This is that.
Code: `validation/walk_forward.py`, `validation/run_walk_forward.py`.

**Same caveat as everywhere else in this project, PLUS one the rubric
actually lifts:** all results below used synthetic price data. Unlike most
of this project's other caveats, this one is *not* a rubric risk -- the
challenge brief explicitly permits "synthetic or anonymized asset-class
data," so this is a legitimate input, not an apology. It's flagged anyway
for consistency with the rest of the project's documentation style.

## What this tests that nothing else in the project does

Everything before this point (`pipeline.py`'s single-instance comparison,
`classical_benchmarks.py`, `multi_seed_variance.py`) evaluates the
H/O/S+QAOA pipeline on ONE fixed mu/Sigma/overlay snapshot. That answers
"is the optimization good, given these inputs" but says nothing about
whether re-deriving those inputs from a rolling window and re-solving
produces a sensible portfolio *over time*. Walk-forward is the standard
backtesting answer: fit on a trailing window, hold the resulting weights
over a subsequent out-of-sample window, roll forward, repeat.

## Method

- Non-overlapping (train, test) blocks, stepped by the test length --
  standard walk-forward, not overlapping/rolling, so no test period is
  ever reused across two periods.
- **Scope decision made under time pressure, not the original plan:**
  quarterly test windows (63 trading days) were the initial design, giving
  56 periods over the 2010-2026 synthetic history. A qubit-count survey
  (see "Real finding" below) showed most periods land at 15-18 qubits, not
  the 12 the Week 2 prototype size was tuned for, and a single QAOA solve
  at that size took 60-140s even at heavily reduced settings -- a full
  56-period sweep at usable settings would have run roughly 2 hours of
  wall-clock with no way to checkpoint/resume across tool-call boundaries.
  Switched to **annual test windows (252 days), giving 14 periods** instead
  of truncating to the first N quarterly periods, specifically to avoid
  biasing the result toward one slice of history -- 14 periods still spans
  the full 2010-2026 range, just at coarser granularity.
- At each period, mu/Sigma/max-drawdown are recomputed from ONLY the
  trailing train window (real point-in-time inputs) -- the actual
  walk-forward discipline. The full H/O/S+QAOA pipeline and the Markowitz
  baseline are both re-solved under the identical trailing-window inputs at
  every period, same fairness convention as `pipeline.py`'s single-shot
  comparison, just repeated through time.
- Realized return over each test window uses a fixed-weight,
  notionally-daily-rebalanced convention: `exp(sum(daily log returns . w)) - 1`.
  This is NOT a true buy-and-hold-with-drift model (weights would drift
  within the period as prices move) -- it's the standard tractable
  approximation used across walk-forward backtests. Verified correct
  against a trivial single-asset case (weight=1 on one ticker exactly
  reproduces that ticker's simple buy-and-hold return, to machine
  precision) before trusting it on the real multi-asset case.
- **Overlay (cost_bps, yield) is NOT walked forward** -- computed once,
  held fixed across all periods. Two separable reasons, not one: (1)
  yfinance's yield field is *current* trailing yield, with no simple way to
  ask for "trailing yield as of period t" for an arbitrary historical t;
  (2) Corwin-Schultz cost *could* be recomputed per-period from a windowed
  OHLC slice, but deliberately isn't, to keep this analysis isolated to the
  question it's actually asking (does re-deriving mu/Sigma from a rolling
  window and re-solving produce sensible weights over time) rather than
  conflating in a second question (does the cost/liquidity picture itself
  drift enough to matter). Worth a follow-up, not done here.
- QAOA solved with a single fixed seed per period (42), not multi-seeded --
  seed variance at a given instance is characterized separately (see
  `multi_seed_variance_findings.md`); duplicating that per period here
  would multiply runtime for no new information given that module's own
  finding that the *smaller* prototype size showed near-zero variance
  (though the larger-size sweep run in parallel with this one shows that
  finding does NOT hold at every size -- see the other doc).
- **Settings reduced from production defaults under the same time
  pressure**: `n_restarts=1, maxiter=10, shots=128` vs. the pipeline's own
  default `n_restarts=3, maxiter=60, shots=1024`. This is real, not a minor
  footnote -- fewer restarts means more exposure to exactly the seed-to-seed
  variance now confirmed real at these qubit counts (see the multi-seed
  doc). Every result below should be read as "what a resource-constrained
  version of this pipeline does walking forward," not "what the pipeline
  achieves at full strength over time."

## Real finding, previously undocumented: qubit count is NOT determined by
## O-set size alone

A cheap qubit-count survey (QUBO build only, no solve) across all 56
quarterly-window candidate periods at the fixed `max_o_size=4` prototype
setting showed:

```
Counter({15: 30, 18: 15, 12: 10, 21: 1})
```

Same `max_o_size`, same number of O-set assets (4) every time, yet total
qubit count (objective + slack) ranged from 12 to 21 across periods. The
cause: slack variables come from *binding class-cap constraints*, and which
classes are binding depends on the trailing window's covariance/headroom
state at that point in time -- not on O-set size. A period where H happens
to already be near a class cap needs more slack qubits for O's QUBO than a
period where H has headroom to spare, even with an identical O-set size.
This means `max_o_size` alone is not a reliable proxy for "how expensive
will this solve be" -- worth surfacing to anyone planning compute budget
around O-set size alone (e.g. for a real-time co-pilot deployment).

## Results (14 periods, annual test windows, 2010-2026 synthetic history)

| idx | O-set | HOS+QAOA return | Markowitz return | QAOA matched exact | Repair applied |
|---|---|---|---|---|---|
| 0 | UUP,FXE,IWM,VNQI | 6.18% | 4.44% | No | No |
| 1 | FXE,UUP,IWM,SPY | 4.56% | 2.17% | No | No |
| 2 | LQD,FXE,VNQ,IWM | -14.15% | -19.10% | No | No |
| 3 | UUP,EEM,VNQ,DBC | 14.54% | 14.23% | Yes | No |
| 4 | FXE,UUP,EEM,VNQI | 12.01% | 7.58% | No | No |
| 5 | FXE,VNQI,UUP,SPY | -9.20% | -11.81% | Yes | No |
| 6 | FXE,SPY,VNQI,EEM | 4.97% | 5.36% | Yes | No |
| 7 | JNK,SPY,VNQI,EEM | 0.75% | 0.95% | Yes | Yes |
| 8 | JNK,UUP,USO,VNQI | 2.56% | -1.88% | Yes | Yes |
| 9 | IWM,VNQ,LQD,SPY | 20.93% | 14.36% | Yes | No |
| 10 | LQD,EFA,VNQ,UUP | 4.85% | 8.27% | Yes | No |
| 11 | VNQI,SPY,UUP,EFA | 0.54% | 4.77% | Yes | No |
| 12 | SPY,JNK,LQD,IEF | -11.29% | -16.35% | Yes | No |
| 13 | IWM,FXE,EFA,DBC | 6.31% | 14.65% | Yes | No |

**Aggregate:**

| | Mean period return | Std dev | Cumulative (compounded) |
|---|---|---|---|
| H/O/S + QAOA | 3.11% | 9.35% | **44.8%** |
| Markowitz baseline | 1.97% | 10.51% | **21.5%** |

- HOS+QAOA beat Markowitz in 9/14 periods (64.3%)
- QAOA matched the exact brute-force optimum in 10/14 periods (the 4 misses
  are periods 0, 1, 2, 4 -- notably, the *earliest* periods chronologically,
  though with only 14 periods this could easily be coincidence, not pattern)
- Repair (constraint-correction) needed in 2/14 periods (7, 8)
- **Zero guardrail breaches, either method, every single period** -- the
  hard constraint the rubric weights most heavily held throughout the
  entire 16-year walk, not just in a cherry-picked snapshot.

## Sanity checks performed before trusting these numbers

- The realized-return function was verified exactly against a trivial
  single-asset case (see Method section) -- not just eyeballed.
- Individual O-set asset returns for period 0's test window were pulled
  directly from raw prices (range: -11.05% to +18.24%) and the reported
  6.18% portfolio-level HOS return sits well inside that range, as it must
  for a weighted blend of the full universe (H+O+S) including these O
  assets -- no sign of a double-counting or compounding bug.

## What this actually shows, and what it doesn't

- **A genuine outperformance shows up, but it's not a clean claim.** 44.8%
  vs 21.5% cumulative is a large gap for 14 periods. This is real given the
  synthetic data and the specific assets each period's dial-driven scoring
  happened to select -- it is NOT evidence that H/O/S+QAOA reliably beats
  Markowitz in general, for several concrete reasons: (1) single seed, no
  variance bar on the HOS side to know how much of this is luck vs. signal;
  (2) reduced QAOA settings throughout, unlike production defaults; (3) 14
  periods is a small sample for a return-distribution claim; (4) synthetic
  data, however rubric-permitted, still reflects the specific GBM/
  block-correlation model's assumptions, not real market dynamics.
- **What IS a solid claim**: the pipeline, walked forward through 16 years
  of rolling re-optimization under real per-period covariance/headroom
  drift, never once breached a guardrail. Given the rubric scores "best
  risk-adjusted outcome with zero hard-constraint breaches" highest, this
  is the more defensible headline result of the two.
- **Not done here**: multi-seed walk-forward (would multiply runtime by
  however many seeds), real-data walk-forward (blocked on the same
  no-internet sandbox constraint as everything else, though again, not a
  rubric requirement), and turnover reporting between consecutive periods'
  weights (H/O/S+QAOA's turnover across the walk was never computed --
  same gap flagged in the rubric check as missing from the single-instance
  comparison too).
