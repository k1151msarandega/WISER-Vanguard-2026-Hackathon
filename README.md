# Vanguard Quantum for Finance — Multi-Asset Portfolio Construction

Solo submission, WISER Global Quantum+AI Program 2026. Deadline: **Aug 7, 2026**.

## Repo layout

```
vanguard-quantum-portfolio/
├── pyproject.toml          # pip install -e .
├── src/vqportfolio/        # the actual package
│   ├── config.py           # asset universe, sector caps, constants
│   ├── market_data/        # loader (yfinance + synthetic fallback), overlays (returns/vol/cost/yield)
│   ├── baseline/           # classical Markowitz baseline
│   ├── partitioning/       # Week 2: H/O/S logic
│   ├── quantum/            # Week 2: QUBO/QAOA
│   └── validation/         # Week 3: rigor layer (multi-seed, walk-forward, scaling)
├── notebooks/              # what you open in Colab
├── app/                    # Week 3-4: portfolio co-pilot demo
├── tests/
└── data/                   # local cache only, gitignored
```

## Workflow: GitHub + Colab

1. **Push this repo to GitHub** as-is — everything under `data/raw/` and
   `data/processed/` is gitignored, so no market data or generated results get
   committed by accident.
2. **In Colab**, each notebook starts with:
   ```python
   !git clone https://github.com/YOUR_USERNAME/vanguard-quantum-portfolio.git
   %cd vanguard-quantum-portfolio
   !pip install -e . -q
   ```
   This makes `vqportfolio` importable exactly like any installed package —
   no path hacks, no `sys.path.append`.
3. **Locally**, same install command: `pip install -e .` from the repo root,
   then `from vqportfolio... import ...` works from anywhere.

## Critical caveat: real vs. synthetic data

Every data-touching function returns or prints a `used_synthetic` /
`USED_SYNTHETIC_PRICES` flag. **Before trusting any number for the actual
writeup, confirm this is `False`.** It will be `True` if run somewhere without
internet access to Yahoo Finance (e.g. a sandboxed dev environment) — the
fallback exists so the pipeline can be built and tested without blocking on
network access, not as a data source for real results.

| Field | Source |
|---|---|
| Prices / returns / volatility / correlation | Real (yfinance) when internet is available; synthetic GBM fallback otherwise |
| Transaction cost proxy (`cost_bps`) | Synthetic — liquidity-tier spread scaled by realized vol |
| Yield (`yield`) | Synthetic — asset-class-appropriate band, feeds the income dial |
| Sector/asset-class labels | Manually assigned static reference data |

## Status

- [x] **Week 1** (Jul 14–20): data pipeline + classical Markowitz baseline,
      validated (`tests/test_baseline_sanity.py` — monotonic risk vs. risk
      aversion, zero guardrail breaches, no degenerate concentration)
- [x] **Week 2** (Jul 21–27): H/O/S partitioning reworked for continuous
      binarized weights + growth/income/drawdown/cost dials
      (`partitioning/scoring.py`, `partitioning/partition.py`); QUBO
      formulation + QUBO→Ising conversion (`quantum/qubo.py`); QAOA
      prototype (`quantum/qaoa_solver.py`, p=1, validated against exact
      brute-force enumeration since the toy instance is small enough);
      end-to-end pipeline compared once against Markowitz
      (`pipeline.py`) — validated (`tests/test_week2_sanity.py`)
- [ ] **Week 3** (Jul 28–Aug 3): equal-footing classical benchmarks, multi-seed
      variance reporting, walk-forward validation, scaling analysis, co-pilot
      demo skeleton
- [ ] **Week 4** (Aug 4–7): buffer, writeup, packaging

### Real data sourcing (post-Week-2)

An explicit hunt to replace hardcoded/invented values with real, cited data
before continuing further. Not everything closed — documented honestly below.

**Closed:**
- **Real risk-free rate**: FRED `DGS3MO`, `market_data/risk_free_rate.py`.
  Live pull when reachable; cached fallback is a real cited value (3.70% as
  of 2026-05-12), not invented. Wired into `solve_markowitz()` — the
  baseline now reports a real Sharpe ratio.
- **Real guardrail calibration**: `market_data/vanguard_calibration.py`
  contains Vanguard's own published LifeStrategy fund family (Income 20/80,
  Conservative Growth ~41/59, Moderate Growth ~61/39, Growth 80/20 —
  stock/bond), fetched directly from Vanguard's fact sheets. Used to widen
  `ASSET_CLASS_CAPS`' Equities/Fixed Income caps from invented values (60%/
  55%, which were actually *tighter* than Vanguard's own real Income and
  Growth funds) to 80%/80%, matching the real observed range. Also wired
  into `pipeline.py`'s reporting via `nearest_lifestrategy_fund()` — every
  portfolio now reports which real Vanguard fund it most resembles.
- **Real liquidity tiers**: `market_data/liquidity.py`, `compute_liquidity_tiers()`.
  Computed from real average daily *dollar* volume (Volume × Close, the
  standard cross-asset liquidity normalization), tercile-bucketed within
  our own 15-asset universe. `config.py`'s old hand-guessed tiers are now
  fallback-only (used only when the underlying OHLCV is itself synthetic --
  computing "real" tiers from synthetic volume would carry no real signal
  and just add a layer of false confidence). Wired into `overlays.py`'s
  Corwin-Schultz fallback path and exposed directly as a `liquidity_tier`
  column in the main cost/yield overlay.

- **HF/DuckDB dataset** (`defeatbeta/yahoo-finance-data`), `market_data/loader.py`,
  `load_ohlc()` / `load_ohlc_with_sources()`. Hybrid sourcing: HF/DuckDB
  (queried directly via predicate pushdown, no full-file download) for the
  9 tickers it covers (SPY, IWM, GLD, DBC, USO, UUP, FXE, TLT,
  JNK-replacing-HYG — real history back to 1996 in some cases, updated
  through 2026-07-27), yfinance for the remaining 6 (EFA, EEM, IEF, LQD,
  VNQ, VNQI), synthetic only as a last resort for whatever neither source
  provides. `load_ohlc()` keeps its original `(df, bool)` signature for
  backward compatibility; `load_ohlc_with_sources()` exposes the full
  per-ticker breakdown ('hf'/'yfinance'/'synthetic') for transparency.
  Schema and data quality were verified by hand before writing any parsing
  code, not assumed: a `DESCRIBE` query confirmed the real column names
  (`symbol, report_date, open, close, high, low, volume`), a >15%
  single-day-move sweep across all 9 tickers flagged only USO around the
  March/April 2020 oil crash, and a direct look at USO's close price
  through its actual April 2020 reverse-split date showed a smooth
  progression with no ~8x discontinuity — confirming `close` is
  split-adjusted, not raw, before trusting it in returns/volatility
  calculations. The merge/pivot logic itself was tested against a local
  mock file matching the confirmed schema before being pointed at the real
  dataset, since this sandbox can't reach huggingface.co to test the real
  path directly.
- One thing to actually check once this runs with real internet access (not
  verifiable from here): HF and yfinance are independent sources merged via
  an outer join on date -- worth confirming there's no meaningful date-range
  or trading-calendar mismatch at the boundary between HF-sourced and
  yfinance-sourced tickers before trusting cross-asset covariance numbers.

**Explicitly NOT closed, not glossed over:**
- Commodities/Currencies/Alternatives caps (20%/15%/20%) remain **reasoned
  judgment calls**. Vanguard's own core balanced-fund lineup doesn't hold
  these asset classes at all, so there is no real Vanguard policy document
  to calibrate against — confirmed by checking, not assumed.

### Post-audit fixes (post-Week-2, pre-Week-3)

Before moving to Week 3, we did a critical audit of Week 2 rather than assume
it was ready to build on. Five real issues were found and fixed — not just
documented as caveats:

1. **Real cost data.** `cost_bps` now comes from the actual Corwin-Schultz
   (2012) high-low spread estimator run on real OHLC data (`load_ohlc()` +
   `corwin_schultz_spread()` in `market_data/overlays.py`), not a crude
   vol-scaled guess. Falls back to the old heuristic only per-ticker, only
   if OHLC is unreachable or degenerate.
2. **Real yield data.** `yield` now comes from yfinance's real trailing
   dividend yield (`fetch_real_yield()`), not a randomly sampled band. A
   guard against "all yields came back zero" catches the offline case
   without misclassifying legitimately zero-yield assets (GLD, USO, currency
   ETFs) as failures.
3. **Dial conflation fixed.** The `drawdown` dial previously applied its
   full weight to *both* variance and max-drawdown silently — moving one
   dial secretly moved two things. `Dials.drawdown_variance_share` now makes
   that split explicit and documented (`partitioning/scoring.py`).
4. **H allocation is now a real optimization, not a heuristic.**
   `water_filling_allocate()` (proportional-to-score heuristic) is no longer
   the default. `optimize_locked_allocation()` solves the same
   mean-variance-cost QP as the Markowitz baseline, restricted to H's
   membership, with the budget as a ceiling (not a forced total) — provably
   optimal for its own objective, and it can leave H's members at zero
   weight if that's genuinely optimal (something water-filling could never
   do; see `partitioning/partition.py`).
5. **QAOA reliability fixed.** An 8-seed sweep on the old single-seed
   random-init QAOA found 5 exact-optimum hits and 1 catastrophic failure
   (objective gap of 1.07, ~60x worse than the "good" seeds) — the original
   Week 2 result was a lucky seed, not a representative one. Replaced with
   **warm-start QAOA** (initial quantum state biased toward a classical
   relaxation of the QUBO, via `relax_and_warm_start()`) + **multi-restart**
   (best-of-3). Note: we deliberately did *not* adopt the XY-mixer/
   Dicke-state constrained-mixer approach flagged in the Week 2 literature
   review — that technique fits cardinality-selection problems (choose K of
   N), not our binarized-continuous-weight encoding, and copying it in
   anyway would have been cutting a different corner. Post-fix spot check
   (3 seeds): 0/3 failures, all exact-optimum. Not a full statistical claim
   (each solve takes ~1-2 min at practical settings) — a proper multi-seed
   variance analysis is Week 3's job as a batch run.

### Known simplifications still open (deliberate, not hidden)

- Risk/drawdown terms in the H/O/S *scoring* function (which asset ranks
  where, before allocation) are still per-asset (marginal), not
  portfolio-level — a conviction score, not a substitute for the QUBO's
  actual covariance-aware objective or H's now-optimized allocation.
- O-set cost term assumes building from cash (no turnover-vs-prior-O-weights
  term yet) — fine for a cold-start prototype, needs revisiting once
  walk-forward rebalancing is in scope.
- `max_o_size=4` and `bits=3` were chosen for brute-force-validation
  convenience during prototyping, not because that's the "right" size for
  the real (larger, real-data) universe — Week 3's scaling analysis and the
  eventual real-data run should pick this deliberately.
- The brief separately lists "Diversification," "Liquidity," and "Sector or
  exposure limits" as distinct constraint types (see the actual challenge
  brief). We only implement asset-class-level caps — literal liquidity
  constraints and finer sector granularity aren't in yet.
- AI-assistance disclosure for the submission is still open — see the
  brief's explicit requirement that AI tool use be documented and all
  submitted work defensible as the team's own.

## Guardrails enforced (hard constraints in the classical baseline)

- Per-asset weight cap: 25%
- Asset-class exposure caps: Equities 60%, Fixed Income 55%, Commodities 20%,
  Currencies 15%, Alternatives 20%
- Turnover cap (configurable)

## Install

```bash
pip install -e .              # core
pip install -e ".[quantum]"   # + qiskit, for Week 2
pip install -e ".[app]"       # + streamlit, for Week 3-4
```
