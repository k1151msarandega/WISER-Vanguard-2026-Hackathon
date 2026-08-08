"""
Walk-forward validation -- the Week 3 item flagged in the handoff notes as
"not started at all."

What this tests that nothing else in the project does: everything up to
this point (pipeline.py's single-instance comparison, classical_benchmarks,
multi_seed_variance) evaluates the H/O/S+QAOA pipeline on ONE fixed
mu/Sigma/overlay snapshot. That answers "is the optimization good, given
these inputs" but says nothing about "does re-deriving those inputs from a
rolling trailing window and re-solving actually produce a sensible,
out-of-sample-defensible portfolio over time." Walk-forward is the standard
backtesting answer to that: fit on a trailing window, hold the resulting
weights over a subsequent out-of-sample test window, roll forward, repeat.

Method:
  - Non-overlapping (train_window_days, test_window_days) blocks, stepped by
    test_window_days -- standard walk-forward, not an overlapping/rolling
    scheme, so no test period is ever reused across two periods and no
    result here double-counts the same realized return.
  - At each period: mu/Sigma/max-drawdown are recomputed from ONLY the
    trailing train window (real point-in-time inputs, not the full-history
    snapshot used everywhere else in the project) -- this is the actual
    walk-forward discipline: at decision time t, the pipeline only ever
    sees data up to t.
  - The full H/O/S+QAOA pipeline (partition -> lock H -> QAOA-solve O) and
    the Markowitz baseline are both re-solved at every period, under the
    identical trailing-window inputs, so the comparison stays fair/apples-
    to-apples exactly the way pipeline.py's single-shot comparison already
    is -- this just repeats that comparison through time instead of once.
  - Realized return over the subsequent test window is computed under a
    fixed-weight, notionally-daily-rebalanced convention: sum of the
    period's daily log returns dotted with the period's target weight
    vector, exponentiated. This is the standard tractable approximation
    used across walk-forward backtests (equivalent to "hold this static
    weight vector, rebalanced back to target every day") -- NOT a true
    buy-and-hold-with-drift model, where weights would drift within the
    period as constituent prices move. Documented here explicitly because
    it's a real modeling choice, not an oversight: implementing true
    buy-and-hold drift would require tracking a full daily weight path per
    period for no real benefit to what this analysis is trying to show
    (whether the *decision* the pipeline makes each period is any good).

Deliberate scope limitation, flagged rather than hidden: `overlay` (cost_bps
+ yield) is NOT re-derived per period from the trailing window -- it is
computed ONCE (via the same real Corwin-Schultz/yield pipeline used
elsewhere) and held fixed across all periods. Two different reasons, not
one: (1) yield is fetched as yfinance's *current* trailing dividend yield --
there is no simple way to ask yfinance for "what was the trailing yield as
of period t" for an arbitrary historical t, so a genuinely point-in-time
yield series isn't available without a different data source; (2)
Corwin-Schultz cost COULD in principle be recomputed per-period from a
windowed OHLC slice, but is deliberately not, to isolate what this analysis
is actually trying to test -- whether re-deriving mu/Sigma/drawdown from a
rolling window and re-solving produces sensible weights over time -- from a
second, separable question (does the cost/liquidity picture itself drift
enough over 2010-2026 to matter) that walk-forward-ing the overlay too
would conflate into the same numbers. Worth revisiting as a follow-up, not
done here.

QAOA is solved with a SINGLE fixed seed per period (not multi-seeded) --
seed variance at a given instance is already characterized separately in
multi_seed_variance.py; duplicating that here per period would multiply
runtime for no new information given that module's finding (near-zero
variance at the sizes tested with warm-start + multi-restart).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vqportfolio.config import TICKERS, ASSET_CLASS_OF, ASSET_CLASS_CAPS
from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
from vqportfolio.partitioning.partition import partition_assets, build_locked_allocation, PartitionConfig
from vqportfolio.market_data.overlays import compute_returns_and_risk
from vqportfolio.quantum.qubo import build_o_set_qubo
from vqportfolio.quantum.qaoa_solver import solve_with_qaoa_and_validate
from vqportfolio.baseline.markowitz import solve_markowitz


@dataclass(frozen=True)
class WalkForwardConfig:
    train_window_days: int = 756   # ~3 trading years -- enough history for a
                                    # non-degenerate 15-asset covariance estimate
    test_window_days: int = 63     # ~1 trading quarter -- the out-of-sample
                                    # hold/realize period, and the step size
                                    # (non-overlapping, see module docstring)
    dials: Dials = field(default_factory=Dials)
    partition_config: PartitionConfig = field(default_factory=PartitionConfig)
    risk_aversion: float = 3.0
    qaoa_seed: int = 42            # fixed across periods -- see module docstring
    n_restarts: int = 2
    maxiter: int = 40
    shots: int = 512


@dataclass
class PeriodResult:
    period_index: int
    train_start_idx: int
    train_end_idx: int             # exclusive; also test_start_idx
    test_end_idx: int              # exclusive
    o_set: list[str]
    o_budget: float
    hos_weights: pd.Series          # full-universe weights, H+O(QAOA,repaired)+S(=0)
    mw_weights: pd.Series
    hos_realized_return: float      # over the test window, fixed-weight convention
    mw_realized_return: float
    hos_guardrail_breaches: dict
    mw_guardrail_breaches: dict
    qaoa_matches_exact: bool
    repair_applied: bool
    solve_ok: bool
    error: str | None = None


def _fixed_weight_realized_return(weights: pd.Series, log_returns_window: pd.DataFrame) -> float:
    """Realized return of a fixed weight vector over a window of daily log
    returns, under the rebalanced-daily-to-target convention documented in
    the module docstring: exp(sum_t w . r_t) - 1."""
    w = weights.reindex(log_returns_window.columns).fillna(0.0).values
    daily_port_log_ret = log_returns_window.values @ w
    return float(np.exp(daily_port_log_ret.sum()) - 1.0)


def count_periods(n_days: int, config: WalkForwardConfig) -> int:
    """How many non-overlapping (train, test) periods fit in n_days --
    used by callers to plan chunking across multiple runs/tool calls
    without having to actually solve anything."""
    count = 0
    train_start = 0
    while train_start + config.train_window_days + config.test_window_days <= n_days:
        count += 1
        train_start += config.test_window_days
    return count


def run_walk_forward(
    prices: pd.DataFrame,
    overlay: pd.DataFrame,
    config: WalkForwardConfig | None = None,
    start_period: int = 0,
    max_periods: int | None = None,
) -> list[PeriodResult]:
    """Run the walk-forward sweep over `prices` (already loaded, real or
    synthetic-fallback -- caller's responsibility to have checked
    used_synthetic before trusting results). `overlay` is the (cost_bps,
    yield) frame computed once -- see module docstring for why it is not
    re-derived per period.

    `start_period`/`max_periods` let a long sweep be split across multiple
    calls (each QAOA-solve period is real wall-clock time, same reasoning
    as validation/run_ablation_at_size.py splitting by size): periods
    before `start_period` are skipped cheaply (just advancing the window,
    no solve), and at most `max_periods` are actually solved.
    """
    config = config or WalkForwardConfig()
    tickers = list(prices.columns)
    n_days = len(prices)

    results: list[PeriodResult] = []
    period_idx = 0
    train_start = 0
    while True:
        train_end = train_start + config.train_window_days
        test_end = train_end + config.test_window_days
        if test_end > n_days:
            break

        if period_idx < start_period:
            period_idx += 1
            train_start += config.test_window_days
            continue
        if max_periods is not None and (period_idx - start_period) >= max_periods:
            break

        train_prices = prices.iloc[train_start:train_end]
        test_prices = prices.iloc[train_end:test_end + 1]  # +1: need one extra row so
                                                             # log-return diff over the
                                                             # test window has test_window_days
                                                             # observations, not test_window_days-1

        try:
            mu, sigma, _ = compute_returns_and_risk(train_prices)
            mdd = per_asset_max_drawdown(train_prices)

            scores_df = compute_asset_scores(
                mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, config.dials,
            )
            partition = partition_assets(scores_df["score"], config.partition_config)
            h_weights, o_budget = build_locked_allocation(
                mu, sigma, overlay["cost_bps"], partition, config.partition_config, config.risk_aversion,
            )

            class_headroom = {}
            for asset_class, cap in ASSET_CLASS_CAPS.items():
                idx = [t for t in partition["H"] if ASSET_CLASS_OF[t] == asset_class]
                used = h_weights.loc[idx].sum() if idx else 0.0
                class_headroom[asset_class] = max(cap - used, 0.0)

            qubo_result = build_o_set_qubo(
                partition["O"], mu, sigma, overlay["cost_bps"], o_budget, class_headroom,
                max_weight_per_asset=config.partition_config.max_weight_per_asset,
                risk_aversion=config.risk_aversion,
            )
            comparison = solve_with_qaoa_and_validate(
                qubo_result, o_budget, class_headroom,
                seed=config.qaoa_seed, n_restarts=config.n_restarts,
                maxiter=config.maxiter, shots=config.shots,
            )

            hos_weights = pd.Series(0.0, index=tickers)
            hos_weights.loc[h_weights.index] = h_weights.values
            hos_weights.loc[comparison.qaoa_weights.index] += comparison.qaoa_weights.values

            mw_result = solve_markowitz(
                mu, sigma, overlay["cost_bps"], tickers,
                risk_aversion=config.risk_aversion,
                max_weight_per_asset=config.partition_config.max_weight_per_asset,
            )
            mw_weights = mw_result["weights"]

            test_log_returns = np.log(test_prices / test_prices.shift(1)).dropna(how="all")

            hos_realized = _fixed_weight_realized_return(hos_weights, test_log_returns)
            mw_realized = _fixed_weight_realized_return(mw_weights, test_log_returns)

            hos_breaches = {
                ac: round(hos_weights[[t for t in tickers if ASSET_CLASS_OF[t] == ac]].sum() - cap, 4)
                for ac, cap in ASSET_CLASS_CAPS.items()
                if hos_weights[[t for t in tickers if ASSET_CLASS_OF[t] == ac]].sum() > cap + 1e-4
            }

            results.append(PeriodResult(
                period_index=period_idx,
                train_start_idx=train_start,
                train_end_idx=train_end,
                test_end_idx=test_end,
                o_set=partition["O"],
                o_budget=o_budget,
                hos_weights=hos_weights,
                mw_weights=mw_weights,
                hos_realized_return=hos_realized,
                mw_realized_return=mw_realized,
                hos_guardrail_breaches=hos_breaches,
                mw_guardrail_breaches=mw_result["guardrail_breaches"],
                qaoa_matches_exact=comparison.qaoa_matches_exact,
                repair_applied=comparison.repair_applied,
                solve_ok=True,
            ))
        except Exception as e:
            results.append(PeriodResult(
                period_index=period_idx, train_start_idx=train_start, train_end_idx=train_end,
                test_end_idx=test_end, o_set=[], o_budget=float("nan"),
                hos_weights=pd.Series(dtype=float), mw_weights=pd.Series(dtype=float),
                hos_realized_return=float("nan"), mw_realized_return=float("nan"),
                hos_guardrail_breaches={}, mw_guardrail_breaches={},
                qaoa_matches_exact=False, repair_applied=False, solve_ok=False, error=str(e),
            ))

        period_idx += 1
        train_start += config.test_window_days

    return results


def summarize(results: list[PeriodResult]) -> dict:
    ok = [r for r in results if r.solve_ok]
    hos = np.array([r.hos_realized_return for r in ok])
    mw = np.array([r.mw_realized_return for r in ok])

    hos_cum = float(np.prod(1 + hos) - 1) if len(hos) else float("nan")
    mw_cum = float(np.prod(1 + mw) - 1) if len(mw) else float("nan")

    total_breaches_hos = sum(1 for r in ok if r.hos_guardrail_breaches)
    total_breaches_mw = sum(1 for r in ok if r.mw_guardrail_breaches)

    return {
        "n_periods": len(results),
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "hos_mean_period_return": float(np.mean(hos)) if len(hos) else float("nan"),
        "hos_std_period_return": float(np.std(hos)) if len(hos) else float("nan"),
        "hos_cumulative_return": hos_cum,
        "mw_mean_period_return": float(np.mean(mw)) if len(mw) else float("nan"),
        "mw_std_period_return": float(np.std(mw)) if len(mw) else float("nan"),
        "mw_cumulative_return": mw_cum,
        "hos_beats_mw_fraction": float(np.mean(hos > mw)) if len(hos) else float("nan"),
        "n_periods_hos_guardrail_breach": total_breaches_hos,
        "n_periods_mw_guardrail_breach": total_breaches_mw,
        "n_periods_qaoa_matched_exact": sum(r.qaoa_matches_exact for r in ok),
        "n_periods_repair_applied": sum(r.repair_applied for r in ok),
    }
