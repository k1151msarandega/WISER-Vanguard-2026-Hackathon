"""
End-to-end Week 2 pipeline: partition -> lock H -> QAOA-optimize O -> combine
with S (=0) -> compare once against the classical Markowitz baseline solved
on the *same* full universe under the *same* guardrails.

This is the "compared once against Markowitz" checkpoint from the Week 2
plan. It is explicitly a single-instance sanity check, not a rigor claim --
Week 3 replaces this with multi-seed, multi-window, equal-footing benchmarks.
Treat every number printed here as "does this look sane," not "is this good."
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vqportfolio.config import TICKERS, ASSET_CLASS_OF, ASSET_CLASS_CAPS
from vqportfolio.market_data.loader import load_prices
from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
from vqportfolio.partitioning.partition import (
    partition_assets, build_locked_allocation, PartitionConfig,
)
from vqportfolio.quantum.qubo import build_o_set_qubo
from vqportfolio.quantum.qaoa_solver import solve_with_qaoa_and_validate
from vqportfolio.baseline.markowitz import solve_markowitz
from vqportfolio.market_data.vanguard_calibration import nearest_lifestrategy_fund


@dataclass
class PipelineResult:
    partition: dict
    h_weights: pd.Series
    qaoa_full_weights: pd.Series  # H + repaired O, combined, indexed over full universe
    qaoa_matches_exact: bool
    repair_applied: bool
    markowitz_result: dict


def run_pipeline(
    dials: Dials | None = None,
    partition_config: PartitionConfig | None = None,
    risk_aversion: float = 3.0,
) -> PipelineResult:
    """Load market data fresh, then run the pipeline. For callers that
    already have mu/sigma/overlay/mdd cached (e.g. the Streamlit app, which
    loads once and reuses across reruns), use run_pipeline_with_data()
    instead to avoid redundant reloads -- this wrapper exists so every
    existing caller (tests, __main__, notebooks) keeps working unchanged."""
    dials = dials or Dials()
    partition_config = partition_config or PartitionConfig()

    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns)
    mdd = per_asset_max_drawdown(prices)

    result = run_pipeline_with_data(dials, partition_config, risk_aversion, mu, sigma, overlay, mdd)
    return result, used_synthetic, mu, sigma


def run_pipeline_with_data(
    dials: Dials,
    partition_config: PartitionConfig,
    risk_aversion: float,
    mu: pd.Series,
    sigma: pd.DataFrame,
    overlay: pd.DataFrame,
    mdd: pd.Series,
) -> PipelineResult:
    """Core pipeline logic, given already-loaded market data. Split out from
    run_pipeline() so callers who load data once (e.g. a Streamlit app
    caching across reruns) don't silently re-fetch on every call."""
    scores_df = compute_asset_scores(mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, dials)
    partition = partition_assets(scores_df["score"], partition_config)
    h_weights, o_budget = build_locked_allocation(mu, sigma, overlay["cost_bps"], partition, partition_config)

    class_headroom = {}
    for asset_class, cap in ASSET_CLASS_CAPS.items():
        idx = [t for t in partition["H"] if ASSET_CLASS_OF[t] == asset_class]
        used = h_weights.loc[idx].sum() if idx else 0.0
        class_headroom[asset_class] = max(cap - used, 0.0)

    qubo_result = build_o_set_qubo(
        partition["O"], mu, sigma, overlay["cost_bps"], o_budget, class_headroom,
        max_weight_per_asset=partition_config.max_weight_per_asset,
        risk_aversion=risk_aversion,
    )
    comparison = solve_with_qaoa_and_validate(qubo_result, o_budget, class_headroom)

    # Combine H + O(QAOA, repaired) + S(=0) into one full-universe weight vector.
    full_weights = pd.Series(0.0, index=TICKERS)
    full_weights.loc[h_weights.index] = h_weights.values
    full_weights.loc[comparison.qaoa_weights.index] += comparison.qaoa_weights.values
    # S is already 0 by construction (never touched)

    markowitz_result = solve_markowitz(
        mu, sigma, overlay["cost_bps"], TICKERS,
        risk_aversion=risk_aversion,
        max_weight_per_asset=partition_config.max_weight_per_asset,
    )

    return PipelineResult(
        partition=partition,
        h_weights=h_weights,
        qaoa_full_weights=full_weights,
        qaoa_matches_exact=comparison.qaoa_matches_exact,
        repair_applied=comparison.repair_applied,
        markowitz_result=markowitz_result,
    )


def portfolio_stats(weights: pd.Series, mu: pd.Series, sigma: pd.DataFrame) -> dict:
    w = weights.values
    equity_weight = float(weights[[t for t in weights.index if ASSET_CLASS_OF[t] == "Equities"]].sum())
    return {
        "expected_return": float(mu.values @ w),
        "risk_variance": float(w @ sigma.values @ w),
        "equity_weight": equity_weight,
        "nearest_vanguard_fund": nearest_lifestrategy_fund(equity_weight),
        "guardrail_breaches": {
            ac: round(weights[[t for t in weights.index if ASSET_CLASS_OF[t] == ac]].sum() - cap, 4)
            for ac, cap in ASSET_CLASS_CAPS.items()
            if weights[[t for t in weights.index if ASSET_CLASS_OF[t] == ac]].sum() > cap + 1e-4
        },
    }


if __name__ == "__main__":
    result, used_synthetic, mu, sigma = run_pipeline()

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}\n")
    print(f"H ({len(result.partition['H'])}): {result.partition['H']}")
    print(f"O ({len(result.partition['O'])}): {result.partition['O']}")
    print(f"S ({len(result.partition['S'])}): {result.partition['S']}\n")
    print(f"QAOA matched exact optimum on O-set: {result.qaoa_matches_exact}")
    print(f"Repair applied to O-set solution: {result.repair_applied}\n")

    hos_stats = portfolio_stats(result.qaoa_full_weights, mu, sigma)
    mw = result.markowitz_result

    print("=== H/O/S + QAOA (full portfolio) ===")
    print(f"Expected return: {hos_stats['expected_return']:.4f}")
    print(f"Risk (variance): {hos_stats['risk_variance']:.5f}")
    print(f"Equity weight: {hos_stats['equity_weight']:.1%}  -> resembles Vanguard's "
          f"{hos_stats['nearest_vanguard_fund'].name} ({hos_stats['nearest_vanguard_fund'].ticker}, "
          f"{hos_stats['nearest_vanguard_fund'].total_stock_pct:.0f}% stock)")
    print(f"Guardrail breaches: {hos_stats['guardrail_breaches']}")
    print(f"Sum of weights: {result.qaoa_full_weights.sum():.4f}\n")

    mw_equity_weight = float(mw["weights"][[t for t in mw["weights"].index if ASSET_CLASS_OF[t] == "Equities"]].sum())
    mw_nearest = nearest_lifestrategy_fund(mw_equity_weight)
    print("=== Classical Markowitz (full universe, same constraints) ===")
    print(f"Expected return: {mw['expected_return']:.4f}")
    print(f"Risk (variance): {mw['risk_variance']:.5f}")
    print(f"Sharpe ratio: {mw['sharpe_ratio']:.4f} (rf={mw['risk_free_rate']:.4%}, "
          f"{'FALLBACK' if mw['risk_free_rate_is_fallback'] else 'live FRED'})")
    print(f"Equity weight: {mw_equity_weight:.1%}  -> resembles Vanguard's "
          f"{mw_nearest.name} ({mw_nearest.ticker}, {mw_nearest.total_stock_pct:.0f}% stock)")
    print(f"Guardrail breaches: {mw['guardrail_breaches']}\n")

    print("Side-by-side weights:")
    comparison_df = pd.DataFrame({
        "H/O/S + QAOA": result.qaoa_full_weights,
        "Markowitz": mw["weights"],
    }).round(4)
    print(comparison_df[(comparison_df.T != 0).any()])
