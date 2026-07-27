"""
Per-asset conviction scoring for H(old)/O(ptimize)/S(kip) partitioning.

This replaces the AQC hold/sell scoring function (a single Sharpe-style
score) with Vanguard's four named, user-facing dials: growth, income,
drawdown control, cost sensitivity. Each raw input is on a wildly different
scale (expected return ~0.01-0.09, cost in bps ~2-8, max drawdown ~0.1-0.6),
so every component is z-scored across the universe before being combined --
without this, cost or drawdown would numerically dominate or vanish purely
because of units, not because of the dial weights the user actually set.

Known simplification, documented rather than hidden: the risk and drawdown
terms here are *marginal* (per-asset variance / per-asset historical max
drawdown), not portfolio-level (which depends on covariance and weights).
This is a deliberate scope choice for the partitioning/scoring stage -- it's
a conviction score used to decide what's confidently H/S, not the final
portfolio risk number. Portfolio-level risk is still computed properly by
the QUBO objective (Week 2 quantum stage) and the Markowitz baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Dials:
    """User-facing tunable goals. Values are relative weights, not required
    to sum to 1 -- they get z-scored components combined linearly, so only
    relative magnitude matters."""
    growth: float = 1.0
    income: float = 0.5
    drawdown: float = 1.0
    cost: float = 0.5
    stability: float = 0.3  # preference for keeping existing holdings (turnover-aversion)


def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def per_asset_max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """Historical max drawdown per asset (positive number, e.g. 0.35 = -35% peak-to-trough)."""
    running_max = prices.cummax()
    drawdown = 1 - prices / running_max
    return drawdown.max()


def compute_asset_scores(
    mu: pd.Series,
    sigma: pd.DataFrame,
    cost_bps: pd.Series,
    yield_series: pd.Series,
    max_drawdown: pd.Series,
    dials: Dials,
    prev_weights: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute a per-asset conviction score plus its z-scored components
    (returned alongside the score for explainability -- the co-pilot demo
    needs to be able to say *why* an asset scored the way it did, not just
    the final number)."""
    tickers = mu.index
    marginal_variance = pd.Series(np.diag(sigma.loc[tickers, tickers]), index=tickers)

    z_growth = _zscore(mu)
    z_risk = _zscore(marginal_variance)       # higher variance -> higher z -> subtracted
    z_cost = _zscore(cost_bps.loc[tickers])   # higher cost -> higher z -> subtracted
    z_income = _zscore(yield_series.loc[tickers])
    z_drawdown = _zscore(max_drawdown.loc[tickers])  # higher drawdown -> higher z -> subtracted

    if prev_weights is not None:
        z_stability = prev_weights.reindex(tickers).fillna(0.0)
    else:
        z_stability = pd.Series(0.0, index=tickers)

    score = (
        dials.growth * z_growth
        - dials.drawdown * z_risk  # variance is part of "risk," folded into drawdown-control dial's spirit
        - dials.cost * z_cost
        + dials.income * z_income
        - dials.drawdown * z_drawdown
        + dials.stability * z_stability
    )

    return pd.DataFrame({
        "score": score,
        "z_growth": z_growth,
        "z_risk": z_risk,
        "z_cost": z_cost,
        "z_income": z_income,
        "z_drawdown": z_drawdown,
        "z_stability": z_stability,
    }).sort_values("score", ascending=False)


if __name__ == "__main__":
    from vqportfolio.market_data.loader import load_prices
    from vqportfolio.market_data.overlays import compute_returns_and_risk, synthetic_cost_and_yield
    from vqportfolio.config import TICKERS

    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = synthetic_cost_and_yield(TICKERS, log_returns)
    mdd = per_asset_max_drawdown(prices)

    dials = Dials()
    scores = compute_asset_scores(mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, dials)

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}\n")
    print(scores.round(3))
