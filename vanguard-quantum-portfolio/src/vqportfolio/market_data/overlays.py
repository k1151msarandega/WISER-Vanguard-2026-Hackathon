"""
Derive model inputs from price data:
  - annualized expected return (mu) and covariance (Sigma) from real/synthetic prices
  - synthetic transaction cost proxy per asset
  - synthetic trailing yield per asset (feeds the "income" dial later)

Cost and yield are labeled synthetic deliberately -- Vanguard's brief allows
"synthetic or anonymized data," and true bid-ask/yield feeds require paid data
we don't have. Keeping them in a clearly separate function (not mixed into
the real price series) means anyone auditing the pipeline can see exactly
which fields are real market data and which are modeled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vqportfolio.config import ASSET_CLASS_OF, LIQUIDITY_TIER_OF


def compute_returns_and_risk(prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return (annualized mu, annualized covariance Sigma, daily log returns)."""
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    mu = log_returns.mean() * 252
    sigma = log_returns.cov() * 252
    return mu, sigma, log_returns


def synthetic_cost_and_yield(
    tickers: list[str],
    log_returns: pd.DataFrame,
    seed: int = 11,
) -> pd.DataFrame:
    """Generate a per-asset synthetic cost proxy and trailing yield.

    Cost proxy: liquidity-tier-based base spread (bps) scaled by realized
    volatility, loosely mirroring the intuition behind the Corwin-Schultz
    high-low spread estimator (more volatile / less liquid -> wider spread)
    without requiring intraday high/low data we don't have in all cases.

    Yield: asset-class-appropriate trailing yield band (fixed income and
    alternatives yield more than growth equities, currencies ~0), sampled
    per-asset so the income dial has something real to trade off against risk.
    """
    rng = np.random.default_rng(seed)
    realized_vol = log_returns.std() * np.sqrt(252)

    base_spread_bps = {1: 2.0, 2: 6.0, 3: 15.0}  # by liquidity tier
    yield_band = {
        "Equities": (0.010, 0.025),
        "Fixed Income": (0.025, 0.055),
        "Commodities": (0.000, 0.005),
        "Currencies": (0.000, 0.010),
        "Alternatives": (0.030, 0.060),
    }

    rows = []
    for t in tickers:
        tier = LIQUIDITY_TIER_OF[t]
        cls = ASSET_CLASS_OF[t]
        vol = realized_vol.get(t, 0.15)

        cost_bps = base_spread_bps[tier] * (1 + vol)  # wider spread when vol is higher
        lo, hi = yield_band[cls]
        yld = rng.uniform(lo, hi)

        rows.append({"ticker": t, "asset_class": cls, "cost_bps": cost_bps, "yield": yld})

    return pd.DataFrame(rows).set_index("ticker")


if __name__ == "__main__":
    from vqportfolio.market_data.loader import load_prices
    from vqportfolio.config import TICKERS

    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = synthetic_cost_and_yield(TICKERS, log_returns)

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}\n")
    print("Annualized expected return (mu):")
    print(mu.round(4), "\n")
    print("Annualized volatility (sqrt of diag of Sigma):")
    print(np.sqrt(np.diag(sigma)).round(4), "\n")
    print("Cost/yield overlay:")
    print(overlay.round(4))
