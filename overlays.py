"""
Derive model inputs from price data:
  - annualized expected return (mu) and covariance (Sigma) from Close prices
  - transaction cost proxy per asset via the Corwin-Schultz (2012) high-low
    spread estimator, computed from real OHLC when available
  - trailing dividend yield per asset, fetched from yfinance when available

Both cost and yield now source real data first, falling back to a documented
synthetic estimate only when the real source is unreachable (no internet) or
genuinely degenerate (e.g. flat OHLC in a data glitch) -- not as the default
path. `overlay.attrs['cost_synthetic']` / `overlay.attrs['yield_synthetic']`
report which path was actually used, so nothing is silently approximated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vqportfolio.config import ASSET_CLASS_OF, LIQUIDITY_TIER_OF
from vqportfolio.market_data.loader import load_ohlc
from vqportfolio.market_data.liquidity import compute_liquidity_tiers


def compute_returns_and_risk(prices: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Return (annualized mu, annualized covariance Sigma, daily log returns)."""
    log_returns = np.log(prices / prices.shift(1)).dropna(how="all")
    mu = log_returns.mean() * 252
    sigma = log_returns.cov() * 252
    return mu, sigma, log_returns


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> float:
    """Corwin & Schultz (2012) bid-ask spread estimator from daily high/low.

    Uses overlapping 2-day windows: for each pair of consecutive days,
    combines the two 1-day high-low log-ranges (beta) with the 2-day
    high-low log-range (gamma) to back out an implied spread. Per-day
    negative estimates (a known artifact of the estimator, not a data
    error) are clipped to zero before averaging, which is the standard
    treatment in the literature.
    """
    log_hl = np.log(high / low)
    beta = (log_hl ** 2) + (log_hl.shift(-1) ** 2)

    high_2d = pd.concat([high, high.shift(-1)], axis=1).max(axis=1)
    low_2d = pd.concat([low, low.shift(-1)], axis=1).min(axis=1)
    gamma = np.log(high_2d / low_2d) ** 2

    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))

    spread = spread.clip(lower=0.0)
    return float(spread.mean(skipna=True))


def compute_cost_bps(
    tickers: list[str],
    ohlc: pd.DataFrame | None = None,
    ohlc_is_synthetic: bool | None = None,
) -> tuple[pd.Series, bool]:
    """Per-asset transaction cost proxy in bps, via Corwin-Schultz on real
    OHLC when possible. Falls back to the old liquidity-tier heuristic only
    if OHLC is unavailable or degenerate (e.g. High == Low throughout,
    which would make the estimator divide by ~1 and produce garbage).

    If `ohlc` is passed in externally, `ohlc_is_synthetic` must be passed
    too -- otherwise there's no way to know whether the caller's OHLC was
    real or synthetic, and silently assuming "real" would let the fallback
    path try to compute "real" liquidity tiers from synthetic volume data.
    """
    if ohlc is None:
        ohlc, used_synthetic = load_ohlc(tickers)
    else:
        if ohlc_is_synthetic is None:
            raise ValueError("ohlc_is_synthetic must be provided when ohlc is passed explicitly")
        used_synthetic = ohlc_is_synthetic

    cost_bps = {}
    degenerate = False
    for t in tickers:
        try:
            high, low = ohlc["High"][t], ohlc["Low"][t]
            if (high <= low).mean() > 0.5:  # more than half the days flat/inverted -> bad data
                raise ValueError("degenerate high/low series")
            spread = corwin_schultz_spread(high, low)
            if not np.isfinite(spread):
                raise ValueError("non-finite spread estimate")
            cost_bps[t] = spread * 10_000
        except Exception:
            degenerate = True
            cost_bps[t] = np.nan

    result = pd.Series(cost_bps)
    if result.isna().any():
        # fill any degenerate tickers with the liquidity-tier heuristic --
        # using REAL tiers (computed from real Volume) when the underlying
        # OHLC is real, not the config.py hand-guess. Only degenerate to the
        # hand-guess if the OHLC itself is synthetic (real tiers from
        # synthetic volume would carry no real signal either).
        fallback = _liquidity_tier_cost_heuristic(tickers, ohlc, used_synthetic)
        result = result.fillna(fallback)
        degenerate = True

    return result, (used_synthetic or degenerate)


def _liquidity_tier_cost_heuristic(
    tickers: list[str],
    ohlc: pd.DataFrame,
    used_synthetic_ohlc: bool,
) -> pd.Series:
    """Liquidity-tier base spread. Only used per-ticker when Corwin-Schultz
    can't be computed (missing/degenerate OHLC), not as the default path.
    Tiers come from compute_liquidity_tiers() -- real average dollar volume
    when OHLC is real, config.py's hand-guess only when it isn't."""
    base_spread_bps = {1: 2.0, 2: 6.0, 3: 15.0}
    tiers, _ = compute_liquidity_tiers(tickers, ohlc, used_synthetic_ohlc)
    return pd.Series({t: base_spread_bps[tiers[t]] for t in tickers})


def fetch_real_yield(tickers: list[str]) -> tuple[pd.Series, bool]:
    """Trailing dividend yield per asset via yfinance. Falls back to a
    synthetic asset-class-appropriate band only if yfinance itself is
    unreachable (no internet) -- a genuinely zero yield (e.g. GLD, USO,
    currency ETFs typically pay ~0%) is a real value, not a fallback
    trigger, and is kept as 0.0 rather than replaced."""
    try:
        import yfinance as yf

        yields = {}
        for t in tickers:
            info = yf.Ticker(t).info
            y = info.get("dividendYield") or info.get("yield") or 0.0
            # yfinance has historically reported this as either a fraction
            # (0.02) or a percentage (2.0) depending on version -- normalize.
            if y > 1.0:
                y = y / 100.0
            yields[t] = float(y)

        if all(v == 0.0 for v in yields.values()):
            # every single ticker came back zero -- almost certainly means
            # the info call failed silently (offline) rather than a real
            # all-zero-yield universe. Fall back rather than trust this.
            raise RuntimeError("all yields came back zero -- likely offline")

        return pd.Series(yields), False
    except Exception:
        return _synthetic_yield(tickers), True


def _synthetic_yield(tickers: list[str], seed: int = 11) -> pd.Series:
    rng = np.random.default_rng(seed)
    yield_band = {
        "Equities": (0.010, 0.025),
        "Fixed Income": (0.025, 0.055),
        "Commodities": (0.000, 0.005),
        "Currencies": (0.000, 0.010),
        "Alternatives": (0.030, 0.060),
    }
    yields = {}
    for t in tickers:
        lo, hi = yield_band[ASSET_CLASS_OF[t]]
        yields[t] = rng.uniform(lo, hi)
    return pd.Series(yields)


def compute_cost_and_yield(
    tickers: list[str],
    log_returns: pd.DataFrame,
    seed: int = 11,
) -> pd.DataFrame:
    """Combined cost + yield + liquidity-tier overlay. `log_returns` is
    accepted for call-site compatibility with the old synthetic-only
    version but is no longer the source of cost -- kept as an
    unused-but-accepted parameter so existing call sites don't all need
    signature changes.

    `overlay.attrs['cost_synthetic']` / `overlay.attrs['yield_synthetic']`
    / `overlay.attrs['liquidity_tier_synthetic']` report which path was
    actually used for each field.
    """
    ohlc, used_synthetic_ohlc = load_ohlc(tickers)

    cost_bps, cost_synthetic = compute_cost_bps(tickers, ohlc=ohlc, ohlc_is_synthetic=used_synthetic_ohlc)
    yield_series, yield_synthetic = fetch_real_yield(tickers)
    liquidity_tiers, tier_synthetic = compute_liquidity_tiers(tickers, ohlc, used_synthetic_ohlc)

    overlay = pd.DataFrame({
        "ticker": tickers,
        "asset_class": [ASSET_CLASS_OF[t] for t in tickers],
        "cost_bps": cost_bps.reindex(tickers).values,
        "yield": yield_series.reindex(tickers).values,
        "liquidity_tier": [liquidity_tiers[t] for t in tickers],
    }).set_index("ticker")

    overlay.attrs["cost_synthetic"] = cost_synthetic
    overlay.attrs["yield_synthetic"] = yield_synthetic
    overlay.attrs["liquidity_tier_synthetic"] = tier_synthetic
    return overlay


if __name__ == "__main__":
    from vqportfolio.market_data.loader import load_prices
    from vqportfolio.config import TICKERS

    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns)

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}")
    print(f"USED_SYNTHETIC_COST   = {overlay.attrs['cost_synthetic']}")
    print(f"USED_SYNTHETIC_YIELD  = {overlay.attrs['yield_synthetic']}\n")
    print("Annualized expected return (mu):")
    print(mu.round(4), "\n")
    print("Annualized volatility (sqrt of diag of Sigma):")
    print(np.sqrt(np.diag(sigma)).round(4), "\n")
    print("Cost/yield overlay:")
    print(overlay.round(4))
