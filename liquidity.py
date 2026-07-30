"""
Real liquidity tier computation, replacing the hand-guessed tiers in
config.py's AssetMeta (which were never based on anything -- notably, tier 3
was never even used across all 15 assets, a tell that they were eyeballed).

Methodology: average daily *dollar* volume (Volume x Close), not raw share
volume -- share counts alone aren't comparable across instruments at very
different price levels (e.g. GLD ~$180-220/share vs UUP ~$27/share trade
very differently in share-count terms for the same actual capital flow).
Dollar volume is the standard normalization for cross-asset liquidity
comparison.

Tiering is *relative to our own 15-asset universe* (tercile buckets of this
specific universe's dollar-volume distribution), not against fixed
market-wide thresholds -- this is a deliberate, documented choice: what
matters for our cost-proxy fallback is relative liquidity within the
assets we're actually choosing between, not where each ETF sits in the
global liquidity distribution of all tradable instruments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vqportfolio.config import TICKERS, LIQUIDITY_TIER_OF


def compute_liquidity_tiers(
    tickers: list[str],
    ohlc: pd.DataFrame,
    used_synthetic_ohlc: bool,
) -> tuple[dict[str, int], bool]:
    """Compute liquidity tiers (1=most liquid, 3=least) from real average
    daily dollar volume within our universe.

    Returns (tiers, used_fallback). Falls back to config.py's hand-guessed
    LIQUIDITY_TIER_OF only if the underlying OHLCV is itself synthetic --
    synthetic Volume carries no real liquidity signal (it's an arbitrary
    per-ticker lognormal draw, see loader._synthetic_ohlc), so computing
    "tiers" from it would just be guessing through an extra layer of
    indirection rather than actually fixing anything.
    """
    if used_synthetic_ohlc:
        return {t: LIQUIDITY_TIER_OF.get(t, 2) for t in tickers}, True

    dollar_volume = (ohlc["Volume"][tickers] * ohlc["Close"][tickers]).mean()

    if dollar_volume.isna().any() or (dollar_volume <= 0).any():
        # a specific ticker's volume data is degenerate even though the
        # overall pull was real -- fall back per-ticker, not universe-wide
        bad = dollar_volume[dollar_volume.isna() | (dollar_volume <= 0)].index.tolist()
        dollar_volume = dollar_volume.drop(index=bad)

    tiers = {}
    if len(dollar_volume) >= 3:
        # tercile buckets within our own universe
        q_high, q_low = dollar_volume.quantile([2/3, 1/3])
        for t in dollar_volume.index:
            if dollar_volume[t] >= q_high:
                tiers[t] = 1
            elif dollar_volume[t] >= q_low:
                tiers[t] = 2
            else:
                tiers[t] = 3
    else:
        for t in dollar_volume.index:
            tiers[t] = 2  # too few real observations to tercile meaningfully

    for t in tickers:
        if t not in tiers:
            tiers[t] = LIQUIDITY_TIER_OF.get(t, 2)  # per-ticker fallback for degenerate cases

    return tiers, False


if __name__ == "__main__":
    from vqportfolio.market_data.loader import load_ohlc

    ohlc, used_synthetic = load_ohlc()
    tiers, used_fallback = compute_liquidity_tiers(TICKERS, ohlc, used_synthetic)

    print(f"USED_SYNTHETIC_OHLC = {used_synthetic}")
    print(f"USED_FALLBACK_TIERS = {used_fallback}\n")

    if not used_fallback:
        dollar_volume = (ohlc["Volume"][TICKERS] * ohlc["Close"][TICKERS]).mean()
        print("Ticker  Avg $ Volume       Tier")
        for t in sorted(TICKERS, key=lambda x: -dollar_volume.get(x, 0)):
            dv = dollar_volume.get(t, float("nan"))
            print(f"  {t:6s} {dv:>15,.0f}   {tiers[t]}")
    else:
        print("(synthetic OHLCV -- tiers are the config.py hand-guessed fallback, "
              "run with internet access for real tiers)")
        for t in TICKERS:
            print(f"  {t}: {tiers[t]}")
