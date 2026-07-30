"""
Risk-free rate, sourced from FRED (DGS3MO -- 3-Month Treasury Constant
Maturity), same real-data-first pattern as market_data/loader.py: try a live
pull, fall back to a cached, dated, cited value only if unreachable.

FRED publishes the full series as CSV with no auth/key required:
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO
This sandbox can't reach fred.stlouisfed.org (network restricted to package
registries), so the live path is written for local/Colab use; the fallback
constant is a real, cited, point-in-time value, not an invented one --
confirmed by fetching https://fred.stlouisfed.org/series/DGS3MO directly:
  "2026-05-12: 3.70" (Percent, Not Seasonally Adjusted, Daily)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Real, cited fallback -- NOT an invented number, just not live.
_FALLBACK_RATE = 0.0370
_FALLBACK_DATE = "2026-05-12"
_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"


def load_risk_free_rate_series(start: str | None = None, end: str | None = None) -> tuple[pd.Series, bool]:
    """Load the full DGS3MO daily series as a decimal (not percentage) rate.
    Returns (series, used_fallback). FRED marks missing trading days with
    '.', which this drops rather than forward-fills, to avoid silently
    inventing rate data on holidays."""
    try:
        raw = pd.read_csv(_FRED_CSV_URL)
        raw.columns = ["date", "rate"]
        raw = raw[raw["rate"] != "."]
        raw["date"] = pd.to_datetime(raw["date"])
        raw["rate"] = raw["rate"].astype(float) / 100.0
        series = raw.set_index("date")["rate"]
        if start:
            series = series[series.index >= start]
        if end:
            series = series[series.index <= end]
        if series.empty:
            raise RuntimeError("FRED returned no observations in range")
        return series, False
    except Exception:
        idx = pd.DatetimeIndex([_FALLBACK_DATE])
        return pd.Series([_FALLBACK_RATE], index=idx, name="rate"), True


def current_risk_free_rate() -> tuple[float, bool]:
    """Convenience: latest available risk-free rate as a single decimal
    value (e.g. 0.037), plus whether it's the live or fallback value."""
    series, used_fallback = load_risk_free_rate_series()
    return float(series.iloc[-1]), used_fallback


if __name__ == "__main__":
    rate, used_fallback = current_risk_free_rate()
    print(f"USED_FALLBACK_RATE = {used_fallback}")
    print(f"Current risk-free rate: {rate:.4%}")
    if used_fallback:
        print(f"  (cached value as of {_FALLBACK_DATE} -- run with internet access for the live series)")
