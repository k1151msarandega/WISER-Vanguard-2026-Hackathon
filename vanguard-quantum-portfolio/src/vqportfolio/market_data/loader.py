"""
Price data loader.

Primary path: pull real daily adjusted-close prices via yfinance.
Fallback path: generate synthetic-but-realistic price series with the same
statistical shape (asset-class-appropriate drift/vol, cross-asset correlation
structure) so the rest of the pipeline can be developed and tested even when
yfinance isn't reachable (e.g. in a sandboxed environment).

The fallback is NOT meant to be the final data source for the actual
submission -- run this locally with internet access so USED_SYNTHETIC_PRICES
comes back False before you generate any results you plan to report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vqportfolio.config import TICKERS, ASSET_CLASS_OF, START_DATE, END_DATE

# Rough annualized (drift, vol) by asset class, used ONLY for synthetic
# fallback generation. These are order-of-magnitude realistic, not fitted
# to any specific historical period.
_CLASS_DRIFT_VOL = {
    "Equities": (0.08, 0.17),
    "Fixed Income": (0.03, 0.07),
    "Commodities": (0.04, 0.20),
    "Currencies": (0.01, 0.08),
    "Alternatives": (0.06, 0.18),
}


def _synthetic_prices(tickers: list[str], start: str, end: str, seed: int = 7) -> pd.DataFrame:
    """Generate correlated synthetic daily price series via correlated GBM.

    Correlation structure: assets within the same class are more correlated
    with each other than across classes, which mirrors real markets and is
    important -- a naive iid fallback would make the Markowitz baseline look
    trivially good (no real diversification tradeoff to solve).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)
    n_assets = len(tickers)

    classes = [ASSET_CLASS_OF[t] for t in tickers]
    unique_classes = sorted(set(classes))

    # Build a block correlation matrix: 0.55 within class, 0.15 across classes.
    corr = np.full((n_assets, n_assets), 0.15)
    for c in unique_classes:
        idx = [i for i, cl in enumerate(classes) if cl == c]
        for i in idx:
            for j in idx:
                corr[i, j] = 0.55
    np.fill_diagonal(corr, 1.0)

    # Nudge to nearest PSD matrix (cheap fix via eigenvalue clipping).
    eigval, eigvec = np.linalg.eigh(corr)
    eigval_clipped = np.clip(eigval, 1e-6, None)
    corr_psd = eigvec @ np.diag(eigval_clipped) @ eigvec.T
    d = np.sqrt(np.diag(corr_psd))
    corr_psd = corr_psd / np.outer(d, d)

    chol = np.linalg.cholesky(corr_psd)

    daily_drift = np.array([_CLASS_DRIFT_VOL[c][0] for c in classes]) / 252
    daily_vol = np.array([_CLASS_DRIFT_VOL[c][1] for c in classes]) / np.sqrt(252)

    z = rng.standard_normal((n_days, n_assets))
    z_corr = z @ chol.T
    log_returns = daily_drift + daily_vol * z_corr

    log_prices = np.cumsum(log_returns, axis=0)
    prices = 100 * np.exp(log_prices)  # start every series at 100

    return pd.DataFrame(prices, index=dates, columns=tickers)


def load_prices(
    tickers: list[str] | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
) -> tuple[pd.DataFrame, bool]:
    """Load daily adjusted-close prices.

    Returns (prices_df, used_synthetic_fallback).
    """
    tickers = tickers or TICKERS
    try:
        import yfinance as yf  # noqa: F401 -- optional dependency

        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            raise RuntimeError("yfinance returned no data")
        prices = raw["Close"] if "Close" in raw else raw
        prices = prices.dropna(how="all")
        if prices.empty:
            raise RuntimeError("yfinance returned an empty frame after cleaning")
        return prices, False
    except Exception:
        prices = _synthetic_prices(tickers, start, end)
        return prices, True


if __name__ == "__main__":
    prices, used_synthetic = load_prices()
    print(f"Loaded {prices.shape[0]} days x {prices.shape[1]} tickers")
    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}")
    print(prices.tail())
