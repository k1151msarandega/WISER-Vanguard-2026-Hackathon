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

load_ohlc() additionally sources from the defeatbeta/yahoo-finance-data
dataset on Hugging Face (queried via DuckDB with predicate pushdown -- no
full-file download) for the subset of tickers it actually covers, since it
gives longer, real history than yfinance for those specific tickers. Schema
and data quality were verified by hand before writing this (DESCRIBE query
confirmed columns; a >15%-single-day-move sweep + a direct look at USO's
April 2020 reverse-split date confirmed `close` is split-adjusted, not raw).
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

# Confirmed via direct DuckDB query against the real dataset (not assumed).
HF_PARQUET_URL = "https://huggingface.co/datasets/defeatbeta/yahoo-finance-data/resolve/main/data/stock_prices.parquet"
HF_COVERED_TICKERS = {"SPY", "IWM", "GLD", "DBC", "USO", "UUP", "FXE", "TLT", "JNK"}


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


def _synthetic_ohlc(tickers: list[str], start: str, end: str, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV: Close from the same correlated-GBM model as before,
    High/Low simulated around it via a vol-scaled intraday range, Volume
    drawn from a per-ticker lognormal base rate. Used only as a fallback
    when real OHLCV isn't reachable -- and note Volume here is NOT used to
    derive liquidity tiers (that would be circular: synthetic volume
    encodes no real liquidity information). See
    market_data.liquidity.compute_liquidity_tiers for how the fallback
    there is handled honestly.
    """
    close = _synthetic_prices(tickers, start, end, seed)
    rng = np.random.default_rng(seed + 1)
    daily_vol = close.pct_change().std()  # per-ticker realized daily vol

    # intraday range roughly proportional to daily vol, positive by construction
    range_frac = pd.DataFrame(
        np.abs(rng.normal(loc=0.6, scale=0.2, size=close.shape)) * daily_vol.values,
        index=close.index, columns=close.columns,
    ).clip(lower=0.001)

    high = close * (1 + range_frac / 2)
    low = close * (1 - range_frac / 2)

    # arbitrary per-ticker base volume level, no real liquidity signal
    base_volume = rng.lognormal(mean=15, sigma=1.5, size=len(tickers))
    volume = pd.DataFrame(
        rng.lognormal(mean=0, sigma=0.3, size=close.shape) * base_volume,
        index=close.index, columns=close.columns,
    ).round()

    return pd.concat({"Close": close, "High": high, "Low": low, "Volume": volume}, axis=1)


def _load_from_hf(tickers: list[str], start: str, end: str) -> pd.DataFrame | None:
    """Query the real HF/DuckDB dataset for whichever of `tickers` it
    covers. Returns None (not a partial frame) if duckdb isn't installed or
    the query fails outright -- callers fall back to yfinance/synthetic for
    everything in that case, rather than silently mixing an empty result in."""
    candidates = [t for t in tickers if t in HF_COVERED_TICKERS]
    if not candidates:
        return None
    try:
        import duckdb

        tickers_sql = ",".join(f"'{t}'" for t in candidates)
        query = f"""
            SELECT symbol, report_date, open, close, high, low, volume
            FROM '{HF_PARQUET_URL}'
            WHERE symbol IN ({tickers_sql})
              AND report_date >= '{start}' AND report_date <= '{end}'
            ORDER BY report_date
        """
        df = duckdb.sql(query).df()
        if df.empty:
            return None
        df["report_date"] = pd.to_datetime(df["report_date"])
        df = df.set_index(["report_date", "symbol"])[["open", "close", "high", "low", "volume"]]
        df.columns = ["Open", "Close", "High", "Low", "Volume"]
        pivoted = df.unstack("symbol")
        pivoted.index.name = None
        return pivoted
    except Exception:
        return None


def _load_ohlc_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame | None:
    """yfinance path for a given ticker subset. Returns None (not raising)
    on any failure -- callers treat that as "fall through to synthetic"."""
    if not tickers:
        return None
    try:
        import yfinance as yf

        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            return None
        needed = ["High", "Low", "Close", "Volume"]
        if not all(f in raw for f in needed):
            return None
        ohlc = raw[needed].dropna(how="all")
        return ohlc if not ohlc.empty else None
    except Exception:
        return None


def load_ohlc(
    tickers: list[str] | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
) -> tuple[pd.DataFrame, bool]:
    """Load daily High/Low/Close/Volume, hybrid-sourced: HF/DuckDB for
    tickers it covers (real, longer history), yfinance for the rest,
    synthetic as a last resort for anything neither source provides.

    Returns (ohlcv_df, any_ticker_used_synthetic) -- kept as a single bool
    for backward compatibility with every existing caller. For the full
    per-ticker breakdown (which source each ticker actually came from), use
    load_ohlc_with_sources() instead.
    """
    ohlcv, sources = load_ohlc_with_sources(tickers, start, end)
    any_synthetic = any(s == "synthetic" for s in sources.values())
    return ohlcv, any_synthetic


def load_ohlc_with_sources(
    tickers: list[str] | None = None,
    start: str = START_DATE,
    end: str = END_DATE,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Same as load_ohlc() but returns the full per-ticker source map
    ('hf' | 'yfinance' | 'synthetic') instead of a single collapsed bool."""
    tickers = tickers or TICKERS
    sources: dict[str, str] = {}
    pieces: list[pd.DataFrame] = []

    hf_df = _load_from_hf(tickers, start, end)
    hf_tickers_got = []
    if hf_df is not None:
        hf_tickers_got = [t for t in tickers if ("Close", t) in hf_df.columns]
        if hf_tickers_got:
            pieces.append(hf_df[[c for c in hf_df.columns if c[1] in hf_tickers_got]])
            for t in hf_tickers_got:
                sources[t] = "hf"

    remaining = [t for t in tickers if t not in sources]
    yf_df = _load_ohlc_yfinance(remaining, start, end) if remaining else None
    yf_tickers_got = []
    if yf_df is not None:
        yf_tickers_got = [t for t in remaining if ("Close", t) in yf_df.columns]
        if yf_tickers_got:
            pieces.append(yf_df)
            for t in yf_tickers_got:
                sources[t] = "yfinance"

    still_missing = [t for t in tickers if t not in sources]
    if still_missing:
        synth = _synthetic_ohlc(still_missing, start, end)
        pieces.append(synth)
        for t in still_missing:
            sources[t] = "synthetic"

    combined = pd.concat(pieces, axis=1).sort_index(axis=1)
    return combined, sources


if __name__ == "__main__":
    prices, used_synthetic = load_prices()
    print(f"Loaded {prices.shape[0]} days x {prices.shape[1]} tickers")
    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}")
    print(prices.tail())

    ohlc, sources = load_ohlc_with_sources()
    print(f"\nSource breakdown: { {s: list(sources.values()).count(s) for s in set(sources.values())} }")
    for t in sorted(sources, key=lambda t: sources[t]):
        print(f"  {t}: {sources[t]}")
    print()
    print(ohlc["High"].tail(2))
    print(ohlc["Low"].tail(2))
    print(ohlc["Volume"].tail(2))
