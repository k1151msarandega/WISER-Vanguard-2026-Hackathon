"""
Asset universe configuration for the Vanguard Multi-Asset Portfolio Construction challenge.

This is the single source of truth for which instruments represent which asset
classes. Everything downstream (data loader, synthetic overlays, Markowitz
baseline, QUBO formulation) reads from here so there is exactly one place to
add/remove tickers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetMeta:
    ticker: str
    asset_class: str      # top-level bucket used for sector/exposure guardrails
    sub_class: str        # finer label, useful for explainability narratives
    liquidity_tier: int    # 1 = most liquid (broad index ETFs) .. 3 = least liquid


# NOTE: this list is deliberately ~18 tickers (not 50+). Vanguard's brief cares
# about formulation rigor and explainability, not universe size, and a smaller
# universe keeps the QUBO tractable for QAOA simulation in week 2.
UNIVERSE: list[AssetMeta] = [
    # --- Equities ---
    AssetMeta("SPY", "Equities", "US Large Cap", 1),
    AssetMeta("IWM", "Equities", "US Small Cap", 1),
    AssetMeta("EFA", "Equities", "Developed Intl", 1),
    AssetMeta("EEM", "Equities", "Emerging Markets", 2),

    # --- Fixed Income ---
    AssetMeta("IEF", "Fixed Income", "Treasuries (Mid)", 1),
    AssetMeta("TLT", "Fixed Income", "Treasuries (Long)", 1),
    AssetMeta("LQD", "Fixed Income", "Investment Grade Credit", 1),
    AssetMeta("HYG", "Fixed Income", "High Yield Credit", 2),

    # --- Commodities ---
    AssetMeta("GLD", "Commodities", "Gold", 1),
    AssetMeta("DBC", "Commodities", "Broad Commodities", 2),
    AssetMeta("USO", "Commodities", "Oil", 2),

    # --- Currencies ---
    AssetMeta("UUP", "Currencies", "USD Index", 2),
    AssetMeta("FXE", "Currencies", "EUR", 2),

    # --- Alternatives ---
    AssetMeta("VNQ", "Alternatives", "US REITs", 1),
    AssetMeta("VNQI", "Alternatives", "Intl REITs", 2),
]

TICKERS: list[str] = [a.ticker for a in UNIVERSE]
ASSET_CLASS_OF: dict[str, str] = {a.ticker: a.asset_class for a in UNIVERSE}
LIQUIDITY_TIER_OF: dict[str, int] = {a.ticker: a.liquidity_tier for a in UNIVERSE}

# Sector/asset-class exposure guardrails (max weight per asset class).
# These are the "guardrail breaches" metric Vanguard's rubric asks us to report.
ASSET_CLASS_CAPS: dict[str, float] = {
    "Equities": 0.60,
    "Fixed Income": 0.55,
    "Commodities": 0.20,
    "Currencies": 0.15,
    "Alternatives": 0.20,
}

# History window for price pulls.
START_DATE = "2010-01-01"
END_DATE = "2026-07-01"

# Bit-depth for binarized weight encoding (used in week 2 QUBO formulation,
# defined here since the data layer's precision should match it eventually).
WEIGHT_BITS = 4  # 2^4 = 16 discrete allocation levels per asset
