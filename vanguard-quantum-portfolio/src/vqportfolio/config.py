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
    liquidity_tier: int    # 1=most liquid .. 3=least. HAND-GUESSED, fallback only --
                            # real tiers are now computed from actual average
                            # daily dollar volume in market_data/liquidity.py;
                            # this static guess is used only when real OHLCV
                            # itself is unavailable (synthetic fallback case).


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
    AssetMeta("JNK", "Fixed Income", "High Yield Credit", 2),  # substituted for HYG: same asset class
                                                                # (high-yield corporate credit), HYG has
                                                                # no coverage in the HF dataset, JNK does

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
#
# Equities and Fixed Income caps are grounded in Vanguard's own published
# LifeStrategy fund family (fact sheets, June 30 2026, fetched directly from
# workplace.vanguard.com/.../fact-sheet/F0122.pdf, F0724.pdf, F0723.pdf, plus
# a search-confirmed snapshot of F0914) -- their four real balanced funds
# span the full range Vanguard itself is willing to hold:
#
#   Fund                              Ticker  Stock   Bond
#   LifeStrategy Income                VASIX   20%     80%
#   LifeStrategy Conservative Growth   VSCGX   ~41%    ~59%
#   LifeStrategy Moderate Growth       VSMGX   ~61%    ~39%   (current holdings)
#   LifeStrategy Growth                VASGX   80%     20%
#
# Our PREVIOUS caps (Equities 60%, Fixed Income 55%) were tighter than what
# Vanguard's own real, most conservative fund (80% bonds) or most aggressive
# fund (80% stocks) actually holds -- i.e. they'd have blocked us from ever
# replicating Vanguard's own real Income or Growth fund postures. Widened to
# 80% for both to match the real observed range.
#
# Commodities, Currencies, and Alternatives caps remain REASONED JUDGMENT
# CALLS, not data-grounded -- Vanguard's core LifeStrategy/balanced-fund
# lineup is stocks-and-bonds only and doesn't hold these asset classes at
# all, so there is no real Vanguard policy document to calibrate against.
# This is a genuine, searched-for-and-confirmed gap, not an oversight.
ASSET_CLASS_CAPS: dict[str, float] = {
    "Equities": 0.80,       # grounded: VASGX (LifeStrategy Growth) target
    "Fixed Income": 0.80,   # grounded: VASIX (LifeStrategy Income) target
    "Commodities": 0.20,    # judgment call -- no real Vanguard product to calibrate against
    "Currencies": 0.15,     # judgment call -- no real Vanguard product to calibrate against
    "Alternatives": 0.20,   # judgment call -- no real Vanguard product to calibrate against
}

# History window for price pulls.
START_DATE = "2010-01-01"
END_DATE = "2026-07-01"

# Bit-depth for binarized weight encoding (used in week 2 QUBO formulation,
# defined here since the data layer's precision should match it eventually).
WEIGHT_BITS = 4  # 2^4 = 16 discrete allocation levels per asset
