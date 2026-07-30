"""
Real Vanguard LifeStrategy fund family data, used to calibrate and validate
our own growth dial against Vanguard's actual published risk spectrum,
rather than an invented one.

Source: Vanguard fact sheets, fetched directly, June 30 2026 (fund numbers
0122, 0724, 0723; fund 0914's figures are from a June 30 2026 "investment
profile" search snapshot, current holdings rather than stated target, but
consistent with the fund's 60/40 name):
  - workplace.vanguard.com/assets/corp/fund_communications/pdf_publish/us-products/fact-sheet/F0122.pdf  (VASGX, Growth, 80/20)
  - workplace.vanguard.com/assets/corp/fund_communications/pdf_publish/us-products/fact-sheet/F0724.pdf  (VSCGX, Conservative Growth, 40/60)
  - workplace.vanguard.com/assets/corp/fund_communications/pdf_publish/us-products/fact-sheet/F0723.pdf  (VASIX, Income, 20/80)
  - fund 0914 (VSMGX, Moderate Growth, 60/40): current holdings per search snapshot

This is intentionally limited to stocks/bonds -- Vanguard's core balanced
fund lineup does not hold commodities, currencies, or alternatives, so this
data cannot (and should not be made to) calibrate those asset classes. That
is a real, confirmed gap, documented in config.py alongside the guardrail
caps it does and doesn't inform.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LifeStrategyFund:
    name: str
    ticker: str
    total_stock_pct: float  # % of fund, 0-100
    total_bond_pct: float
    us_stock_pct: float
    intl_stock_pct: float
    us_bond_pct: float
    intl_bond_pct: float
    is_stated_target: bool  # True = fund's own stated target; False = current holdings snapshot


LIFESTRATEGY_FAMILY: list[LifeStrategyFund] = [
    LifeStrategyFund("LifeStrategy Income", "VASIX", 20.0, 80.0, 12.0, 8.0, 56.0, 24.0, True),
    LifeStrategyFund("LifeStrategy Conservative Growth", "VSCGX", 41.0, 59.0, 25.0, 16.0, 41.5, 17.5, True),
    LifeStrategyFund("LifeStrategy Moderate Growth", "VSMGX", 61.3, 38.7, 36.9, 24.9, 27.0, 11.2, False),
    LifeStrategyFund("LifeStrategy Growth", "VASGX", 80.0, 20.0, 48.0, 32.0, 14.0, 6.0, True),
]

# Real, explicit Vanguard-published tolerance band (older VASIX fact sheet,
# 2007-era document structure -- directional evidence of real-world drift
# tolerance, not a current-fund-structure guarantee; the current 4-fund
# lineup doesn't carry a separate short-term-investments sleeve the way this
# older document describes):
#   "asset allocation ranges are expected to be 5%-30% stocks,
#    50%-75% bonds, and 20%-45% short-term investments" (VASIX, F0723 062007)
VASIX_HISTORICAL_TOLERANCE_BAND = {
    "stocks": (0.05, 0.30),
    "bonds": (0.50, 0.75),
    "short_term": (0.20, 0.45),
}


def nearest_lifestrategy_fund(equity_weight: float) -> LifeStrategyFund:
    """Given a portfolio's total equity weight (0-1), return the real
    Vanguard LifeStrategy fund it most closely resembles. Intended for the
    co-pilot demo: 'at this dial setting, your allocation resembles
    Vanguard's own {fund.name} Fund ({fund.ticker}).'"""
    equity_pct = equity_weight * 100
    return min(LIFESTRATEGY_FAMILY, key=lambda f: abs(f.total_stock_pct - equity_pct))


if __name__ == "__main__":
    print("Vanguard LifeStrategy family (real, published targets/holdings):\n")
    for f in LIFESTRATEGY_FAMILY:
        kind = "stated target" if f.is_stated_target else "current holdings"
        print(f"  {f.name} ({f.ticker}): {f.total_stock_pct:.1f}% stock / "
              f"{f.total_bond_pct:.1f}% bond  [{kind}]")

    print("\nNearest-fund lookup examples:")
    for eq in [0.15, 0.45, 0.65, 0.85]:
        fund = nearest_lifestrategy_fund(eq)
        print(f"  {eq:.0%} equities -> nearest: {fund.name} ({fund.ticker}, {fund.total_stock_pct:.1f}%)")
