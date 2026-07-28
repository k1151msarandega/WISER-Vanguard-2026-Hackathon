"""
Sanity checks for the Markowitz baseline. Not a full test suite -- just the
checks that would catch an embarrassing solver/formulation bug before it
propagates into the QUBO comparison in week 2:

  1. Higher risk aversion -> lower realized risk (monotonicity).
  2. Weights always sum to 1 and respect the per-asset cap.
  3. Guardrail breaches are always empty (solver enforces as hard constraints).
  4. Portfolio isn't degenerate (doesn't collapse to a single asset).
"""

from __future__ import annotations

from vqportfolio.market_data.loader import load_prices
from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
from vqportfolio.config import TICKERS
from vqportfolio.baseline.markowitz import solve_markowitz


def run_sanity_checks() -> None:
    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns)

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}")
    if used_synthetic:
        print("  -> reminder: run this locally with internet access before trusting "
              "any numbers you plan to report in the writeup.\n")

    risk_levels = [1.0, 3.0, 6.0, 12.0]
    realized_risks = []

    for ra in risk_levels:
        result = solve_markowitz(mu, sigma, overlay["cost_bps"], TICKERS, risk_aversion=ra)
        w = result["weights"]

        assert abs(w.sum() - 1.0) < 1e-4, f"weights don't sum to 1 at risk_aversion={ra}"
        assert (w <= 0.25 + 1e-4).all(), f"per-asset cap violated at risk_aversion={ra}"
        assert result["guardrail_breaches"] == {}, f"guardrail breach at risk_aversion={ra}"
        assert (w > 0.01).sum() >= 4, f"degenerate portfolio (too concentrated) at risk_aversion={ra}"

        realized_risks.append(result["risk_variance"])
        print(f"risk_aversion={ra:5.1f}  ->  return={result['expected_return']:.4f}  "
              f"risk={result['risk_variance']:.5f}  turnover={result['turnover']:.3f}  "
              f"n_active={int((w > 0.01).sum())}")

    # Monotonicity: risk should (weakly) decrease as risk_aversion increases.
    for i in range(len(realized_risks) - 1):
        assert realized_risks[i] >= realized_risks[i + 1] - 1e-6, (
            f"risk did not decrease monotonically between risk_aversion="
            f"{risk_levels[i]} and {risk_levels[i+1]}"
        )

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    run_sanity_checks()
