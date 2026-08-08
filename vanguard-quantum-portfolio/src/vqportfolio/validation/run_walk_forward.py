"""
Runner for walk-forward validation. Loads price/overlay data once, then
rolls the full H/O/S+QAOA pipeline + Markowitz baseline through
non-overlapping train/test windows (see validation/walk_forward.py for the
method and its documented scope limitations).

Usage: python -m vqportfolio.validation.run_walk_forward [train_days] [test_days] [max_o_size] [n_restarts] [maxiter] [shots]
"""

from __future__ import annotations

import sys

from vqportfolio.market_data.loader import load_prices
from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
from vqportfolio.partitioning.partition import PartitionConfig
from vqportfolio.config import TICKERS
from vqportfolio.validation.walk_forward import WalkForwardConfig, run_walk_forward, summarize, count_periods


def main(
    train_days: int = 756,
    test_days: int = 63,
    max_o_size: int = 4,
    n_restarts: int = 2,
    maxiter: int = 40,
    shots: int = 512,
    start_period: int = 0,
    max_periods: int | None = None,
) -> None:
    prices, used_synthetic = load_prices()
    # overlay computed once, held fixed across all periods -- see
    # walk_forward.py module docstring for why
    _, _, log_returns_full = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns_full)

    config = WalkForwardConfig(
        train_window_days=train_days,
        test_window_days=test_days,
        partition_config=PartitionConfig(max_o_size=max_o_size),
        n_restarts=n_restarts,
        maxiter=maxiter,
        shots=shots,
    )
    total_periods = count_periods(len(prices), config)

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}")
    print(f"Price history: {len(prices)} days x {len(prices.columns)} tickers")
    print(f"train_days={train_days} test_days={test_days} max_o_size={max_o_size} "
          f"n_restarts={n_restarts} maxiter={maxiter} shots={shots}")
    print(f"TOTAL_PERIODS_AVAILABLE={total_periods}  running start_period={start_period} "
          f"max_periods={max_periods}\n")

    results = run_walk_forward(prices, overlay, config, start_period=start_period, max_periods=max_periods)

    for r in results:
        print(
            f"PERIOD_RESULT idx={r.period_index} train=[{r.train_start_idx}:{r.train_end_idx}) "
            f"test=[{r.train_end_idx}:{r.test_end_idx}) O={r.o_set} "
            f"hos_return={r.hos_realized_return:.5f} mw_return={r.mw_realized_return:.5f} "
            f"matches_exact={r.qaoa_matches_exact} repair={r.repair_applied} "
            f"hos_breach={bool(r.hos_guardrail_breaches)} mw_breach={bool(r.mw_guardrail_breaches)} "
            f"ok={r.solve_ok} error={r.error}"
        )

    summary = summarize(results)
    print(f"\nWALK_FORWARD_SUMMARY {summary}")


if __name__ == "__main__":
    train_days = int(sys.argv[1]) if len(sys.argv) > 1 else 756
    test_days = int(sys.argv[2]) if len(sys.argv) > 2 else 63
    max_o_size = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_restarts = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    maxiter = int(sys.argv[5]) if len(sys.argv) > 5 else 40
    shots = int(sys.argv[6]) if len(sys.argv) > 6 else 512
    start_period = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    max_periods = int(sys.argv[8]) if len(sys.argv) > 8 else None
    main(train_days, test_days, max_o_size, n_restarts, maxiter, shots, start_period, max_periods)
