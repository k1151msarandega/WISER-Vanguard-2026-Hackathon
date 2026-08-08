"""
Runner for the multi-seed QAOA variance sweep. Parametrized to run a subset
of seeds per call (each QAOA solve takes real time -- splitting across
calls avoids a single tool-call timeout swallowing a long sweep).

Usage: python -m vqportfolio.validation.run_multi_seed_sweep <seed_start> <seed_end> [n_restarts] [maxiter]
Prints one SEED_RESULT line per seed (parseable), plus a summary at the end
of the seeds actually run in that call. Aggregate across calls externally.
"""

from __future__ import annotations

import sys

from vqportfolio.market_data.loader import load_prices
from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
from vqportfolio.partitioning.partition import partition_assets, build_locked_allocation, PartitionConfig
from vqportfolio.quantum.qubo import build_o_set_qubo
from vqportfolio.validation.multi_seed_variance import run_seed, summarize
from vqportfolio.validation.scaling_ablation import vectorized_brute_force
from vqportfolio.validation.classical_benchmarks import weight_space_objective
from vqportfolio.config import TICKERS, ASSET_CLASS_CAPS, ASSET_CLASS_OF


def main(seed_start: int, seed_end: int, n_restarts: int = 2, maxiter: int = 40,
         max_o_size: int = 4, shots: int = 512) -> None:
    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns)
    mdd = per_asset_max_drawdown(prices)

    dials = Dials()
    scores_df = compute_asset_scores(mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, dials)
    pconfig = PartitionConfig(max_o_size=max_o_size)
    partition = partition_assets(scores_df["score"], pconfig)
    h_weights, o_budget = build_locked_allocation(mu, sigma, overlay["cost_bps"], partition, pconfig)

    class_headroom = {}
    for asset_class, cap in ASSET_CLASS_CAPS.items():
        idx = [t for t in partition["H"] if ASSET_CLASS_OF[t] == asset_class]
        used = h_weights.loc[idx].sum() if idx else 0.0
        class_headroom[asset_class] = max(cap - used, 0.0)

    qubo_result = build_o_set_qubo(partition["O"], mu, sigma, overlay["cost_bps"], o_budget, class_headroom)
    o_tickers = partition["O"]

    x_exact, _ = vectorized_brute_force(qubo_result.qubo)
    exact_weights = qubo_result.decode_weights(x_exact)
    val_exact = weight_space_objective(exact_weights, o_tickers, mu, sigma, overlay["cost_bps"])

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}")
    print(f"O-set: {o_tickers}, exact_objective={val_exact:.6f}\n")

    results = []
    for seed in range(seed_start, seed_end):
        r = run_seed(
            qubo_result, o_tickers, mu, sigma, overlay["cost_bps"], o_budget, class_headroom,
            seed=seed, n_restarts=n_restarts, maxiter=maxiter, shots=shots,
        )
        results.append(r)
        print(f"SEED_RESULT seed={r.seed} objective={r.objective:.6f} "
              f"gap={val_exact - r.objective:.6f} matches_exact={r.matches_exact} "
              f"repair_applied={r.repair_applied} ok={r.solve_ok} error={r.error}")

    summary = summarize(results, exact_objective=val_exact)
    print(f"\nBATCH_SUMMARY seeds={seed_start}-{seed_end - 1} {summary}")


if __name__ == "__main__":
    seed_start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seed_end = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    n_restarts = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    maxiter = int(sys.argv[4]) if len(sys.argv) > 4 else 40
    max_o_size = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    shots = int(sys.argv[6]) if len(sys.argv) > 6 else 512
    main(seed_start, seed_end, n_restarts, maxiter, max_o_size, shots)
