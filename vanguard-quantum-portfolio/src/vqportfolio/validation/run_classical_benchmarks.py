"""
Full equal-footing comparison: exact brute force, QAOA, ILP (diagonal risk),
greedy, and random -- all solving the identical discretized O-set problem.

Usage: python -m vqportfolio.validation.run_classical_benchmarks
"""

from __future__ import annotations

from vqportfolio.market_data.loader import load_prices
from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
from vqportfolio.partitioning.partition import partition_assets, build_locked_allocation, PartitionConfig
from vqportfolio.quantum.qubo import build_o_set_qubo
from vqportfolio.quantum.qaoa_solver import solve_with_qaoa_and_validate
from vqportfolio.validation.classical_benchmarks import (
    solve_ilp_diagonal_risk, solve_greedy, solve_random, weight_space_objective, BenchmarkResult,
)
from vqportfolio.validation.scaling_ablation import vectorized_brute_force
from vqportfolio.config import TICKERS, ASSET_CLASS_CAPS, ASSET_CLASS_OF


def run_full_comparison(n_random_samples: int = 500, seed: int = 7, risk_aversion: float = 3.0) -> list[BenchmarkResult]:
    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns)
    mdd = per_asset_max_drawdown(prices)

    dials = Dials()
    scores_df = compute_asset_scores(mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, dials)
    pconfig = PartitionConfig()
    partition = partition_assets(scores_df["score"], pconfig)
    h_weights, o_budget = build_locked_allocation(mu, sigma, overlay["cost_bps"], partition, pconfig)

    class_headroom = {}
    for asset_class, cap in ASSET_CLASS_CAPS.items():
        idx = [t for t in partition["H"] if ASSET_CLASS_OF[t] == asset_class]
        used = h_weights.loc[idx].sum() if idx else 0.0
        class_headroom[asset_class] = max(cap - used, 0.0)

    qubo_result = build_o_set_qubo(
        partition["O"], mu, sigma, overlay["cost_bps"], o_budget, class_headroom, risk_aversion=risk_aversion,
    )
    o_tickers = partition["O"]

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}")
    print(f"O-set: {o_tickers}, budget={o_budget:.4f}\n")

    results = []

    # Every method's reported objective is recomputed from its decoded WEIGHTS via the same
    # weight_space_objective() -- not from bits/QUBO internals -- so exact, QAOA (whose repaired
    # weights aren't bit-representable), and the three classical methods are all on the exact
    # same footing, not just "close enough" conventions.
    x_exact, _ = vectorized_brute_force(qubo_result.qubo)
    exact_weights = qubo_result.decode_weights(x_exact)
    val_exact = weight_space_objective(exact_weights, o_tickers, mu, sigma, overlay["cost_bps"], risk_aversion)
    results.append(BenchmarkResult("Exact (brute force)", exact_weights, val_exact, 0.0, True))

    qaoa_comp = solve_with_qaoa_and_validate(qubo_result, o_budget, class_headroom)
    qaoa_obj = weight_space_objective(qaoa_comp.qaoa_weights, o_tickers, mu, sigma, overlay["cost_bps"], risk_aversion)
    results.append(BenchmarkResult(
        "QAOA (warm-start + multi-restart)", qaoa_comp.qaoa_weights,
        qaoa_obj, 0.0, True,
        extra={"matches_exact": qaoa_comp.qaoa_matches_exact, "repair_applied": qaoa_comp.repair_applied},
    ))

    results.append(solve_ilp_diagonal_risk(
        qubo_result, o_tickers, mu, sigma, overlay["cost_bps"], o_budget, class_headroom,
        risk_aversion=risk_aversion,
    ))
    results.append(solve_greedy(
        qubo_result, o_tickers, mu, sigma, overlay["cost_bps"], o_budget, class_headroom,
        risk_aversion=risk_aversion,
    ))
    results.append(solve_random(
        qubo_result, o_tickers, mu, sigma, overlay["cost_bps"], o_budget, class_headroom,
        risk_aversion=risk_aversion, n_samples=n_random_samples, seed=seed,
    ))

    print(f"{'Method':<35} {'Objective':>12} {'Gap to exact':>14} {'Time (s)':>10}")
    for r in results:
        gap = r.objective - val_exact
        print(f"{r.method:<35} {r.objective:>12.6f} {gap:>14.6f} {r.solve_time_s:>10.4f}")

    return results


if __name__ == "__main__":
    run_full_comparison()
