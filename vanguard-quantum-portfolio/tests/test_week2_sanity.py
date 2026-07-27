"""
Sanity checks for Week 2: partitioning, QUBO construction, and QAOA solving.
Same spirit as test_baseline_sanity.py -- catch embarrassing bugs now, before
Week 3's rigor pass builds on top of this.
"""

from __future__ import annotations

import numpy as np

from vqportfolio.market_data.loader import load_prices
from vqportfolio.market_data.overlays import compute_returns_and_risk, synthetic_cost_and_yield
from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
from vqportfolio.partitioning.partition import partition_assets, build_locked_allocation, PartitionConfig
from vqportfolio.quantum.qubo import build_o_set_qubo
from vqportfolio.quantum.qaoa_solver import solve_with_qaoa_and_validate
from vqportfolio.config import TICKERS, ASSET_CLASS_OF, ASSET_CLASS_CAPS


def run_week2_sanity_checks() -> None:
    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = synthetic_cost_and_yield(TICKERS, log_returns)
    mdd = per_asset_max_drawdown(prices)

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}")
    if used_synthetic:
        print("  -> reminder: run this locally with internet access before trusting "
              "any numbers you plan to report in the writeup.\n")

    dials = Dials()
    scores_df = compute_asset_scores(mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, dials)
    pconfig = PartitionConfig()
    partition = partition_assets(scores_df["score"], pconfig)

    # --- partition checks ---
    all_partitioned = set(partition["H"]) | set(partition["O"]) | set(partition["S"])
    assert all_partitioned == set(TICKERS), "partition doesn't cover the full universe"
    assert len(set(partition["H"]) & set(partition["O"])) == 0, "H and O overlap"
    assert len(set(partition["H"]) & set(partition["S"])) == 0, "H and S overlap"
    assert len(set(partition["O"]) & set(partition["S"])) == 0, "O and S overlap"
    assert len(partition["O"]) <= pconfig.max_o_size, "O-set exceeds max_o_size"
    print(f"Partition OK: |H|={len(partition['H'])} |O|={len(partition['O'])} |S|={len(partition['S'])}")

    h_weights, o_budget = build_locked_allocation(scores_df["score"], partition, pconfig)

    # --- H weight checks ---
    assert (h_weights <= pconfig.max_weight_per_asset + 1e-6).all(), "H violates per-asset cap"
    for asset_class, cap in ASSET_CLASS_CAPS.items():
        idx = [t for t in partition["H"] if ASSET_CLASS_OF[t] == asset_class]
        used = h_weights.loc[idx].sum() if idx else 0.0
        assert used <= cap + 1e-6, f"H violates {asset_class} cap"
    assert h_weights.sum() <= pconfig.h_budget_cap + 1e-6, "H exceeds its budget cap"
    print(f"H weights OK: sum={h_weights.sum():.4f}, O budget={o_budget:.4f}")

    class_headroom = {}
    for asset_class, cap in ASSET_CLASS_CAPS.items():
        idx = [t for t in partition["H"] if ASSET_CLASS_OF[t] == asset_class]
        used = h_weights.loc[idx].sum() if idx else 0.0
        class_headroom[asset_class] = max(cap - used, 0.0)

    # --- QUBO checks ---
    qubo_result = build_o_set_qubo(
        partition["O"], mu, sigma, overlay["cost_bps"], o_budget, class_headroom,
    )
    n_obj_vars = len(qubo_result.bit_index)
    assert n_obj_vars == len(partition["O"]) * qubo_result.bits, "objective variable count mismatch"
    assert qubo_result.qubo.get_num_binary_vars() >= n_obj_vars, "QUBO has fewer vars than expected"
    print(f"QUBO OK: {n_obj_vars} objective vars, "
          f"{qubo_result.qubo.get_num_binary_vars()} total (incl. slack)")

    # --- QAOA + exact + repair checks ---
    comparison = solve_with_qaoa_and_validate(qubo_result, o_budget, class_headroom)
    assert (comparison.qaoa_weights >= -1e-9).all(), "QAOA produced a negative weight"
    assert comparison.qaoa_weights.sum() <= o_budget + 1e-6, "QAOA (post-repair) exceeds O budget"
    for asset_class, headroom in class_headroom.items():
        idx = [t for t in comparison.qaoa_weights.index if ASSET_CLASS_OF[t] == asset_class]
        if idx:
            used = comparison.qaoa_weights.loc[idx].sum()
            assert used <= headroom + 1e-6, f"QAOA (post-repair) violates {asset_class} headroom"
    print(f"QAOA OK: matches exact={comparison.qaoa_matches_exact}, "
          f"repair_applied={comparison.repair_applied}, "
          f"objective gap={abs(comparison.qaoa_objective - comparison.exact_objective):.6f}")

    print("\nAll Week 2 sanity checks passed.")


if __name__ == "__main__":
    run_week2_sanity_checks()
