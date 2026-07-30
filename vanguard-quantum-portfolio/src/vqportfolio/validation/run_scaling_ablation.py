"""
Runner for the partitioning-vs-MPS scaling ablation. Builds a deliberately
larger O-set (bypassing the usual max_o_size=4 prototyping cap) from the
real pipeline's covariance-driven QUBO, then compares statevector vs. MPS
(at several bond-dimension caps) on both runtime and solution quality
against the true optimum.
"""

from __future__ import annotations

from vqportfolio.market_data.loader import load_prices
from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
from vqportfolio.partitioning.partition import partition_assets, build_locked_allocation, PartitionConfig
from vqportfolio.quantum.qubo import build_o_set_qubo
from vqportfolio.config import TICKERS, ASSET_CLASS_CAPS, ASSET_CLASS_OF
from vqportfolio.validation.scaling_ablation import run_backend_ablation, results_to_dataframe


def main():
    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns)
    mdd = per_asset_max_drawdown(prices)

    dials = Dials()
    scores_df = compute_asset_scores(mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, dials)

    # Deliberately larger O than the Week 2 prototype (max_o_size=4) to
    # actually stress-test scaling behavior -- 7 assets x 3 bits = 21
    # objective qubits, comfortably in range for the vectorized exact
    # brute-force validator (~2^22-24 tractable) while being large enough
    # to show real MPS behavior differences.
    pconfig = PartitionConfig(max_o_size=5, h_fraction=0.30, s_fraction=0.25)
    partition = partition_assets(scores_df["score"], pconfig)
    h_weights, o_budget = build_locked_allocation(mu, sigma, overlay["cost_bps"], partition, pconfig)

    class_headroom = {}
    for asset_class, cap in ASSET_CLASS_CAPS.items():
        idx = [t for t in partition["H"] if ASSET_CLASS_OF[t] == asset_class]
        used = h_weights.loc[idx].sum() if idx else 0.0
        class_headroom[asset_class] = max(cap - used, 0.0)

    qubo_result = build_o_set_qubo(
        partition["O"], mu, sigma, overlay["cost_bps"], o_budget, class_headroom,
    )
    n = qubo_result.qubo.get_num_binary_vars()

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}")
    print(f"O-set: {partition['O']} ({len(partition['O'])} assets)")
    print(f"Total qubits (objective + slack): {n}\n")

    results = run_backend_ablation(
        qubo_result.qubo,
        bond_dimensions=[8, 16, 32, None],
        shots=512,
    )

    df = results_to_dataframe(results)
    print("\n=== Summary ===")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    main()
