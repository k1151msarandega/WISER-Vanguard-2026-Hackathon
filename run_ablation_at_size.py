"""
Run the backend ablation at a single, caller-specified O-set size. Called
repeatedly (once per size, and once per depth p) by an external sweep so
each run stays within a single tool-call time budget -- exact brute-force
validation is exponential, so larger sizes take meaningfully longer and
shouldn't all be crammed into one call.

Findings from actually running this: docs/mps_scaling_findings.md.

Usage: python -m vqportfolio.validation.run_ablation_at_size <max_o_size> [n_restarts] [maxiter] [compute_exact: 0|1] [reps]

Examples actually used to produce the scaling curve in the findings doc:
  python -m vqportfolio.validation.run_ablation_at_size 3          # 9 qubits, defaults
  python -m vqportfolio.validation.run_ablation_at_size 5 2 60 0 2 # 15 qubits, p=2, skip exact recompute
  python -m vqportfolio.validation.run_ablation_at_size 7 2 60 0   # 28 qubits -- FAILS: AerSimulator's
                                                                    # default statevector target caps at
                                                                    # 27 qubits (CircuitTooWideForTarget),
                                                                    # a hard architectural limit, not slow
"""

from __future__ import annotations

import sys

from vqportfolio.market_data.loader import load_prices
from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
from vqportfolio.partitioning.partition import partition_assets, build_locked_allocation, PartitionConfig
from vqportfolio.quantum.qubo import build_o_set_qubo
from vqportfolio.config import TICKERS, ASSET_CLASS_CAPS, ASSET_CLASS_OF
from vqportfolio.validation.scaling_ablation import run_backend_ablation, results_to_dataframe


def run_at_size(max_o_size: int, n_restarts: int = 2, maxiter: int = 60,
                 bond_dimensions: list[int] | None = None, compute_exact: bool = True,
                 reps: int = 1) -> None:
    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns)
    mdd = per_asset_max_drawdown(prices)

    dials = Dials()
    scores_df = compute_asset_scores(mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, dials)

    pconfig = PartitionConfig(max_o_size=max_o_size, h_fraction=0.30, s_fraction=0.25)
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
    print(f"max_o_size={max_o_size}  reps(p)={reps}  O-set: {partition['O']} ({len(partition['O'])} assets)")
    print(f"Total qubits (objective + slack): {n}\n")

    results = run_backend_ablation(
        qubo_result.qubo,
        bond_dimensions=bond_dimensions if bond_dimensions is not None else [16, None],
        shots=256,
        n_restarts=n_restarts,
        maxiter=maxiter,
        compute_exact=compute_exact,
        reps=reps,
    )

    df = results_to_dataframe(results)
    print("\n=== Summary ===")
    print(f"SIZE_RESULT max_o_size={max_o_size} n_qubits={n}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    max_o_size = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_restarts = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    maxiter = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    compute_exact = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
    reps = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    run_at_size(max_o_size, n_restarts, maxiter, compute_exact=compute_exact, reps=reps)
