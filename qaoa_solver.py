"""
QAOA solver for the O-set QUBO, with exact validation.

Implemented directly against qiskit's QAOAAnsatz + StatevectorSampler (V2
primitives) rather than qiskit_algorithms' QAOA/MinimumEigenOptimizer: that
package (0.4.0) predates qiskit 2.x's primitive API changes and is broken
against the qiskit/qiskit-aer versions available here. This is a deliberate,
documented substitution, not a shortcut -- the manual loop below does exactly
what qiskit_algorithms' QAOA does internally (bind parameters, sample,
compute expected cost, classically optimize), just without the incompatible
dependency, and it's arguably more transparent about what's happening.

Week 2 goal per the project plan: "QAOA produces a feasible (post-repair)
portfolio on a toy instance, compared once against Markowitz." Since this
toy instance is small enough (13 qubits here) to brute-force exactly, we
validate QAOA against the *true* optimum for this QUBO -- a stronger check
than the plan asked for, worth doing while the problem is still this small.
Brute force stops scaling around ~20-25 qubits; Week 3's scaling analysis
picks up from there.

Repair step: QAOA sampling is not guaranteed to land on a feasible
bitstring (constraint penalties are soft). We take the best sampled
bitstring, decode it to weights, and if budget/class caps are violated,
scale down proportionally rather than discard the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler

from vqportfolio.config import ASSET_CLASS_OF
from vqportfolio.quantum.qubo import QuboBuildResult


@dataclass
class SolveComparison:
    qaoa_weights: pd.Series
    qaoa_objective: float
    exact_weights: pd.Series
    exact_objective: float
    qaoa_matches_exact: bool
    repair_applied: bool


def _bitstring_to_x(bitstring: str, n: int) -> np.ndarray:
    """Qiskit's measurement bitstrings are little-endian: qubit 0 is the
    rightmost character. Convert to x[i] = bit value of qubit/variable i."""
    return np.array([int(bitstring[n - 1 - i]) for i in range(n)])


def _brute_force_exact(qubo, n: int) -> tuple[np.ndarray, float]:
    """Exact minimum by enumeration. Fine up to ~20-25 vars; this toy
    instance (13) is comfortably inside that range."""
    best_x, best_val = None, np.inf
    for bits in product([0, 1], repeat=n):
        x = np.array(bits)
        val = qubo.objective.evaluate(x)
        if val < best_val:
            best_val, best_x = val, x
    return best_x, best_val


def _run_qaoa(qubo, reps: int = 1, seed: int = 42, shots: int = 4096) -> tuple[np.ndarray, float]:
    n = qubo.get_num_binary_vars()
    hamiltonian, offset = qubo.to_ising()

    ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=reps)
    # Decompose into native gates before measuring/sampling. Left undecomposed,
    # the cost-evolution block stays an opaque instruction that some simulator
    # paths synthesize via a full matrix exponential of the 2^n x 2^n operator
    # instead of native two-qubit gates -- correct, but catastrophically slow
    # (that's what the scipy sparse splu/spsolve calls turned out to be).
    circuit = ansatz.decompose(reps=3)
    circuit.measure_all()

    sampler = StatevectorSampler(seed=seed)

    def expected_cost(params: np.ndarray) -> float:
        bound = circuit.assign_parameters(params)
        result = sampler.run([bound], shots=shots).result()
        counts = result[0].data.meas.get_counts()
        total = sum(counts.values())
        exp = 0.0
        for bitstring, count in counts.items():
            x = _bitstring_to_x(bitstring, n)
            exp += (count / total) * qubo.objective.evaluate(x)
        return exp

    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(0, np.pi / 2, size=2 * reps)
    opt_result = minimize(expected_cost, initial_point, method="COBYLA",
                           options={"maxiter": 100, "rhobeg": 0.5})

    # Final sample: take the best individual bitstring observed at the
    # optimized parameters, not just the expectation -- standard practice,
    # since we care about the best solution found, not the average.
    bound = circuit.assign_parameters(opt_result.x)
    final = sampler.run([bound], shots=shots).result()
    counts = final[0].data.meas.get_counts()

    best_x, best_val = None, np.inf
    for bitstring, _count in counts.items():
        x = _bitstring_to_x(bitstring, n)
        val = qubo.objective.evaluate(x)
        if val < best_val:
            best_val, best_x = val, x

    return best_x, best_val


def repair_weights(
    weights: pd.Series,
    o_budget: float,
    max_weight_per_asset: float,
    class_headroom: dict[str, float],
) -> tuple[pd.Series, bool]:
    """Scale down proportionally if budget or any class cap is violated.
    Returns (repaired weights, whether repair was needed)."""
    repaired = weights.clip(upper=max_weight_per_asset).copy()
    changed = not np.allclose(repaired.values, weights.values, atol=1e-9)

    if repaired.sum() > o_budget + 1e-9:
        repaired = repaired * (o_budget / repaired.sum())
        changed = True

    for asset_class, headroom in class_headroom.items():
        idx = [t for t in repaired.index if ASSET_CLASS_OF[t] == asset_class]
        if not idx:
            continue
        class_total = repaired.loc[idx].sum()
        if class_total > headroom + 1e-9 and class_total > 0:
            repaired.loc[idx] = repaired.loc[idx] * (headroom / class_total)
            changed = True

    return repaired, changed


def solve_with_qaoa_and_validate(
    qubo_result: QuboBuildResult,
    o_budget: float,
    class_headroom: dict[str, float],
    reps: int = 1,
    seed: int = 42,
) -> SolveComparison:
    qubo = qubo_result.qubo
    n = qubo.get_num_binary_vars()

    qaoa_x, qaoa_val = _run_qaoa(qubo, reps=reps, seed=seed)
    exact_x, exact_val = _brute_force_exact(qubo, n)

    qaoa_weights = qubo_result.decode_weights(qaoa_x)
    exact_weights = qubo_result.decode_weights(exact_x)

    qaoa_weights_repaired, repair_applied = repair_weights(
        qaoa_weights, o_budget, qubo_result.max_weight_per_asset, class_headroom,
    )

    matches = bool(np.array_equal(qaoa_x, exact_x))

    return SolveComparison(
        qaoa_weights=qaoa_weights_repaired,
        qaoa_objective=float(qaoa_val),
        exact_weights=exact_weights,
        exact_objective=float(exact_val),
        qaoa_matches_exact=matches,
        repair_applied=repair_applied,
    )


if __name__ == "__main__":
    from vqportfolio.market_data.loader import load_prices
    from vqportfolio.market_data.overlays import compute_returns_and_risk, synthetic_cost_and_yield
    from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
    from vqportfolio.partitioning.partition import partition_assets, build_locked_allocation, PartitionConfig
    from vqportfolio.quantum.qubo import build_o_set_qubo
    from vqportfolio.config import TICKERS, ASSET_CLASS_CAPS, ASSET_CLASS_OF

    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = synthetic_cost_and_yield(TICKERS, log_returns)
    mdd = per_asset_max_drawdown(prices)

    dials = Dials()
    scores_df = compute_asset_scores(mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, dials)
    pconfig = PartitionConfig()
    partition = partition_assets(scores_df["score"], pconfig)
    h_weights, o_budget = build_locked_allocation(scores_df["score"], partition, pconfig)

    class_headroom = {}
    for asset_class, cap in ASSET_CLASS_CAPS.items():
        idx = [t for t in partition["H"] if ASSET_CLASS_OF[t] == asset_class]
        used = h_weights.loc[idx].sum() if idx else 0.0
        class_headroom[asset_class] = max(cap - used, 0.0)

    qubo_result = build_o_set_qubo(partition["O"], mu, sigma, overlay["cost_bps"], o_budget, class_headroom)
    comparison = solve_with_qaoa_and_validate(qubo_result, o_budget, class_headroom)

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}\n")
    print(f"QAOA objective (QUBO units): {comparison.qaoa_objective:.6f}")
    print(f"Exact objective (QUBO units): {comparison.exact_objective:.6f}")
    print(f"QAOA bitstring matches exact optimum: {comparison.qaoa_matches_exact}")
    print(f"Repair needed on QAOA solution: {comparison.repair_applied}\n")
    print("QAOA weights (post-repair):")
    print(comparison.qaoa_weights.round(4))
    print(f"  sum = {comparison.qaoa_weights.sum():.4f}  (budget = {o_budget:.4f})\n")
    print("Exact weights:")
    print(comparison.exact_weights.round(4))
    print(f"  sum = {comparison.exact_weights.sum():.4f}")
