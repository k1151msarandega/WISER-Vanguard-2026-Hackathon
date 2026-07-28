"""
QAOA solver for the O-set QUBO: warm-started + multi-restart, validated
against exact brute-force enumeration.

Why warm-start rather than the XY-mixer/Dicke-state approach flagged in the
Week 2 literature review: that technique is built for *cardinality-selection*
problems (choose exactly K of N, one-hot structure) -- our O-set uses
*binarized continuous weights*, a genuinely different encoding that a
cardinality-constrained mixer doesn't map onto cleanly. The correct
state-of-the-art match for this formulation is warm-start QAOA (Egger et al.
2021): bias the initial quantum state toward a classical relaxation's
solution instead of uniform superposition.

Simplification, documented rather than hidden: Egger et al.'s full method
also replaces the mixer itself with a "tilted" per-qubit mixer whose ground
state is the biased state (so the bias survives across QAOA layers, not just
at t=0). We implement the biased *initial state* only, keeping the standard
X-mixer -- the tilted-mixer generator has sign/normalization details we
weren't confident recalling exactly from memory, and an incorrectly
implemented mixer would silently produce wrong physics, which is worse than
a documented partial version of a correct technique. At p=1 (current depth)
the initial-state bias still meaningfully steers the classical optimizer's
starting point even though the mixer will erode it as p grows -- worth
revisiting if we push to deeper circuits.

Reliability fix (the actual bug this replaces): single-seed random-angle
initialization was unstable -- an 8-seed sweep showed 5 seeds landing exactly
on the optimum and one landing catastrophically far off (see project log).
Multi-restart (several small perturbations around the warm-started point,
best-of-N by realized objective) directly addresses that. Spot-checked
across 3 seeds after this fix (0, 1, 99): all landed exactly on the optimum,
0/3 failures vs. the earlier 1/7 catastrophic-failure rate -- not a full
statistical claim (each full solve takes ~2 minutes at these settings, too
slow to sweep dozens of seeds interactively), but real evidence the fix
helps. A proper multi-seed variance analysis across many seeds belongs in
Week 3's rigor pass, run as a batch job rather than interactively.

Defaults (n_restarts=3, shots=1024, maxiter=60) are tuned for practical
runtime (~1-2 min per solve) rather than maximum robustness -- increase them
if runtime isn't a constraint for a particular run.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from qiskit import QuantumCircuit
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
    n_restarts_used: int


def _bitstring_to_x(bitstring: str, n: int) -> np.ndarray:
    """Qiskit's measurement bitstrings are little-endian: qubit 0 is the
    rightmost character. Convert to x[i] = bit value of qubit/variable i."""
    return np.array([int(bitstring[n - 1 - i]) for i in range(n)])


def _brute_force_exact(qubo, n: int) -> tuple[np.ndarray, float]:
    """Exact minimum by enumeration. Fine up to ~20-25 vars."""
    best_x, best_val = None, np.inf
    for bits in product([0, 1], repeat=n):
        x = np.array(bits)
        val = qubo.objective.evaluate(x)
        if val < best_val:
            best_val, best_x = val, x
    return best_x, best_val


def relax_and_warm_start(qubo, n: int) -> np.ndarray:
    """Continuous relaxation of the QUBO objective (box constraints [0,1]
    only -- the QUBO from QuadraticProgramToQubo has no remaining explicit
    constraints, they're already folded into the objective as penalty
    terms). Solved via local nonlinear optimization (scipy), not a convex
    solver: the objective's quadratic form isn't guaranteed PSD (it's a
    maximize-return-minus-risk objective negated for minimization, plus
    convex penalty terms -- the sum isn't necessarily convex), so a generic
    local optimizer is used rather than assuming convexity that may not
    hold. This only needs to produce a *reasonable directional bias* for
    warm-starting, not a certified global optimum.
    """
    lin = qubo.objective.linear.to_array()
    quad = qubo.objective.quadratic.to_array()
    q0 = qubo.objective.constant

    def f(x):
        return x @ quad @ x + lin @ x + q0

    def grad(x):
        return (quad + quad.T) @ x + lin

    x0 = np.full(n, 0.5)
    result = minimize(f, x0, jac=grad, bounds=[(0, 1)] * n, method="L-BFGS-B")
    # clip away from exact 0/1: Ry(0) or Ry(pi) collapses to a computational
    # basis state with zero superposition, killing QAOA's ability to explore
    return np.clip(result.x, 1e-3, 1 - 1e-3)


def _build_warm_start_circuit(hamiltonian, bias: np.ndarray, reps: int):
    n = len(bias)
    theta = 2 * np.arcsin(np.sqrt(bias))
    init = QuantumCircuit(n)
    for i, t in enumerate(theta):
        init.ry(t, i)

    ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=reps, initial_state=init)
    # Decompose into native gates before measuring/sampling -- left
    # undecomposed, the cost-evolution block stays an opaque instruction
    # that some simulator paths synthesize via a full matrix exponential of
    # the 2^n x 2^n operator instead of native gates (catastrophically slow
    # for anything beyond a handful of qubits; this cost ~2 min of debugging
    # via scipy sparse splu/spsolve calls the first time around).
    circuit = ansatz.decompose(reps=3)
    circuit.measure_all()
    return circuit


def _run_qaoa_warm_start(
    qubo,
    reps: int = 1,
    seed: int = 42,
    shots: int = 1024,
    n_restarts: int = 3,
    maxiter: int = 60,
) -> tuple[np.ndarray, float, int]:
    n = qubo.get_num_binary_vars()
    hamiltonian, offset = qubo.to_ising()

    bias = relax_and_warm_start(qubo, n)
    circuit = _build_warm_start_circuit(hamiltonian, bias, reps)
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
    best_x, best_val = None, np.inf

    # Multi-restart: several small perturbations around the warm-started
    # near-zero angle point (standard convention -- large initial angles
    # would immediately wash out the bias), best-of-N by realized objective
    # on the actual sampled bitstrings, not just the expectation.
    for restart in range(n_restarts):
        initial_point = rng.normal(loc=0.0, scale=0.3, size=2 * reps)
        opt_result = minimize(expected_cost, initial_point, method="COBYLA",
                               options={"maxiter": maxiter, "rhobeg": 0.3})

        bound = circuit.assign_parameters(opt_result.x)
        final = sampler.run([bound], shots=shots).result()
        counts = final[0].data.meas.get_counts()

        for bitstring, _count in counts.items():
            x = _bitstring_to_x(bitstring, n)
            val = qubo.objective.evaluate(x)
            if val < best_val:
                best_val, best_x = val, x

    return best_x, best_val, n_restarts


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
    n_restarts: int = 3,
) -> SolveComparison:
    qubo = qubo_result.qubo
    n = qubo.get_num_binary_vars()

    qaoa_x, qaoa_val, restarts_used = _run_qaoa_warm_start(
        qubo, reps=reps, seed=seed, n_restarts=n_restarts,
    )
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
        n_restarts_used=restarts_used,
    )


if __name__ == "__main__":
    from vqportfolio.market_data.loader import load_prices
    from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
    from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
    from vqportfolio.partitioning.partition import partition_assets, build_locked_allocation, PartitionConfig
    from vqportfolio.quantum.qubo import build_o_set_qubo
    from vqportfolio.config import TICKERS, ASSET_CLASS_CAPS, ASSET_CLASS_OF

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

    qubo_result = build_o_set_qubo(partition["O"], mu, sigma, overlay["cost_bps"], o_budget, class_headroom)
    comparison = solve_with_qaoa_and_validate(qubo_result, o_budget, class_headroom)

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}\n")
    print(f"QAOA objective (QUBO units, best of {comparison.n_restarts_used} restarts): "
          f"{comparison.qaoa_objective:.6f}")
    print(f"Exact objective (QUBO units): {comparison.exact_objective:.6f}")
    print(f"QAOA bitstring matches exact optimum: {comparison.qaoa_matches_exact}")
    print(f"Repair needed on QAOA solution: {comparison.repair_applied}\n")
    print("QAOA weights (post-repair):")
    print(comparison.qaoa_weights.round(4))
    print(f"  sum = {comparison.qaoa_weights.sum():.4f}  (budget = {o_budget:.4f})\n")
    print("Exact weights:")
    print(comparison.exact_weights.round(4))
    print(f"  sum = {comparison.exact_weights.sum():.4f}")
