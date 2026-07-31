"""
Partitioning-vs-MPS scaling ablation.

Tests two independent scaling levers -- H/O/S partitioning (shrinks the
problem before it reaches the quantum solver) and MPS simulation (lets the
quantum solver handle more qubits than statevector) -- separately, not just
combined, so we can report which one is actually doing the work.

Full write-up with tables: docs/mps_scaling_findings.md. Summary:
  - MPS's benefit depends on entanglement STRUCTURE, not just qubit count --
    an early adversarial random-dense-Hamiltonian test choked badly even at
    ~20 qubits, but our actual covariance-structured QUBO scaled cleanly
    through 21 qubits.
  - MPS's benefit is NOT free: uncapped MPS runtime scales sharply with QAOA
    circuit depth p (0.31s -> 3.49s -> 7.87s for p=1,2,3 at fixed qubit
    count), independent of qubit count. Since QAOA needs depth for solution
    quality, MPS's qubit-count advantage and QAOA's depth requirement work
    against each other -- not a simple "MPS helps" story.
  - A hard architectural ceiling exists at 27 qubits for the current
    statevector-based angle-optimization step (AerSimulator's default
    target), independent of the MPS question -- found empirically, not
    previously documented.
  - All of the above used SYNTHETIC price data (no internet access in the
    dev sandbox) -- needs re-confirmation with real data before being
    treated as a final result.

Fairness note: QAOA parameters are optimized ONCE via fast, exact statevector
simulation, then the *same* fixed circuit is sampled through each backend for
comparison. This isolates "does the backend affect final sampling fidelity"
from "does the whole optimization pipeline differ" -- optimizing separately
per backend would let sampling-noise differences masquerade as fidelity
differences.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator

from vqportfolio.quantum.qaoa_solver import relax_and_warm_start, _bitstring_to_x


def vectorized_brute_force(qubo, chunk_size: int = 2_000_000) -> tuple[np.ndarray, float]:
    """Exact minimum via chunked, vectorized enumeration. Extends practical
    exact validation from ~2^13-16 (the old itertools.product loop) to
    ~2^22-24 by avoiding per-combination Python function-call overhead --
    still exponential, still won't reach real problem sizes, but pushes the
    point where we're forced to trust approximate methods further out."""
    n = qubo.get_num_binary_vars()
    lin = qubo.objective.linear.to_array()
    quad = qubo.objective.quadratic.to_array()
    q0 = qubo.objective.constant

    N = 1 << n
    best_val, best_x = np.inf, None
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        idx = np.arange(start, end)
        bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        vals = np.einsum('ij,jk,ik->i', bits, quad, bits) + bits @ lin + q0
        chunk_best = np.argmin(vals)
        if vals[chunk_best] < best_val:
            best_val, best_x = vals[chunk_best], bits[chunk_best]
    return best_x.astype(int), float(best_val)


def _optimize_qaoa_params(qubo, reps: int = 1, seed: int = 42,
                           shots: int = 1024, maxiter: int = 60,
                           n_restarts: int = 2) -> tuple[QuantumCircuit, np.ndarray]:
    """Optimize QAOA angles once via fast exact (statevector) simulation.
    Returns the bound circuit template (unbound parameters) and the best
    angle vector found -- callers bind and sample through whichever backend
    they're comparing."""
    n = qubo.get_num_binary_vars()
    hamiltonian, offset = qubo.to_ising()

    bias = relax_and_warm_start(qubo, n)
    theta = 2 * np.arcsin(np.sqrt(bias))
    init = QuantumCircuit(n)
    for i, t in enumerate(theta):
        init.ry(t, i)

    ansatz = QAOAAnsatz(cost_operator=hamiltonian, reps=reps, initial_state=init)
    circuit = ansatz.decompose(reps=3)
    circuit.measure_all()

    sim = AerSimulator(method="statevector")

    def expected_cost(params):
        bound = circuit.assign_parameters(params)
        result = sim.run(transpile(bound, sim), shots=shots, seed_simulator=seed).result()
        counts = result.get_counts()
        total = sum(counts.values())
        return sum((c / total) * qubo.objective.evaluate(_bitstring_to_x(bs.replace(" ", ""), n))
                   for bs, c in counts.items())

    rng = np.random.default_rng(seed)
    best_params, best_cost = None, np.inf
    for _ in range(n_restarts):
        x0 = rng.normal(0, 0.3, size=2 * reps)
        res = minimize(expected_cost, x0, method="COBYLA", options={"maxiter": maxiter})
        if res.fun < best_cost:
            best_cost, best_params = res.fun, res.x

    return circuit, best_params


@dataclass
class BackendResult:
    backend: str
    max_bond_dimension: int | None
    n_qubits: int
    runtime_s: float
    objective: float
    gap_to_exact: float | None


def run_backend_ablation(
    qubo,
    reps: int = 1,
    seed: int = 42,
    shots: int = 1024,
    bond_dimensions: list[int] | None = None,
    compute_exact: bool = True,
    n_restarts: int = 2,
    maxiter: int = 60,
) -> list[BackendResult]:
    """Sample the same optimized QAOA circuit through statevector and MPS
    (at several bond-dimension caps), comparing solution quality via
    Hamiltonian expectation value <psi|H|psi> rather than sampled-bitstring
    decoding.

    Why expectation value and not "best sampled bitstring": an earlier
    version of this function compared backends by decoding get_counts()
    bitstrings and evaluating the QUBO objective on each. That produced a
    nonsensical result -- capped MPS (bond<=8) beating *uncapped* MPS, which
    is impossible in principle (uncapped MPS is mathematically exact, i.e.
    equivalent to statevector, so it cannot be worse than a truncated
    version of itself). Root-caused via a small 6-qubit check: exact
    statevectors from both methods have fidelity 1.0 (the physics is
    correct), but their get_counts() bitstring labels don't agree with each
    other -- a bit-ordering mismatch specific to comparing sampled output
    *across* backend methods, not a truncation or physics effect. Using
    save_expectation_value() instead sidesteps bitstring decoding entirely
    (verified to agree exactly between backends on the same 6-qubit check)
    and is also the more standard metric for QAOA solution quality anyway.

    Framing note: `gap_to_exact` here is (expectation value) - (true minimum
    objective), i.e. it reflects how much of the QAOA state's probability
    mass sits on suboptimal outcomes, not "was the optimal bitstring ever
    sampled." That's a different (but standard, and arguably more
    informative) quality metric than "best observed sample" -- it's exactly
    the quantity QAOA's own classical optimizer was minimizing during
    training, so evaluating quality the same way is consistent, not a
    downgrade.

    Note this does NOT affect the production solver in qaoa_solver.py, which
    always samples through one backend consistently and never compares
    bitstrings across backends -- the bug only manifests in cross-backend
    comparison, which is what this ablation specifically does.
    """
    n = qubo.get_num_binary_vars()
    bond_dimensions = bond_dimensions if bond_dimensions is not None else [8, 16, 32, None]

    exact_val = None
    if compute_exact:
        t0 = time.time()
        _, exact_val = vectorized_brute_force(qubo)
        print(f"  exact (vectorized brute force): {time.time() - t0:.2f}s, obj={exact_val:.6f}")

    hamiltonian, offset = qubo.to_ising()
    circuit, params = _optimize_qaoa_params(qubo, reps=reps, seed=seed, shots=shots,
                                             maxiter=maxiter, n_restarts=n_restarts)
    bound = circuit.assign_parameters(params)

    # replace the measure_all() at the end with save_expectation_value --
    # need the pre-measurement circuit, so rebuild without the measurement
    bound_no_meas = bound.remove_final_measurements(inplace=False)
    bound_no_meas.save_expectation_value(hamiltonian, range(n))

    results = []

    sv_sim = AerSimulator(method="statevector")
    t0 = time.time()
    sv_exp = sv_sim.run(transpile(bound_no_meas, sv_sim)).result().data(0)["expectation_value"]
    sv_elapsed = time.time() - t0
    sv_obj = float(sv_exp.real + offset)
    results.append(BackendResult(
        "statevector", None, n, sv_elapsed, sv_obj,
        (sv_obj - exact_val) if exact_val is not None else None,
    ))
    print(f"  statevector: {sv_elapsed:.2f}s, obj(expectation)={sv_obj:.6f}")

    for bd in bond_dimensions:
        kwargs = {"method": "matrix_product_state"}
        if bd is not None:
            kwargs["matrix_product_state_max_bond_dimension"] = bd
        mps_sim = AerSimulator(**kwargs)
        t0 = time.time()
        try:
            mps_exp = mps_sim.run(transpile(bound_no_meas, mps_sim)).result().data(0)["expectation_value"]
            mps_elapsed = time.time() - t0
            mps_obj = float(mps_exp.real + offset)
            label = f"MPS (bond<={bd})" if bd is not None else "MPS (uncapped)"
            print(f"  {label}: {mps_elapsed:.2f}s, obj(expectation)={mps_obj:.6f}")
            results.append(BackendResult(
                "mps", bd, n, mps_elapsed, mps_obj,
                (mps_obj - exact_val) if exact_val is not None else None,
            ))
        except Exception as e:
            print(f"  MPS (bond<={bd}): FAILED ({e})")
            results.append(BackendResult("mps", bd, n, np.inf, np.inf, None))

    return results


def results_to_dataframe(results: list[BackendResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "backend": r.backend,
        "max_bond_dim": r.max_bond_dimension if r.max_bond_dimension else "uncapped",
        "n_qubits": r.n_qubits,
        "runtime_s": round(r.runtime_s, 3),
        "objective": round(r.objective, 6),
        "gap_to_exact": round(r.gap_to_exact, 6) if r.gap_to_exact is not None else None,
    } for r in results])
