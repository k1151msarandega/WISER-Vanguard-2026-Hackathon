"""
Equal-footing classical benchmarks for the O-set: ILP, greedy, and random,
all solving the EXACT SAME discretized problem QAOA solves -- same
bit-encoded weights (same `bits`, same per-asset cap enforced by the
encoding), same budget equality constraint, same class-cap inequalities.

This is the comparison flagged as missing after Week 2: the existing
Markowitz baseline (baseline/markowitz.py) solves a genuinely different
problem -- continuous weights over the FULL 15-asset universe, with H not
pre-fixed. That's a legitimate, useful comparison for a different question
("how does the whole H/O/S+QAOA pipeline compare to unconstrained classical
optimization") but it is NOT an equal-footing comparison for "how good is
QAOA specifically, on the exact combinatorial problem it was handed." This
module is the latter.

All methods are scored via the ORIGINAL (pre-QUBO-conversion) quadratic
program's objective (`quadratic_program.objective`, MAXIMIZE sense --
returns/cost/risk directly, higher is better) rather than the converted
QUBO's penalized objective (`qubo.objective`, MINIMIZE sense, used
elsewhere in the project e.g. qaoa_solver.py/scaling_ablation.py). This is
a deliberate, necessary difference, not an inconsistency to be confused for
a bug: ILP/greedy/random satisfy the TRUE constraints directly and produce
no slack-variable values at all, while `qubo.objective.evaluate()` requires
a full-length bitstring including slack (this was discovered the hard way
-- an early version of this module crashed with a dimension mismatch on an
O-set that needed slack qubits, since only 12 of the required 15 values
were being supplied). Rather than reverse-engineer QuadraticProgramToQubo's
internal slack bit-encoding just to pad a vector that has no real
constraint-violation content anyway, every method here (including QAOA and
exact, which DO have full-length solutions) is scored via the true
objective on just the decision variables. When comparing against
QUBO-minimize-convention numbers reported elsewhere in this project, note
the sign flip and that this excludes penalty-term contamination.

Method design, mirroring the precedent set by earlier (AQC hackathon)
classical-benchmark conventions:
  - ILP: the risk term uses a DIAGONAL-ONLY covariance approximation (drops
    cross-asset covariance), which keeps the objective genuinely LINEAR in
    the binary variables (b_i^2 = b_i for binary b, so even the diagonal
    quadratic term collapses to linear) -- solvable by a real open-source
    MILP solver (PuLP + CBC), not an approximation of a QUBO solver. This is
    a real methodological simplification (ignoring cross-covariance), not a
    shortcut disguised as exact -- documented, not hidden.
  - Greedy: single-pass, deterministic, ranks assets by a marginal
    return-risk-cost score (ignoring interaction effects, like the ILP
    linearization) and fills ticks highest-ranked-first until budget/caps
    bind. No search, no iteration -- intentionally the simplest possible
    baseline.
  - Random: rejection-sampled feasible tick allocations (exact budget match,
    respecting class caps), scored via the real objective, best-of-N and
    mean/std reported -- not just one draw, so the comparison reflects the
    distribution, not luck.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import pulp

from vqportfolio.config import ASSET_CLASS_OF
from vqportfolio.quantum.qubo import QuboBuildResult


@dataclass
class BenchmarkResult:
    method: str
    weights: pd.Series
    objective: float  # QUBO-units, same scale/sign convention as qaoa_solver/scaling_ablation
    solve_time_s: float
    feasible: bool
    extra: dict | None = None  # method-specific info (e.g. random's mean/std across samples)


def weight_space_objective(
    weights: pd.Series,
    o_tickers: list[str],
    mu: pd.Series,
    sigma: pd.DataFrame,
    cost_bps: pd.Series,
    risk_aversion: float = 3.0,
) -> float:
    """The true financial objective (return - risk - cost, MAXIMIZE sense,
    higher is better) computed directly from a weights vector -- not from
    bits/QUBO internals. This is the single source of truth this whole
    module scores every method against, including QAOA: QAOA's *repaired*
    weights are continuous-rescaled (see qaoa_solver.repair_weights) and no
    longer correspond to any exact bitstring, so deriving an objective from
    bits wouldn't even be well-defined for it post-repair. Scoring
    everything from weights directly sidesteps that entirely and is what
    actually matters financially anyway.
    """
    w = weights.reindex(o_tickers).fillna(0.0)
    mu_vec = mu.loc[o_tickers]
    sigma_mat = sigma.loc[o_tickers, o_tickers]
    cost_vec = cost_bps.loc[o_tickers] / 10_000
    return float(mu_vec @ w - risk_aversion * (w.values @ sigma_mat.values @ w.values) - cost_vec @ w)


def _tick_bounds(qubo_result: QuboBuildResult, o_tickers: list[str]) -> tuple[dict, int]:
    """Per-asset max tick value (2^bits - 1) and the coef converting ticks to weight fraction."""
    max_ticks = 2 ** qubo_result.bits - 1
    return {t: max_ticks for t in o_tickers}, max_ticks


def solve_ilp_diagonal_risk(
    qubo_result: QuboBuildResult,
    o_tickers: list[str],
    mu: pd.Series,
    sigma: pd.DataFrame,
    cost_bps: pd.Series,
    o_budget: float,
    class_headroom: dict[str, float],
    risk_aversion: float = 3.0,
    time_limit_s: int = 30,
) -> BenchmarkResult:
    bits = qubo_result.bits
    coef = qubo_result.max_weight_per_asset / (2 ** bits - 1)
    max_ticks, _ = _tick_bounds(qubo_result, o_tickers)

    prob = pulp.LpProblem("o_set_ilp_diagonal", pulp.LpMaximize)
    b = {(t, k): pulp.LpVariable(f"b_{t}_{k}", cat="Binary") for t in o_tickers for k in range(bits)}

    cost_frac = cost_bps.loc[o_tickers] / 10_000
    # objective: linear return/cost term + DIAGONAL-only risk (b_i^2 = b_i for binary vars,
    # so this stays linear -- cross terms i != j are what's dropped for tractability)
    obj_terms = []
    for t in o_tickers:
        for k in range(bits):
            w_tk = coef * (2 ** k)
            lin_coef = w_tk * (mu[t] - cost_frac[t]) - risk_aversion * (w_tk ** 2) * sigma.loc[t, t]
            obj_terms.append(lin_coef * b[(t, k)])
    prob += pulp.lpSum(obj_terms)

    # budget equality (same tick-unit convention as the QUBO)
    budget_ticks = round(o_budget / coef)
    prob += pulp.lpSum((2 ** k) * b[(t, k)] for t in o_tickers for k in range(bits)) == budget_ticks

    # class caps (same convention as the QUBO -- only add if it could bind)
    for asset_class, headroom in class_headroom.items():
        idx = [t for t in o_tickers if ASSET_CLASS_OF[t] == asset_class]
        if not idx:
            continue
        max_possible = len(idx) * qubo_result.max_weight_per_asset
        if max_possible <= headroom + 1e-9:
            continue
        cap_ticks = int(np.floor(headroom / coef))
        prob += pulp.lpSum((2 ** k) * b[(t, k)] for t in idx for k in range(bits)) <= cap_ticks

    t0 = time.time()
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    prob.solve(solver)
    elapsed = time.time() - t0

    feasible = pulp.LpStatus[prob.status] == "Optimal"
    x = np.array([b[(t, k)].value() or 0 for t, k in qubo_result.bit_index])
    weights = qubo_result.decode_weights(x)
    objective = weight_space_objective(weights, o_tickers, mu, sigma, cost_bps, risk_aversion)

    return BenchmarkResult("ILP (diagonal risk)", weights, objective, elapsed, feasible)


def solve_greedy(
    qubo_result: QuboBuildResult,
    o_tickers: list[str],
    mu: pd.Series,
    sigma: pd.DataFrame,
    cost_bps: pd.Series,
    o_budget: float,
    class_headroom: dict[str, float],
    risk_aversion: float = 3.0,
) -> BenchmarkResult:
    bits = qubo_result.bits
    coef = qubo_result.max_weight_per_asset / (2 ** bits - 1)
    max_ticks, _ = _tick_bounds(qubo_result, o_tickers)
    cost_frac = cost_bps.loc[o_tickers] / 10_000

    t0 = time.time()
    # single-pass score, ignoring interaction effects (same simplification as the ILP)
    score = {t: mu[t] - cost_frac[t] - risk_aversion * sigma.loc[t, t] for t in o_tickers}
    ranked = sorted(o_tickers, key=lambda t: -score[t])

    budget_ticks = round(o_budget / coef)
    class_cap_ticks = {}
    for asset_class, headroom in class_headroom.items():
        idx = [t for t in o_tickers if ASSET_CLASS_OF[t] == asset_class]
        if idx:
            class_cap_ticks[asset_class] = int(np.floor(headroom / coef))

    ticks_used = {t: 0 for t in o_tickers}
    class_ticks_used = {ac: 0 for ac in class_cap_ticks}
    remaining_budget = budget_ticks

    for t in ranked:
        if remaining_budget <= 0:
            break
        ac = ASSET_CLASS_OF[t]
        class_room = class_cap_ticks.get(ac, np.inf) - class_ticks_used.get(ac, 0)
        take = min(max_ticks[t], remaining_budget, class_room)
        take = max(take, 0)
        ticks_used[t] = take
        remaining_budget -= take
        if ac in class_ticks_used:
            class_ticks_used[ac] += take
    elapsed = time.time() - t0

    x = np.array([1 if (ticks_used[t] >> k) & 1 else 0 for t, k in qubo_result.bit_index])
    weights = qubo_result.decode_weights(x)
    objective = weight_space_objective(weights, o_tickers, mu, sigma, cost_bps, risk_aversion)

    return BenchmarkResult("Greedy", weights, objective, elapsed, True,
                            extra={"budget_ticks_filled": budget_ticks - remaining_budget,
                                   "budget_ticks_target": budget_ticks})


def solve_random(
    qubo_result: QuboBuildResult,
    o_tickers: list[str],
    mu: pd.Series,
    sigma: pd.DataFrame,
    cost_bps: pd.Series,
    o_budget: float,
    class_headroom: dict[str, float],
    risk_aversion: float = 3.0,
    n_samples: int = 500,
    max_attempts: int = 20_000,
    seed: int = 7,
) -> BenchmarkResult:
    bits = qubo_result.bits
    coef = qubo_result.max_weight_per_asset / (2 ** bits - 1)
    max_ticks, _ = _tick_bounds(qubo_result, o_tickers)
    n = len(o_tickers)

    budget_ticks = round(o_budget / coef)
    class_cap_ticks = {}
    for asset_class, headroom in class_headroom.items():
        idx = [t for t in o_tickers if ASSET_CLASS_OF[t] == asset_class]
        if idx:
            class_cap_ticks[asset_class] = (idx, int(np.floor(headroom / coef)))

    rng = np.random.default_rng(seed)
    t0 = time.time()
    feasible_samples = []
    attempts = 0
    while len(feasible_samples) < n_samples and attempts < max_attempts:
        attempts += 1
        # random composition of budget_ticks into n parts, each in [0, max_ticks]
        raw = rng.integers(0, max_ticks[o_tickers[0]] + 1, size=n) if n > 0 else np.array([])
        if raw.sum() == 0:
            continue
        scaled = np.floor(raw / raw.sum() * budget_ticks).astype(int)
        # fix rounding remainder
        remainder = budget_ticks - scaled.sum()
        for i in range(abs(remainder)):
            idx = i % n
            scaled[idx] += 1 if remainder > 0 else -1
        scaled = np.clip(scaled, 0, [max_ticks[t] for t in o_tickers])
        if scaled.sum() != budget_ticks:
            continue

        sample = dict(zip(o_tickers, scaled))
        ok = True
        for ac, (idx, cap) in class_cap_ticks.items():
            if sum(sample[t] for t in idx) > cap:
                ok = False
                break
        if ok:
            feasible_samples.append(sample)
    elapsed = time.time() - t0

    if not feasible_samples:
        return BenchmarkResult("Random", pd.Series(dtype=float), float("-inf"), elapsed, False,
                                extra={"attempts": attempts, "feasible_found": 0})

    objectives = []
    decoded_weights_per_sample = []
    for sample in feasible_samples:
        x = np.array([1 if (sample[t] >> k) & 1 else 0 for t, k in qubo_result.bit_index])
        w = qubo_result.decode_weights(x)
        decoded_weights_per_sample.append(w)
        objectives.append(weight_space_objective(w, o_tickers, mu, sigma, cost_bps, risk_aversion))

    best_idx = int(np.argmax(objectives))
    weights = decoded_weights_per_sample[best_idx]

    return BenchmarkResult(
        "Random (best-of-N)", weights, objectives[best_idx], elapsed, True,
        extra={"n_feasible_samples": len(feasible_samples), "attempts": attempts,
               "mean_objective": float(np.mean(objectives)), "std_objective": float(np.std(objectives))},
    )


def results_summary(results: list[BenchmarkResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "method": r.method,
        "objective": round(r.objective, 6),
        "solve_time_s": round(r.solve_time_s, 4),
        "feasible": r.feasible,
    } for r in results])
