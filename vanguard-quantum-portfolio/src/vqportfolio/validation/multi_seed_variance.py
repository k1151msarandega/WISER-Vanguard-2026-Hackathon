"""
Multi-seed QAOA variance reporting -- the Week 3 item flagged as missing
after an earlier ad-hoc spot check (7 seeds, interactive, never saved) found
one catastrophic outlier alongside otherwise-exact results. That spot check
motivated the warm-start + multi-restart fix in quantum/qaoa_solver.py, but
no systematic multi-seed run was ever built into the codebase itself --
this module is that.

Every seed's result is scored via weight_space_objective() (see
classical_benchmarks.py) for consistency with the rest of Week 3's
comparisons, not via qubo.objective.evaluate() -- same reasoning as there:
repaired weights aren't bit-representable, so a bits-based objective isn't
well-defined for every seed's output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vqportfolio.quantum.qaoa_solver import solve_with_qaoa_and_validate
from vqportfolio.quantum.qubo import QuboBuildResult
from vqportfolio.validation.classical_benchmarks import weight_space_objective


@dataclass
class SeedResult:
    seed: int
    objective: float
    matches_exact: bool
    repair_applied: bool
    solve_ok: bool
    error: str | None = None


def run_seed(
    qubo_result: QuboBuildResult,
    o_tickers: list[str],
    mu: pd.Series,
    sigma: pd.DataFrame,
    cost_bps: pd.Series,
    o_budget: float,
    class_headroom: dict[str, float],
    seed: int,
    risk_aversion: float = 3.0,
    n_restarts: int = 2,
    maxiter: int = 40,
    shots: int = 512,
) -> SeedResult:
    try:
        comp = solve_with_qaoa_and_validate(
            qubo_result, o_budget, class_headroom, seed=seed,
            n_restarts=n_restarts, maxiter=maxiter, shots=shots,
        )
        obj = weight_space_objective(comp.qaoa_weights, o_tickers, mu, sigma, cost_bps, risk_aversion)
        return SeedResult(seed, obj, comp.qaoa_matches_exact, comp.repair_applied, True)
    except Exception as e:
        return SeedResult(seed, float("nan"), False, False, False, error=str(e))


def summarize(results: list[SeedResult], exact_objective: float | None = None) -> dict:
    ok = [r for r in results if r.solve_ok]
    objectives = np.array([r.objective for r in ok])
    summary = {
        "n_seeds": len(results),
        "n_ok": len(ok),
        "n_failed": len(results) - len(ok),
        "n_matches_exact": sum(r.matches_exact for r in ok),
        "n_repair_applied": sum(r.repair_applied for r in ok),
        "mean_objective": float(np.mean(objectives)) if len(objectives) else float("nan"),
        "std_objective": float(np.std(objectives)) if len(objectives) else float("nan"),
        "min_objective": float(np.min(objectives)) if len(objectives) else float("nan"),
        "max_objective": float(np.max(objectives)) if len(objectives) else float("nan"),
    }
    if exact_objective is not None and len(objectives):
        gaps = exact_objective - objectives  # exact is max-sense too, so gap >= 0 for worse-than-exact
        summary["mean_gap_to_exact"] = float(np.mean(gaps))
        summary["max_gap_to_exact"] = float(np.max(gaps))  # worst single seed
    return summary
