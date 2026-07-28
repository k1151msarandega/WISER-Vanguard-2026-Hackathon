"""
Classical mean-variance (Markowitz) baseline.

This is required by the brief explicitly, and it doubles as the fairness
anchor: every later QAOA/QUBO result gets compared against what a classical
convex solver achieves under the *same* constraints (sector caps, turnover,
per-asset cap) -- not a looser or differently-constrained version of the
problem. That equal-footing comparison was flagged as missing in earlier
work, so the constraint set here is written to be reused as-is by the QUBO
formulation in week 2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import cvxpy as cp

from vqportfolio.config import ASSET_CLASS_OF, ASSET_CLASS_CAPS


def solve_markowitz(
    mu: pd.Series,
    sigma: pd.DataFrame,
    cost_bps: pd.Series,
    tickers: list[str],
    prev_weights: pd.Series | None = None,
    risk_aversion: float = 3.0,
    max_weight_per_asset: float = 0.25,
    max_turnover: float = 1.0,  # 1.0 = unconstrained (first run, no prior portfolio)
) -> dict:
    """Solve max( mu^T w - risk_aversion * w^T Sigma w - cost^T |w - w_prev| )
    subject to: sum(w) = 1, 0 <= w <= max_weight_per_asset,
                sum(w in class c) <= cap_c for each asset class,
                turnover <= max_turnover.

    Returns a dict with weights, objective breakdown, and guardrail-breach
    diagnostics (should be ~0 breaches by construction, but we check anyway --
    solver tolerance issues are a real failure mode worth catching, not
    assuming away).
    """
    n = len(tickers)
    mu_vec = mu.loc[tickers].values
    sigma_mat = sigma.loc[tickers, tickers].values
    cost_vec = (cost_bps.loc[tickers].values) / 10_000  # bps -> fraction

    if prev_weights is None:
        prev_w = np.full(n, 1.0 / n)  # default prior: equal-weight
    else:
        prev_w = prev_weights.reindex(tickers).fillna(0.0).values

    w = cp.Variable(n)
    turnover_expr = cp.norm1(w - prev_w)

    expected_return = mu_vec @ w
    risk = cp.quad_form(w, sigma_mat, assume_PSD=True)
    cost = cost_vec @ cp.abs(w - prev_w)

    objective = cp.Maximize(expected_return - risk_aversion * risk - cost)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= max_weight_per_asset,
        turnover_expr <= max_turnover,
    ]

    for asset_class, cap in ASSET_CLASS_CAPS.items():
        idx = [i for i, t in enumerate(tickers) if ASSET_CLASS_OF[t] == asset_class]
        if idx:
            constraints.append(cp.sum(w[idx]) <= cap)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Markowitz solve failed: status={problem.status}")

    weights = pd.Series(np.clip(w.value, 0, None), index=tickers)
    weights = weights / weights.sum()  # renormalize away tiny solver slack

    # --- guardrail diagnostics ---
    breaches = {}
    for asset_class, cap in ASSET_CLASS_CAPS.items():
        class_weight = weights[[t for t in tickers if ASSET_CLASS_OF[t] == asset_class]].sum()
        if class_weight > cap + 1e-4:
            breaches[asset_class] = round(class_weight - cap, 4)

    realized_turnover = float(np.abs(weights.values - prev_w).sum())

    return {
        "weights": weights,
        "expected_return": float(mu_vec @ weights.values),
        "risk_variance": float(weights.values @ sigma_mat @ weights.values),
        "turnover": realized_turnover,
        "guardrail_breaches": breaches,  # should be {} -- solver enforces caps as hard constraints
        "solver_status": problem.status,
    }


if __name__ == "__main__":
    from vqportfolio.market_data.loader import load_prices
    from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
    from vqportfolio.config import TICKERS

    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns)

    result = solve_markowitz(mu, sigma, overlay["cost_bps"], TICKERS)

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}\n")
    print("Solver status:", result["solver_status"])
    print(f"Expected return: {result['expected_return']:.4f}")
    print(f"Risk (variance): {result['risk_variance']:.4f}")
    print(f"Turnover: {result['turnover']:.4f}")
    print(f"Guardrail breaches: {result['guardrail_breaches']}\n")
    print("Weights:")
    print(result["weights"].sort_values(ascending=False).round(4))
