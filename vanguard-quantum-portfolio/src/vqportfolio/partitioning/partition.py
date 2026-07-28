"""
Partition the universe into H(old/lock), O(ptimize), S(kip) based on
conviction score, then assign locked weights to H via water-filling so
per-asset and per-asset-class caps are respected before the quantum stage
ever runs.

Reworked from the AQC hold/sell version: that version assigned a single bit
per asset (in or out). Here, H assets get a real *weight* (not just "in"),
and O's weights are optimized subject to whatever budget H leaves behind --
O never sees a budget that would blow through the guardrails, because H's
allocation is capped and renormalized before O is even sized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import cvxpy as cp

from vqportfolio.config import ASSET_CLASS_OF, ASSET_CLASS_CAPS


@dataclass(frozen=True)
class PartitionConfig:
    h_fraction: float = 0.35   # top fraction of universe (by score) eligible for H
    s_fraction: float = 0.35   # bottom fraction of universe eligible for S (excluded, weight=0)
    max_o_size: int = 4        # hard cap on |O| -- keeps qubit count tractable for QAOA
    max_weight_per_asset: float = 0.25
    h_budget_cap: float = 0.70  # H can claim at most this fraction of total portfolio


def optimize_locked_allocation(
    mu: pd.Series,
    sigma: pd.DataFrame,
    cost_bps: pd.Series,
    partition: dict,
    config: PartitionConfig | None = None,
    risk_aversion: float = 3.0,
) -> tuple[pd.Series, float]:
    """Allocate weights across H by solving the *same* mean-variance-cost
    objective as the classical Markowitz baseline, restricted to H's
    membership, with h_budget_cap as an upper bound (not a forced total).

    This replaces water_filling_allocate as the default: water-filling
    distributes a fixed budget proportional to score, which is fast but not
    the solution to any actual optimization problem -- it can't tell you
    whether spending the full h_budget_cap on these assets is even a good
    idea. Solving the real constrained QP means H's allocation is provably
    optimal for its own objective (given fixed membership), so any gap
    between the full portfolio and Markowitz is attributable to
    partitioning strategy and O's QAOA solution quality -- not to slack in
    a heuristic H step. It also means H will naturally leave budget on the
    table for O if H's members don't justify using the full ceiling,
    rather than always spending exactly h_budget_cap regardless.

    Returns (H weights, remaining budget for O).
    """
    config = config or PartitionConfig()
    h_tickers = partition["H"]
    n = len(h_tickers)
    if n == 0:
        return pd.Series(dtype=float), 1.0

    mu_h = mu.loc[h_tickers].values
    sigma_h = sigma.loc[h_tickers, h_tickers].values
    cost_h = (cost_bps.loc[h_tickers].values) / 10_000

    w = cp.Variable(n)
    objective = cp.Maximize(mu_h @ w - risk_aversion * cp.quad_form(w, sigma_h, assume_PSD=True) - cost_h @ w)

    constraints = [
        w >= 0,
        w <= config.max_weight_per_asset,
        cp.sum(w) <= config.h_budget_cap,  # ceiling, not equality -- H can leave budget for O
    ]
    for asset_class, cap in ASSET_CLASS_CAPS.items():
        idx = [i for i, t in enumerate(h_tickers) if ASSET_CLASS_OF[t] == asset_class]
        if idx:
            constraints.append(cp.sum(w[idx]) <= cap)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"H allocation solve failed: status={problem.status}")

    h_weights = pd.Series(np.clip(w.value, 0, None), index=h_tickers)
    o_budget = 1.0 - h_weights.sum()
    return h_weights, o_budget


def water_filling_allocate(
    scores: pd.Series,
    budget: float,
    tickers: list[str],
    max_weight_per_asset: float,
    class_caps: dict[str, float],
    class_usage_before: dict[str, float] | None = None,
    max_iter: int = 50,
) -> pd.Series:
    """Allocate `budget` across `tickers` proportional to softmax(scores),
    then iteratively clip to per-asset cap and remaining per-class cap
    headroom, redistributing any clipped-off leftover among assets that
    still have room. Converges when no leftover remains or no asset has
    headroom left (in which case leftover budget goes unallocated -- better
    to under-allocate than silently violate a guardrail).
    """
    class_usage_before = class_usage_before or {}
    n = len(tickers)
    if n == 0 or budget <= 1e-9:
        return pd.Series(0.0, index=tickers)

    s = scores.loc[tickers].values
    exp_s = np.exp(s - s.max())
    raw_weights = exp_s / exp_s.sum() * budget
    weights = pd.Series(raw_weights, index=tickers)

    for _ in range(max_iter):
        # per-asset cap
        capped = weights.clip(upper=max_weight_per_asset)
        # per-class remaining headroom cap
        for asset_class, cap in class_caps.items():
            idx = [t for t in tickers if ASSET_CLASS_OF[t] == asset_class]
            if not idx:
                continue
            used_elsewhere = class_usage_before.get(asset_class, 0.0)
            headroom = max(cap - used_elsewhere, 0.0)
            class_total = capped.loc[idx].sum()
            if class_total > headroom + 1e-9 and class_total > 0:
                capped.loc[idx] = capped.loc[idx] * (headroom / class_total)

        leftover = weights.sum() - capped.sum()
        weights = capped
        if leftover <= 1e-9:
            break

        # redistribute leftover among assets with remaining headroom
        headroom_per_asset = (max_weight_per_asset - weights).clip(lower=0.0)
        if headroom_per_asset.sum() <= 1e-9:
            break  # no room anywhere -- leftover goes unallocated, not force-violated
        add = headroom_per_asset / headroom_per_asset.sum() * leftover
        weights = weights + add

    return weights


def partition_assets(
    scores: pd.Series,
    config: PartitionConfig | None = None,
) -> dict:
    """Return {'H': [...], 'O': [...], 'S': [...]} ticker lists.

    H = top-scoring assets (by h_fraction), further trimmed if needed.
    S = bottom-scoring assets (by s_fraction), weight forced to 0.
    O = whatever's left in the middle, hard-capped at max_o_size (if the
        remaining pool is bigger than max_o_size, keep the highest-scoring
        max_o_size of them and push the rest into S -- documented tradeoff,
        not silently dropped).
    """
    config = config or PartitionConfig()
    ranked = scores.sort_values(ascending=False)
    n = len(ranked)

    n_h = max(1, int(round(n * config.h_fraction)))
    n_s = max(1, int(round(n * config.s_fraction)))
    n_h = min(n_h, n - 1)
    n_s = min(n_s, n - n_h - 1)

    h_tickers = list(ranked.index[:n_h])
    remaining = list(ranked.index[n_h:n - n_s]) if n - n_s > n_h else []
    s_tickers = list(ranked.index[n - n_s:]) if n_s > 0 else []

    if len(remaining) > config.max_o_size:
        # keep the highest-scoring max_o_size of the "remaining" pool in O,
        # push the rest to S rather than silently dropping them
        remaining_ranked = ranked.loc[remaining].sort_values(ascending=False)
        o_tickers = list(remaining_ranked.index[:config.max_o_size])
        pushed_to_s = list(remaining_ranked.index[config.max_o_size:])
        s_tickers = s_tickers + pushed_to_s
    else:
        o_tickers = remaining

    return {"H": h_tickers, "O": o_tickers, "S": s_tickers}


def build_locked_allocation(
    mu: pd.Series,
    sigma: pd.DataFrame,
    cost_bps: pd.Series,
    partition: dict,
    config: PartitionConfig | None = None,
    risk_aversion: float = 3.0,
) -> tuple[pd.Series, float]:
    """Allocate weights across H. Default path: solve the constrained
    mean-variance-cost QP over H's membership (see
    optimize_locked_allocation) -- provably optimal for that objective,
    not a heuristic. Returns (H weights, remaining budget for O).

    water_filling_allocate() is still available directly for anyone who
    wants the old proportional-to-score heuristic for comparison, but it's
    no longer the default -- see the module docstring on
    optimize_locked_allocation for why.
    """
    config = config or PartitionConfig()
    return optimize_locked_allocation(mu, sigma, cost_bps, partition, config, risk_aversion)


if __name__ == "__main__":
    from vqportfolio.market_data.loader import load_prices
    from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
    from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
    from vqportfolio.config import TICKERS

    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns)
    mdd = per_asset_max_drawdown(prices)

    dials = Dials()
    scores_df = compute_asset_scores(mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, dials)

    config = PartitionConfig()
    partition = partition_assets(scores_df["score"], config)
    h_weights, o_budget = build_locked_allocation(mu, sigma, overlay["cost_bps"], partition, config)

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}\n")
    print(f"H ({len(partition['H'])}): {partition['H']}")
    print(f"O ({len(partition['O'])}): {partition['O']}")
    print(f"S ({len(partition['S'])}): {partition['S']}\n")
    print("Locked H weights:")
    print(h_weights.round(4))
    print(f"\nH total: {h_weights.sum():.4f}")
    print(f"Remaining O budget: {o_budget:.4f}")

    # guardrail check on H alone
    for asset_class, cap in ASSET_CLASS_CAPS.items():
        idx = [t for t in partition["H"] if ASSET_CLASS_OF[t] == asset_class]
        used = h_weights.loc[idx].sum() if idx else 0.0
        flag = "  <-- BREACH" if used > cap + 1e-6 else ""
        print(f"  {asset_class}: {used:.4f} / cap {cap}{flag}")
