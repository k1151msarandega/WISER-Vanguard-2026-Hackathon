"""
QUBO formulation for the O(ptimize) set.

Each O-asset's weight is binarized into `bits` binary variables:
    weight_i = (cap / (2**bits - 1)) * sum_k 2**k * b_{i,k}
which gives 2**bits discrete levels between 0 and `max_weight_per_asset`.
Per-asset cap is therefore enforced *by construction* (the encoding can't
represent a weight above the cap), not via a penalty term.

Objective (maximize): mu^T w - risk_aversion * w^T Sigma w - cost^T w
  -- same risk_aversion convention as the classical Markowitz baseline, so
  results are comparable on the same footing.

Constraints:
  - budget equality: sum(w) == o_budget (the leftover after H is locked)
  - class-cap inequalities: only added for asset classes where the class's
    *maximum possible* weight within O (i.e. every O-member of that class
    sitting at max_weight_per_asset) would exceed the remaining headroom
    left by H. Non-binding class constraints are skipped to avoid burning
    extra qubits on slack variables for a constraint that can't possibly
    bind -- documented tradeoff, not an oversight.

We build this via qiskit_optimization.QuadraticProgram and convert with
QuadraticProgramToQubo, which is the standard, auditable way to turn
constrained problems into an unconstrained QUBO (handles equality-as-penalty
and inequality-via-slack-variables automatically rather than hand-rolled
penalty engineering).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

from vqportfolio.config import ASSET_CLASS_OF


@dataclass
class QuboBuildResult:
    quadratic_program: QuadraticProgram
    qubo: QuadraticProgram          # unconstrained, post-conversion
    bit_index: list[tuple[str, int]]  # ordered (ticker, bit_position) for each objective variable
    bits: int
    max_weight_per_asset: float

    def decode_weights(self, x: np.ndarray) -> pd.Series:
        """Decode a bitstring (values for the *objective* variables only --
        slack variables, if any, are ignored) into per-asset weights."""
        coef = self.max_weight_per_asset / (2 ** self.bits - 1)
        weights = {}
        for (ticker, k), bit in zip(self.bit_index, x[: len(self.bit_index)]):
            weights[ticker] = weights.get(ticker, 0.0) + coef * (2 ** k) * bit
        return pd.Series(weights)


def build_o_set_qubo(
    o_tickers: list[str],
    mu: pd.Series,
    sigma: pd.DataFrame,
    cost_bps: pd.Series,
    o_budget: float,
    class_headroom: dict[str, float],
    bits: int = 3,
    max_weight_per_asset: float = 0.25,
    risk_aversion: float = 3.0,
    penalty: float | None = None,
) -> QuboBuildResult:
    n = len(o_tickers)
    coef = max_weight_per_asset / (2 ** bits - 1)
    bit_index = [(t, k) for t in o_tickers for k in range(bits)]
    var_names = [f"b_{t}_{k}" for t, k in bit_index]

    qp = QuadraticProgram(name="o_set_portfolio")
    for name in var_names:
        qp.binary_var(name=name)

    # --- linear part: return - cost ---
    mu_vec = mu.loc[o_tickers]
    cost_vec = cost_bps.loc[o_tickers] / 10_000  # bps -> fraction
    linear = {}
    for (t, k), name in zip(bit_index, var_names):
        linear[name] = coef * (2 ** k) * (mu_vec[t] - cost_vec[t])

    # --- quadratic part: -risk_aversion * w^T Sigma w ---
    sigma_oo = sigma.loc[o_tickers, o_tickers]
    quadratic = {}
    for (t_i, k_i), name_i in zip(bit_index, var_names):
        for (t_j, k_j), name_j in zip(bit_index, var_names):
            c = -risk_aversion * coef * coef * (2 ** k_i) * (2 ** k_j) * sigma_oo.loc[t_i, t_j]
            if abs(c) > 1e-14:
                quadratic[(name_i, name_j)] = quadratic.get((name_i, name_j), 0.0) + c

    qp.maximize(linear=linear, quadratic=quadratic)

    # --- budget equality constraint ---
    # Expressed in integer "tick" units (multiples of `coef`) rather than raw
    # weight fractions: qiskit_optimization's automatic inequality-to-equality
    # slack-variable machinery needs integer coefficients to compute slack
    # bit-width, and floats break that. Same physical constraint, just
    # rescaled by 1/coef. This does mean the budget is matched to the nearest
    # multiple of `coef` rather than exactly -- an inherent consequence of
    # binarized weights at this bit resolution, not a new approximation.
    budget_linear = {name: (2 ** k) for (t, k), name in zip(bit_index, var_names)}
    budget_rhs_ticks = round(o_budget / coef)
    qp.linear_constraint(linear=budget_linear, sense="==", rhs=budget_rhs_ticks, name="budget")

    # --- class cap inequalities (only where binding is even possible) ---
    for asset_class, headroom in class_headroom.items():
        idx = [t for t in o_tickers if ASSET_CLASS_OF[t] == asset_class]
        if not idx:
            continue
        max_possible = len(idx) * max_weight_per_asset
        if max_possible <= headroom + 1e-9:
            continue  # can't possibly bind, skip -- saves slack qubits
        class_linear = {
            name: (2 ** k)
            for (t, k), name in zip(bit_index, var_names)
            if t in idx
        }
        # floor (not round) the tick cap -- conservative, guarantees the
        # decoded weight never exceeds the real headroom even after rounding
        cap_rhs_ticks = int(np.floor(headroom / coef))
        qp.linear_constraint(linear=class_linear, sense="<=", rhs=cap_rhs_ticks, name=f"cap_{asset_class}")

    converter = QuadraticProgramToQubo(penalty=penalty)
    qubo = converter.convert(qp)

    return QuboBuildResult(
        quadratic_program=qp,
        qubo=qubo,
        bit_index=bit_index,
        bits=bits,
        max_weight_per_asset=max_weight_per_asset,
    )


if __name__ == "__main__":
    from vqportfolio.market_data.loader import load_prices
    from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
    from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
    from vqportfolio.partitioning.partition import partition_assets, build_locked_allocation, PartitionConfig
    from vqportfolio.config import TICKERS, ASSET_CLASS_OF, ASSET_CLASS_CAPS

    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns)
    mdd = per_asset_max_drawdown(prices)

    dials = Dials()
    scores_df = compute_asset_scores(mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, dials)
    pconfig = PartitionConfig()
    partition = partition_assets(scores_df["score"], pconfig)
    h_weights, o_budget = build_locked_allocation(mu, sigma, overlay["cost_bps"], partition, pconfig)

    # headroom left in each class after H is locked
    class_headroom = {}
    for asset_class, cap in ASSET_CLASS_CAPS.items():
        idx = [t for t in partition["H"] if ASSET_CLASS_OF[t] == asset_class]
        used = h_weights.loc[idx].sum() if idx else 0.0
        class_headroom[asset_class] = max(cap - used, 0.0)

    result = build_o_set_qubo(
        partition["O"], mu, sigma, overlay["cost_bps"], o_budget, class_headroom,
    )

    print(f"USED_SYNTHETIC_PRICES = {used_synthetic}\n")
    print(f"O-set: {partition['O']}, budget={o_budget:.4f}")
    print(f"Class headroom: {class_headroom}\n")
    print(f"Objective variables: {len(result.bit_index)}")
    print(f"Total QUBO variables (incl. slack): {result.qubo.get_num_binary_vars()}")
    print(f"Constraints in original QP: {len(result.quadratic_program.linear_constraints)}")
