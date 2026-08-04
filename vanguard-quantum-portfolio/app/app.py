"""
Portfolio Co-Pilot demo (Streamlit).

The required deliverable per the brief: "Demonstrate a portfolio co-pilot
that shows the recommended allocation, the trade-offs versus baseline, and
why the solution satisfies constraints."

Design note on why the two halves of this page behave differently: dial
changes (growth/income/drawdown/cost) update the H/O/S partition and H's
locked allocation INSTANTLY (both are fast, <1s -- scoring is a handful of
z-scores, H's allocation is a small convex QP). The quantum optimization of
the O-set is NOT instant (warm-start + multi-restart QAOA takes ~1-2
minutes) and is deliberately gated behind an explicit button rather than
auto-triggered on every slider drag. This isn't just a UX compromise -- it
mirrors the actual H/O/S architecture: H is the fast classical part, O is
where real quantum computation happens. The demo's two-speed interaction
model is the architecture, not a limitation hidden from the user.

Run: streamlit run app/app.py  (from the repo root, after `pip install -e ".[app]"`)
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from vqportfolio.config import TICKERS, ASSET_CLASS_OF, ASSET_CLASS_CAPS
from vqportfolio.market_data.loader import load_prices
from vqportfolio.market_data.overlays import compute_returns_and_risk, compute_cost_and_yield
from vqportfolio.partitioning.scoring import compute_asset_scores, per_asset_max_drawdown, Dials
from vqportfolio.partitioning.partition import partition_assets, build_locked_allocation, PartitionConfig
from vqportfolio.pipeline import run_pipeline_with_data, portfolio_stats
from vqportfolio.market_data.vanguard_calibration import nearest_lifestrategy_fund

st.set_page_config(page_title="Vanguard Quantum Portfolio Co-Pilot", layout="wide")


@st.cache_data(show_spinner="Loading market data...")
def load_market_data():
    prices, used_synthetic = load_prices()
    mu, sigma, log_returns = compute_returns_and_risk(prices)
    overlay = compute_cost_and_yield(TICKERS, log_returns)
    mdd = per_asset_max_drawdown(prices)
    return prices, used_synthetic, mu, sigma, overlay, mdd


prices, used_synthetic, mu, sigma, overlay, mdd = load_market_data()

st.title("Portfolio Co-Pilot")
st.caption("WISER 2026 — Vanguard Quantum for Finance — Multi-Asset Portfolio Construction")

if used_synthetic:
    st.warning(
        "Running on synthetic fallback price data — this environment has no internet access. "
        "Run locally or in Colab with internet access before treating any number here as real.",
        icon="⚠️",
    )

# ---------------------------------------------------------------- sidebar --
st.sidebar.header("Tunable Goals")
st.sidebar.caption("The four dials named in the brief. Growth/income/drawdown/cost.")
growth = st.sidebar.slider("Growth", 0.0, 3.0, 1.0, 0.1)
income = st.sidebar.slider("Income", 0.0, 3.0, 0.5, 0.1)
drawdown = st.sidebar.slider("Drawdown Control", 0.0, 3.0, 1.0, 0.1)
cost = st.sidebar.slider("Cost Sensitivity", 0.0, 3.0, 0.5, 0.1)

with st.sidebar.expander("Advanced"):
    st.caption(
        "The brief specifies exactly four dials, so 'drawdown control' internally "
        "splits between variance-aversion and historical-max-drawdown-aversion "
        "rather than exposing a separate fifth dial."
    )
    drawdown_variance_share = st.slider(
        "Drawdown dial: variance share", 0.0, 1.0, 0.5, 0.05,
        help="1.0 = drawdown dial acts like pure variance-aversion; "
             "0.0 = pure historical-max-drawdown-aversion.",
    )
    risk_aversion = st.slider(
        "Markowitz baseline risk aversion (λ)", 0.5, 12.0, 3.0, 0.5,
        help="Only affects the classical comparison baseline, not the dial-driven H/O/S scoring.",
    )

dials = Dials(
    growth=growth, income=income, drawdown=drawdown, cost=cost,
    drawdown_variance_share=drawdown_variance_share,
)
pconfig = PartitionConfig()

# ------------------------------------------------------- fast, live part --
scores_df = compute_asset_scores(mu, sigma, overlay["cost_bps"], overlay["yield"], mdd, dials)
partition = partition_assets(scores_df["score"], pconfig)
h_weights, o_budget = build_locked_allocation(mu, sigma, overlay["cost_bps"], partition, pconfig)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Asset Scores & Partition")
    st.caption("Updates instantly as dials move — this is the fast classical layer.")
    plot_df = scores_df.reset_index().rename(columns={"index": "ticker"})
    bucket_of = {t: "H (Hold)" for t in partition["H"]}
    bucket_of.update({t: "O (Optimize)" for t in partition["O"]})
    bucket_of.update({t: "S (Skip)" for t in partition["S"]})
    plot_df["Bucket"] = plot_df["ticker"].map(bucket_of)
    fig = px.bar(
        plot_df.sort_values("score", ascending=False), x="ticker", y="score", color="Bucket",
        color_discrete_map={"H (Hold)": "#2E7D32", "O (Optimize)": "#F9A825", "S (Skip)": "#C62828"},
    )
    fig.update_layout(height=350, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch')

with col2:
    st.subheader("Locked (H) Allocation")
    st.caption(f"{len(partition['H'])} assets, provably optimal for this membership "
               f"(convex QP, not a heuristic).")
    h_display = h_weights[h_weights > 0.001].sort_values(ascending=False)
    st.dataframe(h_display.map(lambda x: f"{x:.1%}"), width='stretch')
    st.metric("H budget used", f"{h_weights.sum():.1%}", help=f"Ceiling: {pconfig.h_budget_cap:.0%}")
    st.metric("Remaining O-set budget", f"{o_budget:.1%}")

st.divider()

# --------------------------------------------------- slow, gated quantum --
st.subheader("Quantum Optimization (O-set)")
st.caption(
    f"O-set: {', '.join(partition['O'])} ({len(partition['O'])} assets) — "
    "warm-start + multi-restart QAOA. Takes roughly 1-2 minutes; deliberately "
    "not auto-triggered on every dial move (see module docstring for why)."
)

run_key = (
    tuple(sorted(partition["O"])), round(o_budget, 4),
    growth, income, drawdown, cost, drawdown_variance_share, risk_aversion,
)

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
    st.session_state.last_run_key = None

if st.button("Run Quantum Optimization", type="primary"):
    with st.spinner("Solving the O-set QUBO via warm-start QAOA..."):
        result = run_pipeline_with_data(dials, pconfig, risk_aversion, mu, sigma, overlay, mdd)
        st.session_state.pipeline_result = result
        st.session_state.last_run_key = run_key

if st.session_state.pipeline_result is not None:
    result = st.session_state.pipeline_result

    if run_key != st.session_state.last_run_key:
        st.info("Dials have changed since this solve — click 'Run Quantum Optimization' again to refresh.")

    hos_stats = portfolio_stats(result.qaoa_full_weights, mu, sigma)
    mw = result.markowitz_result
    mw_equity_weight = float(
        mw["weights"][[t for t in mw["weights"].index if ASSET_CLASS_OF[t] == "Equities"]].sum()
    )
    mw_nearest = nearest_lifestrategy_fund(mw_equity_weight)

    if result.qaoa_matches_exact:
        st.success("QAOA matched the exact optimum on this O-set instance."
                    + (" (repair applied)" if result.repair_applied else ""))
    else:
        st.info("QAOA did not exactly match the brute-force optimum on this instance "
                "(expected at this problem size/depth — see docs/mps_scaling_findings.md)."
                + (" Repair applied." if result.repair_applied else ""))

    st.markdown("### Recommended Allocation vs. Classical Baseline")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Expected Return", f"{hos_stats['expected_return']:.2%}",
        delta=f"{(hos_stats['expected_return'] - mw['expected_return']):.2%} vs Markowitz",
    )
    c2.metric(
        "Risk (Variance)", f"{hos_stats['risk_variance']:.4f}",
        delta=f"{(hos_stats['risk_variance'] - mw['risk_variance']):.4f} vs Markowitz",
        delta_color="inverse",
    )
    c3.metric(
        "Resembles (real fund)", hos_stats["nearest_vanguard_fund"].ticker,
        help=f"{hos_stats['nearest_vanguard_fund'].name} "
             f"({hos_stats['nearest_vanguard_fund'].total_stock_pct:.0f}% stock) — "
             "Vanguard's own real LifeStrategy fund family, not an invented comparison.",
    )
    st.caption(
        f"Markowitz baseline: Sharpe {mw['sharpe_ratio']:.3f} "
        f"(rf={mw['risk_free_rate']:.2%}, "
        f"{'cached fallback' if mw['risk_free_rate_is_fallback'] else 'live FRED'}) "
        f"— resembles {mw_nearest.name} ({mw_nearest.ticker})"
    )

    comp_df = pd.DataFrame({
        "H/O/S + QAOA": result.qaoa_full_weights,
        "Markowitz": mw["weights"],
    })
    comp_df = comp_df[(comp_df.T != 0).any()].reset_index().rename(columns={"index": "ticker"})
    comp_long = comp_df.melt(id_vars="ticker", var_name="Method", value_name="Weight")
    fig2 = px.bar(comp_long, x="ticker", y="Weight", color="Method", barmode="group")
    fig2.update_layout(height=350, margin=dict(t=10, b=10), yaxis_tickformat=".0%")
    st.plotly_chart(fig2, width='stretch')

    st.markdown("### Why This Satisfies Constraints")
    guardrail_rows = []
    for ac, cap in ASSET_CLASS_CAPS.items():
        used = result.qaoa_full_weights[[t for t in TICKERS if ASSET_CLASS_OF[t] == ac]].sum()
        guardrail_rows.append({"Asset Class": ac, "Used": used, "Cap": cap, "Headroom": cap - used})
    guardrail_df = pd.DataFrame(guardrail_rows).set_index("Asset Class")
    st.dataframe(
        guardrail_df.style.format({"Used": "{:.1%}", "Cap": "{:.1%}", "Headroom": "{:.1%}"}),
        width='stretch',
    )

    if hos_stats["guardrail_breaches"]:
        st.error(f"Guardrail breaches: {hos_stats['guardrail_breaches']}")
    else:
        st.success("Zero guardrail breaches — every asset-class cap is a hard constraint the solver enforces.")

    st.markdown("### H / O / S Rationale")
    st.markdown(
        f"- **H — Hold ({len(partition['H'])} assets):** top-scoring under the current dial "
        f"settings, locked at a classically-optimized weight (a real convex QP solved for this "
        f"exact membership, not a proportional-to-score heuristic) up to {pconfig.h_budget_cap:.0%} "
        f"of the portfolio.\n"
        f"- **O — Optimize ({len(partition['O'])} assets):** the contested middle — assets whose "
        f"ranking is genuinely ambiguous given the current dials. This is the only part of the "
        f"portfolio the quantum solver touches.\n"
        f"- **S — Skip ({len(partition['S'])} assets):** lowest-scoring under the current dial "
        f"settings, excluded entirely (weight = 0)."
    )
else:
    st.info("Click **Run Quantum Optimization** above to solve the O-set and see the full comparison.")

st.divider()
st.caption(
    "Source: [github repo link] — see README.md and docs/mps_scaling_findings.md for full "
    "methodology, data provenance, and known limitations."
)
