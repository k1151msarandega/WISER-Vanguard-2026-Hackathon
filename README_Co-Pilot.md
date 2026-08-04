# Portfolio Co-Pilot Demo

The interactive artifact required by the brief: shows the recommended
allocation, the trade-offs versus the classical baseline, and why the
solution satisfies constraints.

## Run it

```bash
pip install -e ".[app]"
streamlit run app/app.py
```

## Design

Two-speed interaction, deliberately, not as a workaround:

- **Dials (growth/income/drawdown/cost) update instantly.** Moving a slider
  recomputes asset scores, the H/O/S partition, and H's locked allocation
  live — all fast (<1s: a handful of z-scores plus a small convex QP).
- **The quantum optimization step is gated behind an explicit button.**
  Warm-start + multi-restart QAOA on the O-set takes ~1-2 minutes; it is not
  auto-triggered on every slider drag. This mirrors the actual H/O/S
  architecture — H is the fast classical layer, O is where real quantum
  computation happens — rather than hiding that distinction from the user.

## What it shows

- Live-updating, color-coded (H/O/S) asset score chart as dials move
- H's locked allocation (a real convex-optimized result, not a heuristic)
- On-demand: the O-set's QAOA solution, validated against exact brute force
  when the instance is small enough
- Side-by-side comparison against the classical Markowitz baseline: return,
  risk, Sharpe ratio (real risk-free rate via FRED)
- "Resembles Vanguard's [real fund]" — every portfolio (ours and the
  baseline) is matched against Vanguard's actual published LifeStrategy fund
  family, not an invented comparison
- Explicit guardrail compliance table (used % vs. cap % per asset class)
- Plain-language H/O/S rationale

## Tested

Validated via `streamlit.testing.v1.AppTest` (not just "it imports cleanly"):
initial load, slider interaction, and a full click of the "Run Quantum
Optimization" button — all confirmed exception-free before shipping. See
`vqportfolio.pipeline.run_pipeline_with_data()`, which the app uses instead
of `run_pipeline()` specifically to avoid redundant market-data reloads on
every button click.

## Known limitation

Same caveat as the rest of the project: real data sourcing works when run
with internet access (this repo's dev sandbox has none), and every result
shown falls back to clearly-labeled synthetic data otherwise. Check the
warning banner at the top of the app before trusting any number.
