# Submission Responses

## Project Title

Quantum-Assisted Portfolio Co-Pilot for Guardrail-Constrained Multi-Asset Allocation

## Project Summary

Our team addressed the challenge of building a multi-asset portfolio that balances growth, income, drawdown control, and implementation cost while satisfying hard investment guardrails. We propose a portfolio co-pilot that separates the universe into Hold/Optimize/Skip (H/O/S) buckets: high-conviction assets are allocated by a constrained convex optimizer, low-conviction assets receive zero weight, and the ambiguous O-set is encoded as a discretized QUBO solved by warm-start, multi-restart QAOA and checked against exact brute force when tractable.

The methods combine real or synthetic market-data loading, return/covariance estimation, Corwin-Schultz transaction-cost proxies, yield and liquidity overlays, Vanguard LifeStrategy-calibrated stock/bond guardrails, Markowitz baselines, equal-footing classical benchmarks, multi-seed QAOA variance analysis, MPS scaling tests, and walk-forward validation. The Streamlit demo exposes the four investor dials, shows the recommended allocation versus Markowitz, explains H/O/S membership, and reports guardrail compliance.

Primary findings: the architecture maintained zero asset-class guardrail breaches in the tested walk-forward run, while H/O/S+QAOA compounded 44.8% versus 21.5% for Markowitz on 14 annual synthetic out-of-sample periods. QAOA matched exact brute force on smaller O-set instances but showed real seed variance at 18 qubits. MPS simulation can help with qubit count, but its benefit depends on covariance/entanglement structure and erodes as QAOA depth increases.

## Proposed Solution and Differentiators

The solution is a decision-support co-pilot rather than a black-box optimizer. Users adjust four dials from the prompt -- growth, income, drawdown control, and cost sensitivity -- and immediately see how assets move across H/O/S buckets. The fast H layer uses a real mean-variance-cost convex program, so locked weights are not a score-proportional heuristic. The O layer converts only the contested middle of the universe into a QUBO, which keeps the quantum problem smaller and focuses quantum compute where classical ranking is least decisive. The S layer makes exclusions explicit and explainable.

What stands out is the combination of explainability, guardrails, and validation. The app compares the final H/O/S+QAOA allocation against a full-universe constrained Markowitz baseline, maps portfolios to Vanguard's published LifeStrategy stock/bond profiles for intuitive context, and displays asset-class cap usage and headroom. The research code also includes equal-footing benchmarks that score QAOA, exact brute force, diagonal-risk ILP, greedy, and random feasible solutions on the same discretized O-set objective. The quantum solver uses warm-started QAOA with multiple restarts, then repairs and validates results against constraints rather than assuming sampled bitstrings are feasible. Finally, the project documents negative and nuanced findings -- seed variance, MPS/depth trade-offs, qubit-count surprises from slack variables, and unresolved repair-step discrepancies -- instead of reporting only favorable snapshots.

## Limitations Encountered

The largest limitation was data and execution environment. The code supports live yfinance, FRED, and Hugging Face/DuckDB OHLCV paths, but the development sandbox often required synthetic fallback data; therefore reported performance numbers should be treated as synthetic-data evidence unless rerun with live access. Yield data is based on current trailing yield rather than point-in-time historical yield, and walk-forward cost/yield overlays were held fixed to isolate rolling return/covariance effects.

Compute time also constrained validation. Practical QAOA settings can take one to two minutes per solve, and larger walk-forward periods reached 15-21 total objective-plus-slack qubits even with a fixed four-asset O-set. To finish the walk-forward and variance studies, some runs used reduced QAOA settings: fewer restarts, fewer optimizer iterations, and fewer shots than the default pipeline. Multi-seed testing confirmed that reliability is instance- and size-dependent, especially at 18 qubits.

Methodological limitations remain. The asset universe is intentionally small for QAOA simulation; O-set weights are discretized; H/O/S scoring uses per-asset marginal risk/drawdown features before the covariance-aware allocation stage; O-set costs assume a cold-start allocation rather than turnover from prior holdings; non-stock/bond guardrail caps are reasoned design choices rather than Vanguard-calibrated policy values; and one benchmark instance showed an unresolved gap between an exact-matching QAOA bitstring and repaired weight-space objective.

## Future Testing, Improvement, Expansion, and Implementation

Future work should first rerun the full pipeline with live market data and preserve per-ticker source flags, then repeat the walk-forward analysis with point-in-time cost, liquidity, yield, and turnover estimates. A stronger validation plan would include more O-set instances, more seeds per instance, production QAOA settings, confidence intervals, transaction-cost-adjusted realized returns, turnover constraints, and comparisons to stronger classical MIQP or nonlinear solvers where available.

The quantum side should be expanded by moving the optimization-loop cost evaluation beyond statevector-only simulation, because the current architecture hits a hard width ceiling around 27 qubits. MPS should be tested as part of the optimizer, not only final expectation/sampling comparisons, with depth and restart budgets scaled together so that p=2 or p=3 quality claims are not confounded by insufficient classical optimization. Additional experiments should measure how slack variables, class caps, and O-set size jointly drive qubit count.

For implementation, the Streamlit co-pilot can become an advisor workflow: ingest approved ETF/fund universes, cache audited market data, let an advisor set client preferences with the four dials, produce a proposed allocation and explanation, and require human approval before execution. Productionization would need compliance review, model-risk documentation, monitoring for guardrail breaches and data-source drift, richer sector/liquidity constraints, authentication, reproducible run logs, and an AI-assistance disclosure.
