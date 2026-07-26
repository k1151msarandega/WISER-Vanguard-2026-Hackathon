# Vanguard Quantum for Finance — Multi-Asset Portfolio Construction

Solo submission, WISER Global Quantum+AI Program 2026. Deadline: **Aug 7, 2026**.

## Repo layout

```
vanguard-quantum-portfolio/
├── pyproject.toml          # pip install -e .
├── src/vqportfolio/        # the actual package
│   ├── config.py           # asset universe, sector caps, constants
│   ├── market_data/        # loader (yfinance + synthetic fallback), overlays (returns/vol/cost/yield)
│   ├── baseline/           # classical Markowitz baseline
│   ├── partitioning/       # Week 2: H/O/S logic
│   ├── quantum/            # Week 2: QUBO/QAOA
│   └── validation/         # Week 3: rigor layer (multi-seed, walk-forward, scaling)
├── notebooks/              # what you open in Colab
├── app/                    # Week 3-4: portfolio co-pilot demo
├── tests/
└── data/                   # local cache only, gitignored
```

## Workflow: GitHub + Colab

1. **Push this repo to GitHub** as-is — everything under `data/raw/` and
   `data/processed/` is gitignored, so no market data or generated results get
   committed by accident.
2. **In Colab**, each notebook starts with:
   ```python
   !git clone https://github.com/YOUR_USERNAME/vanguard-quantum-portfolio.git
   %cd vanguard-quantum-portfolio
   !pip install -e . -q
   ```
   This makes `vqportfolio` importable exactly like any installed package —
   no path hacks, no `sys.path.append`.
3. **Locally**, same install command: `pip install -e .` from the repo root,
   then `from vqportfolio... import ...` works from anywhere.

## Critical caveat: real vs. synthetic data

Every data-touching function returns or prints a `used_synthetic` /
`USED_SYNTHETIC_PRICES` flag. **Before trusting any number for the actual
writeup, confirm this is `False`.** It will be `True` if run somewhere without
internet access to Yahoo Finance (e.g. a sandboxed dev environment) — the
fallback exists so the pipeline can be built and tested without blocking on
network access, not as a data source for real results.

| Field | Source |
|---|---|
| Prices / returns / volatility / correlation | Real (yfinance) when internet is available; synthetic GBM fallback otherwise |
| Transaction cost proxy (`cost_bps`) | Synthetic — liquidity-tier spread scaled by realized vol |
| Yield (`yield`) | Synthetic — asset-class-appropriate band, feeds the income dial |
| Sector/asset-class labels | Manually assigned static reference data |

## Status

- [x] **Week 1** (Jul 14–20): data pipeline + classical Markowitz baseline,
      validated (`tests/test_baseline_sanity.py` — monotonic risk vs. risk
      aversion, zero guardrail breaches, no degenerate concentration)
- [ ] **Week 2** (Jul 21–27): H/O/S partitioning reworked for continuous
      binarized weights + growth/income/drawdown/cost dials; QUBO → Ising
      conversion; first QAOA prototype (Qiskit, small O-set, p=1)
- [ ] **Week 3** (Jul 28–Aug 3): equal-footing classical benchmarks, multi-seed
      variance reporting, walk-forward validation, scaling analysis, co-pilot
      demo skeleton
- [ ] **Week 4** (Aug 4–7): buffer, writeup, packaging

## Guardrails enforced (hard constraints in the classical baseline)

- Per-asset weight cap: 25%
- Asset-class exposure caps: Equities 60%, Fixed Income 55%, Commodities 20%,
  Currencies 15%, Alternatives 20%
- Turnover cap (configurable)

## Install

```bash
pip install -e .              # core
pip install -e ".[quantum]"   # + qiskit, for Week 2
pip install -e ".[app]"       # + streamlit, for Week 3-4
```
