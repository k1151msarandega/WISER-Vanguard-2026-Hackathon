# Reproducing the Walk-Forward and Multi-Seed Variance Findings

This documents the *exact* commands run to produce
`docs/walk_forward_findings.md` and `docs/multi_seed_variance_findings.md`,
plus the aggregation code used to turn raw per-period/per-seed output into
the summary tables in those docs. Raw and noise-filtered logs from the
actual runs are alongside this file (`*_raw.log` includes every stderr line
including the no-internet-sandbox yfinance/HF-download noise; `*_clean.log`
strips that noise and keeps only the PERIOD_RESULT/SEED_RESULT/summary
lines that actually matter).

**Why one period/seed per command**: each individual solve took anywhere
from ~8s to ~140s depending on how many total qubits (objective + slack)
that particular instance needed -- see `walk_forward_findings.md`'s
"qubit count is not determined by O-set size alone" finding for why this
varies so much. Running the full sweep as a single long-lived background
process wasn't viable in the dev environment these were run in (no
persistence across tool-call boundaries), hence the one-call-per-unit
pattern below. This has no bearing on reproducing the results elsewhere --
if you have a normal persistent shell, the runner scripts below can be
looped directly.

## Walk-forward (14 annual-step periods)

Setup (once): `pip install -e ".[quantum]"` from repo root.

Ran sequentially, `start_period` incrementing 0 through 13:

```bash
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 0 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 1 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 2 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 3 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 4 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 5 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 6 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 7 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 8 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 9 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 10 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 11 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 12 1
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 13 1
```

Argument order: `train_days test_days max_o_size n_restarts maxiter shots
start_period max_periods`. If you have a persistent shell, this is
equivalent to one call:

```bash
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 0 14
```

**Aggregation code** (used to produce the summary table and stats in the
findings doc, run against the concatenated `PERIOD_RESULT` lines):

```python
import re
import numpy as np

lines = [l for l in open('walk_forward_clean.log') if l.startswith('PERIOD_RESULT')]
hos, mw, matches, repairs = [], [], [], []
for l in lines:
    hos.append(float(re.search(r'hos_return=([-\d.]+)', l).group(1)))
    mw.append(float(re.search(r'mw_return=([-\d.]+)', l).group(1)))
    matches.append(re.search(r'matches_exact=(\w+)', l).group(1) == 'True')
    repairs.append(re.search(r'repair=(\w+)', l).group(1) == 'True')

hos, mw = np.array(hos), np.array(mw)
print(f'n_periods = {len(hos)}')
print(f'HOS+QAOA:  mean={hos.mean():.4f}  std={hos.std():.4f}  cumulative={np.prod(1+hos)-1:.4f}')
print(f'Markowitz: mean={mw.mean():.4f}  std={mw.std():.4f}  cumulative={np.prod(1+mw)-1:.4f}')
print(f'HOS beats MW: {sum(hos>mw)}/{len(hos)} periods ({sum(hos>mw)/len(hos):.1%})')
print(f'QAOA matched exact optimum: {sum(matches)}/{len(matches)} periods')
print(f'Repair needed: {sum(repairs)}/{len(repairs)} periods')
```

## Multi-seed variance at 18 qubits (seeds 0-9)

```bash
python -m vqportfolio.validation.run_multi_seed_sweep 0 1 1 15 5 256
python -m vqportfolio.validation.run_multi_seed_sweep 1 2 1 15 5 256
python -m vqportfolio.validation.run_multi_seed_sweep 2 3 1 15 5 256
python -m vqportfolio.validation.run_multi_seed_sweep 3 4 1 15 5 256
python -m vqportfolio.validation.run_multi_seed_sweep 4 5 1 15 5 256
python -m vqportfolio.validation.run_multi_seed_sweep 5 6 1 15 5 256
python -m vqportfolio.validation.run_multi_seed_sweep 6 7 1 15 5 256
python -m vqportfolio.validation.run_multi_seed_sweep 7 8 1 15 5 256
python -m vqportfolio.validation.run_multi_seed_sweep 8 9 1 15 5 256
python -m vqportfolio.validation.run_multi_seed_sweep 9 10 1 15 5 256
```

Argument order: `seed_start seed_end n_restarts maxiter max_o_size shots`.
Equivalent single call with a persistent shell:

```bash
python -m vqportfolio.validation.run_multi_seed_sweep 0 10 1 15 5 256
```

**Aggregation code:**

```python
import re
import numpy as np

lines = [l for l in open('multi_seed_variance_clean.log') if l.startswith('SEED_RESULT')]
objs, gaps, matches, repairs = [], [], [], []
for l in lines:
    objs.append(float(re.search(r'objective=([-\d.]+)', l).group(1)))
    gaps.append(float(re.search(r'gap=([-\d.]+)', l).group(1)))
    matches.append(re.search(r'matches_exact=(\w+)', l).group(1) == 'True')
    repairs.append(re.search(r'repair_applied=(\w+)', l).group(1) == 'True')

objs, gaps = np.array(objs), np.array(gaps)
print(f'n_seeds = {len(objs)}')
print(f'mean_objective = {objs.mean():.6f}')
print(f'std_objective = {objs.std():.6f}')
print(f'mean_gap_to_exact = {gaps.mean():.6f}')
print(f'max_gap_to_exact = {gaps.max():.6f}')
print(f'n_matches_exact = {sum(matches)}/{len(matches)}')
print(f'n_repair_applied = {sum(repairs)}/{len(repairs)}')
```

## Other verification performed but not part of either findings doc directly

These were sanity/debugging steps run during development, referenced in
the findings docs but not themselves producing a results table -- included
here for full transparency on what was actually checked before trusting
the headline numbers:

1. **Realized-return function correctness** (referenced in
   `walk_forward_findings.md`'s "Sanity checks" section): verified
   `_fixed_weight_realized_return()` against a trivial single-asset case
   (weight=1 on one ticker, 0 elsewhere) exactly reproduces that ticker's
   simple buy-and-hold return over the same window, to machine precision
   (`0.182425` both ways). Confirms no compounding/indexing bug in the
   walk-forward return calculation before trusting it on the real
   multi-asset case.
2. **Qubit-count survey** (the "not determined by O-set size alone"
   finding): ran the QUBO-build step only (no solve) across all 56
   candidate quarterly periods at fixed `max_o_size=4`, and separately
   across the 14 candidate annual periods actually used, confirming total
   qubit count varied 12-21 across periods despite constant O-set size.
   This is a cheap, non-stochastic check (same QUBO build code the actual
   pipeline uses, just stopped before the QAOA solve step) -- reran twice
   to confirm the counts were deterministic given the same trailing window,
   not an artifact of anything stochastic in the build step itself.
3. **Bug verification for the app.py `ticker` KeyError fix**: unit-tested
   the fix in isolation against a synthetic named-index case (mimicking
   yfinance's real column-index naming behavior) before considering it
   resolved, rather than relying only on the report that it worked in
   Colab.
