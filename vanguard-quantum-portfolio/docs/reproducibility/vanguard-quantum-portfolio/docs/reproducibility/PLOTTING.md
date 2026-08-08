# Plotting validation command output

Use `vqportfolio.validation.plot_results` to turn validation-run logs into
slide-ready self-contained HTML/SVG charts and companion CSVs as soon as a
command, or a batch of commands, finishes.

Install the quantum extra for the validation runners. The plotting helper itself
uses only pandas plus the Python standard library and writes self-contained
HTML/SVG charts.

```bash
pip install -e ".[quantum]"
```

## One command -> plots immediately

```bash
python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 0 3 \
  2>&1 | tee docs/reproducibility/real_walk_forward.log
python -m vqportfolio.validation.plot_results docs/reproducibility/real_walk_forward.log \
  --out docs/plots
```

## Many commands -> one combined plot set

Append compatible logs, then plot all of them together:

```bash
python -m vqportfolio.validation.run_multi_seed_sweep 0 5 1 15 5 256 \
  2>&1 | tee docs/reproducibility/real_seed_sweep.log
python -m vqportfolio.validation.run_ablation_at_size 5 2 60 0 1 \
  2>&1 | tee docs/reproducibility/real_mps_size5_p1.log
python -m vqportfolio.validation.run_ablation_at_size 5 2 60 0 2 \
  2>&1 | tee docs/reproducibility/real_mps_size5_p2.log

python -m vqportfolio.validation.plot_results docs/reproducibility/real_*.log \
  --out docs/plots
```

The helper currently recognizes:

- `PERIOD_RESULT` lines from `run_walk_forward`
- `SEED_RESULT` lines from `run_multi_seed_sweep`
- `SIZE_RESULT` plus summary-table rows from `run_ablation_at_size`
- fixed-width method rows from `run_classical_benchmarks`

## Outputs

Depending on what the logs contain, the helper writes:

- `walk_forward_periods.csv`
- `walk_forward_returns.html`
- `walk_forward_cumulative.html`
- `qaoa_seed_results.csv`
- `qaoa_seed_objective.html`
- `qaoa_seed_gap.html`
- `mps_scaling_results.csv`
- `mps_runtime_by_qubits.html`
- `mps_gap_by_qubits.html`
- `classical_benchmarks.csv`
- `classical_benchmark_gap.html`
- `classical_benchmark_time.html`

Open the HTML files in a browser and screenshot them for the slide deck, or
use the CSV files if you want to recreate the charts in PowerPoint/Google
Slides.
