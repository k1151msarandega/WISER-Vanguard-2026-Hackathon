"""Plot validation command logs into slide-ready interactive HTML charts.

Usage examples:

  python -m vqportfolio.validation.run_walk_forward 756 252 4 1 10 128 0 3 \
    2>&1 | tee docs/reproducibility/real_walk_forward.log
  python -m vqportfolio.validation.plot_results docs/reproducibility/real_walk_forward.log \
    --out docs/plots

  python -m vqportfolio.validation.plot_results docs/reproducibility/*.log --out docs/plots

The parser is intentionally based on the stable, machine-readable lines the
validation runners already print: PERIOD_RESULT, SEED_RESULT, SIZE_RESULT,
and the fixed-width benchmark tables. It writes both CSV data and self-contained HTML/SVG
so figures can be dropped into a slide deck or screenshotted immediately after
a command or batch finishes.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


_PERIOD_RE = re.compile(
    r"PERIOD_RESULT idx=(?P<idx>\d+).*?"
    r"O=(?P<o_set>\[.*?\]) "
    r"hos_return=(?P<hos_return>[-\d.]+) "
    r"mw_return=(?P<mw_return>[-\d.]+) "
    r"matches_exact=(?P<matches_exact>\w+) "
    r"repair=(?P<repair>\w+) "
    r"hos_breach=(?P<hos_breach>\w+) "
    r"mw_breach=(?P<mw_breach>\w+) "
    r"ok=(?P<ok>\w+)"
)
_SEED_RE = re.compile(
    r"SEED_RESULT seed=(?P<seed>\d+) "
    r"objective=(?P<objective>[-\d.]+) "
    r"gap=(?P<gap>[-\d.]+) "
    r"matches_exact=(?P<matches_exact>\w+) "
    r"repair_applied=(?P<repair_applied>\w+) "
    r"ok=(?P<ok>\w+)"
)
_SIZE_RE = re.compile(r"SIZE_RESULT max_o_size=(?P<max_o_size>\d+) n_qubits=(?P<n_qubits>\d+)")
_CLASSICAL_ROW_RE = re.compile(
    r"^(?P<method>[A-Za-z].*?)\s+"
    r"(?P<objective>-?\d+\.\d+)\s+"
    r"(?P<gap>-?\d+\.\d+)\s+"
    r"(?P<time>\d+\.\d+)\s*$"
)
_SCALING_ROW_RE = re.compile(
    r"^(?P<backend>statevector|mps)\s+"
    r"(?P<max_bond_dim>uncapped|\d+)\s+"
    r"(?P<n_qubits>\d+)\s+"
    r"(?P<runtime_s>inf|[-\d.]+)\s+"
    r"(?P<objective>inf|[-\d.]+)\s+"
    r"(?P<gap_to_exact>None|nan|inf|[-\d.]+)\s*$"
)


def _as_bool(value: str) -> bool:
    return value == "True"


def _read_lines(paths: list[Path]) -> list[str]:
    lines: list[str] = []
    for path in paths:
        lines.extend(path.read_text().splitlines())
    return lines


def parse_walk_forward(lines: list[str]) -> pd.DataFrame:
    rows = []
    for line in lines:
        match = _PERIOD_RE.search(line)
        if not match:
            continue
        row = match.groupdict()
        rows.append({
            "idx": int(row["idx"]),
            "o_set": row["o_set"],
            "hos_return": float(row["hos_return"]),
            "mw_return": float(row["mw_return"]),
            "matches_exact": _as_bool(row["matches_exact"]),
            "repair": _as_bool(row["repair"]),
            "hos_breach": _as_bool(row["hos_breach"]),
            "mw_breach": _as_bool(row["mw_breach"]),
            "ok": _as_bool(row["ok"]),
        })
    return pd.DataFrame(rows).sort_values("idx") if rows else pd.DataFrame()


def parse_seed_results(lines: list[str]) -> pd.DataFrame:
    rows = []
    for line in lines:
        match = _SEED_RE.search(line)
        if not match:
            continue
        row = match.groupdict()
        rows.append({
            "seed": int(row["seed"]),
            "objective": float(row["objective"]),
            "gap_to_exact": float(row["gap"]),
            "matches_exact": _as_bool(row["matches_exact"]),
            "repair_applied": _as_bool(row["repair_applied"]),
            "ok": _as_bool(row["ok"]),
        })
    return pd.DataFrame(rows).sort_values("seed") if rows else pd.DataFrame()


def parse_scaling_results(lines: list[str]) -> pd.DataFrame:
    rows = []
    current_size: int | None = None
    for line in lines:
        size_match = _SIZE_RE.search(line)
        if size_match:
            current_size = int(size_match.group("max_o_size"))
            continue
        row_match = _SCALING_ROW_RE.search(line.strip())
        if not row_match or current_size is None:
            continue
        row = row_match.groupdict()
        rows.append({
            "max_o_size": current_size,
            "backend": row["backend"],
            "max_bond_dim": row["max_bond_dim"],
            "n_qubits": int(row["n_qubits"]),
            "runtime_s": float(row["runtime_s"]),
            "objective": float(row["objective"]),
            "gap_to_exact": None if row["gap_to_exact"] in {"None", "nan"} else float(row["gap_to_exact"]),
        })
    return pd.DataFrame(rows)


def parse_classical_benchmarks(lines: list[str]) -> pd.DataFrame:
    rows = []
    in_table = False
    for line in lines:
        if "Method" in line and "Objective" in line and "Gap to exact" in line:
            in_table = True
            continue
        if not in_table:
            continue
        match = _CLASSICAL_ROW_RE.match(line)
        if not match:
            if rows:
                in_table = False
            continue
        row = match.groupdict()
        rows.append({
            "method": row["method"].strip(),
            "objective": float(row["objective"]),
            "gap_to_exact": float(row["gap"]),
            "time_s": float(row["time"]),
        })
    return pd.DataFrame(rows)


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False)


def _html_escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _write_html(path: Path, title: str, body: str) -> None:
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{_html_escape(title)}</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;color:#222}"
        ".chart-title{font-size:22px;font-weight:700;margin-bottom:8px}"
        ".note{color:#666;font-size:13px;margin-top:8px}"
        "svg{max-width:100%;height:auto;border:1px solid #ddd;background:#fff}"
        "text{font-family:Arial,sans-serif}</style></head><body>"
        f"<div class='chart-title'>{_html_escape(title)}</div>{body}</body></html>"
    )


def _nice_domain(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    lo, hi = min(values), max(values)
    if lo == hi:
        pad = abs(lo) * 0.1 or 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * 0.12
    return lo - pad, hi + pad


def _bar_svg(labels: list[str], series: dict[str, list[float]], title: str, y_format: str = "{:.3f}") -> str:
    width, height = 960, 520
    left, right, top, bottom = 80, 30, 35, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [v for vals in series.values() for v in vals]
    y_min, y_max = _nice_domain(all_values + [0.0])

    def x_pos(group_idx: int, item_idx: int, n_items: int) -> tuple[float, float]:
        group_w = plot_w / max(len(labels), 1)
        pad = group_w * 0.18
        bar_w = (group_w - 2 * pad) / max(n_items, 1)
        return left + group_idx * group_w + pad + item_idx * bar_w, bar_w * 0.86

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = ["#2E7D32", "#1565C0", "#F9A825", "#C62828", "#6A1B9A", "#00838F"]
    zero_y = y_pos(0.0)
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{_html_escape(title)}'>"]
    parts.append(f"<line x1='{left}' y1='{zero_y:.1f}' x2='{width-right}' y2='{zero_y:.1f}' stroke='#777' stroke-width='1'/>")
    parts.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' stroke='#444'/>")
    parts.append(f"<line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' stroke='#444'/>")
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = y_pos(value)
        parts.append(f"<line x1='{left-4}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#eee'/>")
        parts.append(f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' font-size='12'>{_html_escape(y_format.format(value))}</text>")
    n_series = len(series)
    for s_idx, (name, values) in enumerate(series.items()):
        color = colors[s_idx % len(colors)]
        for i, value in enumerate(values):
            x, bar_w = x_pos(i, s_idx, n_series)
            y = y_pos(max(value, 0.0))
            h = abs(y_pos(value) - zero_y)
            if value < 0:
                y = zero_y
            parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{h:.1f}' fill='{color}'><title>{_html_escape(labels[i])} {name}: {y_format.format(value)}</title></rect>")
    group_w = plot_w / max(len(labels), 1)
    for i, label in enumerate(labels):
        x = left + i * group_w + group_w / 2
        parts.append(f"<text x='{x:.1f}' y='{height-bottom+20}' text-anchor='middle' font-size='12' transform='rotate(-30 {x:.1f},{height-bottom+20})'>{_html_escape(label)}</text>")
    legend_x = left
    for s_idx, name in enumerate(series):
        x = legend_x + s_idx * 175
        color = colors[s_idx % len(colors)]
        parts.append(f"<rect x='{x}' y='{height-30}' width='14' height='14' fill='{color}'/>")
        parts.append(f"<text x='{x+20}' y='{height-18}' font-size='13'>{_html_escape(name)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _line_svg(labels: list[str], series: dict[str, list[float]], title: str, y_format: str = "{:.3f}") -> str:
    width, height = 960, 520
    left, right, top, bottom = 80, 30, 35, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [v for vals in series.values() for v in vals]
    y_min, y_max = _nice_domain(all_values + [0.0])

    def x_pos(i: int) -> float:
        return left + (plot_w * i / max(len(labels) - 1, 1))

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = ["#2E7D32", "#1565C0", "#F9A825", "#C62828", "#6A1B9A", "#00838F"]
    parts = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{_html_escape(title)}'>"]
    parts.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' stroke='#444'/>")
    parts.append(f"<line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' stroke='#444'/>")
    for tick in range(5):
        value = y_min + (y_max - y_min) * tick / 4
        y = y_pos(value)
        parts.append(f"<line x1='{left-4}' y1='{y:.1f}' x2='{width-right}' y2='{y:.1f}' stroke='#eee'/>")
        parts.append(f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' font-size='12'>{_html_escape(y_format.format(value))}</text>")
    for s_idx, (name, values) in enumerate(series.items()):
        color = colors[s_idx % len(colors)]
        points = " ".join(f"{x_pos(i):.1f},{y_pos(v):.1f}" for i, v in enumerate(values))
        parts.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='3'/>")
        for i, value in enumerate(values):
            x, y = x_pos(i), y_pos(value)
            parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}'><title>{_html_escape(labels[i])} {name}: {y_format.format(value)}</title></circle>")
    for i, label in enumerate(labels):
        x = x_pos(i)
        parts.append(f"<text x='{x:.1f}' y='{height-bottom+22}' text-anchor='middle' font-size='12'>{_html_escape(label)}</text>")
    for s_idx, name in enumerate(series):
        x = left + s_idx * 175
        color = colors[s_idx % len(colors)]
        parts.append(f"<rect x='{x}' y='{height-30}' width='14' height='14' fill='{color}'/>")
        parts.append(f"<text x='{x+20}' y='{height-18}' font-size='13'>{_html_escape(name)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def plot_walk_forward(df: pd.DataFrame, out_dir: Path) -> None:
    _write_frame(df, out_dir / "walk_forward_periods.csv")
    if df.empty:
        return

    labels = [str(i) for i in df["idx"]]
    _write_html(
        out_dir / "walk_forward_returns.html",
        "Walk-forward period returns",
        _bar_svg(labels, {"H/O/S + QAOA": list(df["hos_return"]), "Markowitz": list(df["mw_return"])},
                 "Walk-forward period returns", "{:.1%}"),
    )

    hos_cum = list((1 + df["hos_return"]).cumprod() - 1)
    mw_cum = list((1 + df["mw_return"]).cumprod() - 1)
    _write_html(
        out_dir / "walk_forward_cumulative.html",
        "Walk-forward cumulative return",
        _line_svg(labels, {"H/O/S + QAOA": hos_cum, "Markowitz": mw_cum},
                  "Walk-forward cumulative return", "{:.1%}"),
    )


def plot_seed_results(df: pd.DataFrame, out_dir: Path) -> None:
    _write_frame(df, out_dir / "qaoa_seed_results.csv")
    if df.empty:
        return

    labels = [str(s) for s in df["seed"]]
    _write_html(
        out_dir / "qaoa_seed_objective.html",
        "QAOA objective by seed",
        _bar_svg(labels, {"Objective": list(df["objective"])}, "QAOA objective by seed", "{:.6f}"),
    )
    _write_html(
        out_dir / "qaoa_seed_gap.html",
        "QAOA gap to exact by seed",
        _bar_svg(labels, {"Gap to exact": list(df["gap_to_exact"])}, "QAOA gap to exact by seed", "{:.6f}"),
    )


def plot_scaling_results(df: pd.DataFrame, out_dir: Path) -> None:
    _write_frame(df, out_dir / "mps_scaling_results.csv")
    if df.empty:
        return

    labeled = df.copy()
    labeled["backend_label"] = labeled.apply(
        lambda r: "statevector" if r["backend"] == "statevector" else f"MPS bond={r['max_bond_dim']}", axis=1,
    )
    labels = [str(q) for q in sorted(labeled["n_qubits"].unique())]
    runtime_series = {}
    for backend_label, group in labeled.groupby("backend_label"):
        values = []
        by_qubit = dict(zip(group["n_qubits"], group["runtime_s"]))
        for label in labels:
            values.append(by_qubit.get(int(label), 0.0))
        runtime_series[backend_label] = values
    _write_html(
        out_dir / "mps_runtime_by_qubits.html",
        "Backend runtime by qubit count",
        _line_svg(labels, runtime_series, "Backend runtime by qubit count", "{:.3f}"),
    )

    gap_df = labeled.dropna(subset=["gap_to_exact"])
    if not gap_df.empty:
        gap_series = {}
        for backend_label, group in gap_df.groupby("backend_label"):
            values = []
            by_qubit = dict(zip(group["n_qubits"], group["gap_to_exact"]))
            for label in labels:
                values.append(by_qubit.get(int(label), 0.0))
            gap_series[backend_label] = values
        _write_html(
            out_dir / "mps_gap_by_qubits.html",
            "Backend gap to exact by qubit count",
            _line_svg(labels, gap_series, "Backend gap to exact by qubit count", "{:.6f}"),
        )


def plot_classical_benchmarks(df: pd.DataFrame, out_dir: Path) -> None:
    _write_frame(df, out_dir / "classical_benchmarks.csv")
    if df.empty:
        return

    labels = list(df["method"])
    _write_html(
        out_dir / "classical_benchmark_gap.html",
        "O-set benchmark gap to exact",
        _bar_svg(labels, {"Gap to exact": list(df["gap_to_exact"])}, "O-set benchmark gap to exact", "{:.6f}"),
    )
    _write_html(
        out_dir / "classical_benchmark_time.html",
        "O-set benchmark solve time",
        _bar_svg(labels, {"Time (s)": list(df["time_s"])}, "O-set benchmark solve time", "{:.3f}"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot validation runner logs into HTML charts and CSVs.")
    parser.add_argument("logs", nargs="+", type=Path, help="Log files produced by validation commands")
    parser.add_argument("--out", type=Path, default=Path("docs/plots"), help="Output directory for charts/CSVs")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    lines = _read_lines(args.logs)

    walk = parse_walk_forward(lines)
    seeds = parse_seed_results(lines)
    scaling = parse_scaling_results(lines)
    classical = parse_classical_benchmarks(lines)

    plot_walk_forward(walk, args.out)
    plot_seed_results(seeds, args.out)
    plot_scaling_results(scaling, args.out)
    plot_classical_benchmarks(classical, args.out)

    print(f"Wrote plots/CSVs to {args.out}")
    print(f"  walk_forward_periods={len(walk)}")
    print(f"  qaoa_seed_results={len(seeds)}")
    print(f"  mps_scaling_rows={len(scaling)}")
    print(f"  classical_benchmark_rows={len(classical)}")


if __name__ == "__main__":
    main()
