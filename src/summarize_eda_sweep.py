#!/usr/bin/env python3
"""Summarize EDA sweep runs into ranked CSV + markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_runs(sweep_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_dir in sorted([p for p in sweep_root.iterdir() if p.is_dir() and p.name != "summary"]):
        m = run_dir / "train" / "metrics.csv"
        if not m.exists():
            continue
        df = pd.read_csv(m)
        if df.empty:
            continue
        df.insert(0, "run_id", run_dir.name)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _write_report(df: pd.DataFrame, out_md: Path, baseline_run: str) -> None:
    if df.empty:
        out_md.write_text("# EDA Sweep Summary\n\nNo runs found.\n", encoding="utf-8")
        return

    dfr = df.sort_values(["r2_y", "rmse_y"], ascending=[False, True]).reset_index(drop=True)
    lines: list[str] = []
    lines.append("# EDA Sweep Summary")
    lines.append("")
    lines.append(f"- Runs analyzed: **{len(df)}**")
    lines.append(f"- Baseline run: **{baseline_run}**")
    lines.append("")
    lines.append("## Top runs by `r2_y`")
    lines.append("")
    lines.append("| run_id | r2_y | rmse_y | mae_y | n_features |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in dfr.head(10).iterrows():
        lines.append(
            f"| {r['run_id']} | {r['r2_y']:.4f} | {r['rmse_y']:.3f} | "
            f"{r['mae_y']:.3f} | {int(r['n_features'])} |"
        )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    best = dfr.iloc[0]
    lines.append(
        f"- Best current run appears to be **{best['run_id']}** "
        f"(r2_y={best['r2_y']:.4f}, rmse_y={best['rmse_y']:.3f})."
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=Path, default=Path("out") / "sweeps")
    p.add_argument("--baseline-run", default="baseline_totals")
    args = p.parse_args()

    root = args.sweep_root
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    df = _load_runs(root)
    out_csv = summary_dir / "metrics_all.csv"
    out_md = summary_dir / "report.md"

    if df.empty:
        out_csv.write_text("", encoding="utf-8")
        _write_report(df, out_md, args.baseline_run)
        print(f"[warn] no runs found under {root}")
        return

    if args.baseline_run in set(df["run_id"]):
        b = float(df.loc[df["run_id"] == args.baseline_run, "rmse_y"].iloc[0])
        df["rmse_ratio_vs_baseline"] = df["rmse_y"] / b if b > 0 else pd.NA
    else:
        df["rmse_ratio_vs_baseline"] = pd.NA

    df.to_csv(out_csv, index=False)
    _write_report(df, out_md, args.baseline_run)
    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_md}")


if __name__ == "__main__":
    main()
