#!/usr/bin/env python3
"""Generate matplotlib visualizations for recorded sweep metrics under out/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan model sweep artifacts in out/ and generate metric visualizations "
            "(heatmaps, bar charts, line charts, scatter plots)."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("out/models/totals"),
        help="Root containing model sweep outputs and metadata.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/plots/metrics"),
        help="Directory where plot images are written.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        help="Image format for plots (png, pdf, svg, ...).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster outputs.",
    )
    return parser.parse_args()


def _load_runs(input_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for meta_path in sorted(input_root.glob("**/metadata.json")):
        data = json.loads(meta_path.read_text())
        rel = meta_path.relative_to(input_root)
        trials_path = meta_path.with_name("optuna_trials.csv")
        trial_best = np.nan
        trial_median = np.nan
        if trials_path.exists():
            tdf = pd.read_csv(trials_path)
            if "value" in tdf.columns and len(tdf):
                trial_best = float(tdf["value"].min())
                trial_median = float(tdf["value"].median())

        rows.append(
            {
                "dataset": data.get("dataset"),
                "variant": data.get("variant"),
                "target_mode": data.get("target_mode"),
                "run_key": f"{data.get('dataset')}/{data.get('variant')}/{data.get('target_mode')}",
                "path": str(rel),
                "feature_count": data.get("feature_count"),
                "trials_completed": data.get("n_trials_completed"),
                "elapsed_sec": data.get("elapsed_sec"),
                "val_rmse": data.get("metrics", {}).get("val", {}).get("rmse"),
                "val_mae": data.get("metrics", {}).get("val", {}).get("mae"),
                "val_r2": data.get("metrics", {}).get("val", {}).get("r2"),
                "test_rmse": data.get("metrics", {}).get("test", {}).get("rmse"),
                "test_mae": data.get("metrics", {}).get("test", {}).get("mae"),
                "test_r2": data.get("metrics", {}).get("test", {}).get("r2"),
                "baseline_val_rmse": data.get("baseline_metrics", {}).get("val", {}).get("rmse"),
                "baseline_test_rmse": data.get("baseline_metrics", {}).get("test", {}).get("rmse"),
                "trial_best_objective": trial_best,
                "trial_median_objective": trial_median,
                "meta_path": str(meta_path.resolve()),
                "trials_path": str(trials_path.resolve()) if trials_path.exists() else "",
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No metadata.json found under '{input_root}'. Expected model sweep outputs."
        )
    df = pd.DataFrame(rows)
    df["val_rmse_gain_vs_baseline"] = df["baseline_val_rmse"] - df["val_rmse"]
    df["test_rmse_gain_vs_baseline"] = df["baseline_test_rmse"] - df["test_rmse"]
    return df


def _save(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_grouped_bar_rmse(df: pd.DataFrame, out_dir: Path, fmt: str, dpi: int) -> None:
    runs = df.sort_values(["dataset", "variant", "target_mode"]).reset_index(drop=True)
    x = np.arange(len(runs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 1.2), 5))
    ax.bar(x - width / 2, runs["val_rmse"], width=width, label="val_rmse")
    ax.bar(x + width / 2, runs["test_rmse"], width=width, label="test_rmse")
    ax.set_xticks(x)
    ax.set_xticklabels(runs["run_key"], rotation=45, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Validation vs Test RMSE by Run")
    ax.legend()
    _save(fig, out_dir / f"rmse_grouped_bar.{fmt}", dpi)


def _plot_bar_rmse_gain(df: pd.DataFrame, out_dir: Path, fmt: str, dpi: int) -> None:
    runs = df.sort_values("val_rmse_gain_vs_baseline", ascending=False).reset_index(drop=True)
    x = np.arange(len(runs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(runs) * 1.2), 5))
    ax.bar(x - width / 2, runs["val_rmse_gain_vs_baseline"], width=width, label="val gain")
    ax.bar(x + width / 2, runs["test_rmse_gain_vs_baseline"], width=width, label="test gain")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(runs["run_key"], rotation=45, ha="right")
    ax.set_ylabel("Baseline RMSE - Model RMSE")
    ax.set_title("RMSE Improvement vs Baseline")
    ax.legend()
    _save(fig, out_dir / f"rmse_gain_vs_baseline_bar.{fmt}", dpi)


def _plot_scatter_val_vs_test(df: pd.DataFrame, out_dir: Path, fmt: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    modes = sorted(df["target_mode"].dropna().unique().tolist())
    markers = ["o", "s", "^", "D", "P", "X"]
    for idx, mode in enumerate(modes):
        part = df[df["target_mode"] == mode]
        ax.scatter(
            part["val_rmse"],
            part["test_rmse"],
            marker=markers[idx % len(markers)],
            s=80,
            alpha=0.85,
            label=f"mode={mode}",
        )
        for _, row in part.iterrows():
            ax.annotate(
                row["dataset"],
                (row["val_rmse"], row["test_rmse"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.set_xlabel("Validation RMSE")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Validation vs Test RMSE")
    ax.legend()
    _save(fig, out_dir / f"rmse_scatter_val_vs_test.{fmt}", dpi)


def _plot_heatmap_metric(
    df: pd.DataFrame, metric_col: str, out_dir: Path, fmt: str, dpi: int
) -> None:
    for variant in sorted(df["variant"].dropna().unique()):
        part = df[df["variant"] == variant]
        pivot = part.pivot_table(
            index="dataset", columns="target_mode", values=metric_col, aggfunc="mean"
        )
        if pivot.empty:
            continue
        vals = pivot.values.astype(float)
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        im = ax.imshow(vals, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"{metric_col} heatmap (variant={variant})")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i, j]:.3g}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save(fig, out_dir / f"heatmap_{metric_col}_{variant}.{fmt}", dpi)


def _plot_line_trials(input_root: Path, df: pd.DataFrame, out_dir: Path, fmt: str, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    plotted = 0
    for _, row in df.iterrows():
        trials_path = Path(row["trials_path"]) if row["trials_path"] else None
        if not trials_path or not trials_path.exists():
            continue
        tdf = pd.read_csv(trials_path)
        if "value" not in tdf.columns or tdf.empty:
            continue
        y = tdf["value"].cummin().to_numpy()
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, label=row["run_key"])
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best objective so far (val RMSE)")
    ax.set_title("Optuna Optimization Progress")
    ax.legend(fontsize=7, loc="best")
    _save(fig, out_dir / f"optuna_best_so_far_lines.{fmt}", dpi)


def _plot_line_elapsed_vs_val(df: pd.DataFrame, out_dir: Path, fmt: str, dpi: int) -> None:
    runs = df.sort_values("elapsed_sec").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(runs["elapsed_sec"], runs["val_rmse"], marker="o")
    for _, row in runs.iterrows():
        ax.annotate(row["run_key"], (row["elapsed_sec"], row["val_rmse"]), fontsize=7)
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Runtime vs Validation RMSE")
    _save(fig, out_dir / f"runtime_vs_val_rmse_line.{fmt}", dpi)


def _write_summary_csv(df: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_cols = [
        "dataset",
        "variant",
        "target_mode",
        "feature_count",
        "trials_completed",
        "elapsed_sec",
        "val_rmse",
        "test_rmse",
        "val_r2",
        "test_r2",
        "baseline_val_rmse",
        "baseline_test_rmse",
        "val_rmse_gain_vs_baseline",
        "test_rmse_gain_vs_baseline",
        "trial_best_objective",
        "trial_median_objective",
        "meta_path",
        "trials_path",
    ]
    out_path = out_dir / "metrics_summary.csv"
    df[summary_cols].sort_values(["dataset", "variant", "target_mode"]).to_csv(
        out_path, index=False
    )
    return out_path


def run() -> None:
    args = _parse_args()
    df = _load_runs(args.input_root)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = _write_summary_csv(df, out_dir)
    _plot_grouped_bar_rmse(df, out_dir, args.format, args.dpi)
    _plot_bar_rmse_gain(df, out_dir, args.format, args.dpi)
    _plot_scatter_val_vs_test(df, out_dir, args.format, args.dpi)
    _plot_heatmap_metric(df, "val_rmse", out_dir, args.format, args.dpi)
    _plot_heatmap_metric(df, "test_r2", out_dir, args.format, args.dpi)
    _plot_line_trials(args.input_root, df, out_dir, args.format, args.dpi)
    _plot_line_elapsed_vs_val(df, out_dir, args.format, args.dpi)

    print("Metric visualization complete.")
    print(f"Input root: {args.input_root.resolve()}")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Summary CSV: {summary_csv.resolve()}")


if __name__ == "__main__":
    run()
