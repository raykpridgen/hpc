#!/usr/bin/env python3
"""Generate interpretable matplotlib plots for each pipeline stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create stage-by-stage visualizations from out artifacts "
            "(features, leakage, EDA, models, sweeps)."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("out/out"),
        help="Root directory containing stage outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/out/plots/stages"),
        help="Directory for generated plots.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        help="Image format for generated plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for raster plot output.",
    )
    return parser.parse_args()


def _safe_save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _collect_stage_meta(stage_root: Path) -> List[Tuple[Path, Dict[str, object]]]:
    out = []
    if not stage_root.exists():
        return out
    for p in sorted(stage_root.glob("**/metadata.json")):
        out.append((p, _read_json(p)))
    return out


def _plot_stage_overview(
    counts: Dict[str, int], out_dir: Path, fmt: str, dpi: int
) -> None:
    labels = list(counts.keys())
    vals = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(labels))
    ax.bar(x, vals)
    ax.set_title("Pipeline Stage Artifact Coverage")
    ax.set_ylabel("Metadata files detected")
    ax.set_xlabel("Stage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    _safe_save(fig, out_dir / f"00_stage_coverage.{fmt}", dpi)


def _features_df(meta: List[Tuple[Path, Dict[str, object]]]) -> pd.DataFrame:
    rows = []
    for _, d in meta:
        rows.append(
            {
                "dataset": d.get("dataset_name"),
                "runtime_scale_factor": d.get("runtime_scale_factor"),
                "rows_train": d.get("rows", {}).get("train"),
                "rows_val": d.get("rows", {}).get("val"),
                "rows_test": d.get("rows", {}).get("test"),
                "invalid_train": d.get("invalid_io_intensity_rows", {}).get("train"),
                "invalid_val": d.get("invalid_io_intensity_rows", {}).get("val"),
                "invalid_test": d.get("invalid_io_intensity_rows", {}).get("test"),
            }
        )
    return pd.DataFrame(rows)


def _plot_features_stage(df: pd.DataFrame, out_dir: Path, fmt: str, dpi: int) -> None:
    if df.empty:
        return
    df = df.sort_values("dataset")
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(7, len(df) * 1.4), 5))
    ax.bar(x - width, df["rows_train"], width=width, label="train")
    ax.bar(x, df["rows_val"], width=width, label="val")
    ax.bar(x + width, df["rows_test"], width=width, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=20, ha="right")
    ax.set_title("Feature Stage: Split Row Counts")
    ax.set_ylabel("Rows")
    ax.legend()
    _safe_save(fig, out_dir / f"01_features_split_rows.{fmt}", dpi)

    fig, ax = plt.subplots(figsize=(max(7, len(df) * 1.4), 5))
    ax.bar(x - width, df["invalid_train"], width=width, label="invalid train")
    ax.bar(x, df["invalid_val"], width=width, label="invalid val")
    ax.bar(x + width, df["invalid_test"], width=width, label="invalid test")
    ax.set_xticks(x)
    ax.set_xticklabels(df["dataset"], rotation=20, ha="right")
    ax.set_title("Feature Stage: Invalid Target Rows")
    ax.set_ylabel("Count")
    ax.legend()
    _safe_save(fig, out_dir / f"02_features_invalid_rows.{fmt}", dpi)

    fig, ax = plt.subplots(figsize=(max(7, len(df) * 1.4), 4.5))
    x2 = np.arange(len(df))
    ax.bar(x2, df["runtime_scale_factor"])
    ax.set_yscale("log")
    ax.set_title("Feature Stage: Runtime Scale Factor (log axis)")
    ax.set_ylabel("runtime_scale_factor")
    ax.set_xticks(x2)
    ax.set_xticklabels(df["dataset"], rotation=20, ha="right")
    _safe_save(fig, out_dir / f"03_features_runtime_scale.{fmt}", dpi)


def _leakage_df(meta: List[Tuple[Path, Dict[str, object]]]) -> pd.DataFrame:
    rows = []
    for _, d in meta:
        rows.append(
            {
                "dataset": d.get("dataset"),
                "variant": d.get("variant"),
                "drop_count": d.get("drop_count"),
                "feature_columns_count": d.get("feature_columns_count"),
            }
        )
    return pd.DataFrame(rows)


def _plot_leakage_stage(df: pd.DataFrame, out_dir: Path, fmt: str, dpi: int) -> None:
    if df.empty:
        return
    df["run"] = df["dataset"].astype(str) + "/" + df["variant"].astype(str)
    df = df.sort_values("run")
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.2), 5))
    ax.bar(x - width / 2, df["drop_count"], width=width, label="dropped columns")
    ax.bar(
        x + width / 2,
        df["feature_columns_count"],
        width=width,
        label="remaining features",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["run"], rotation=45, ha="right")
    ax.set_title("Leakage Stage: Feature Removal vs Remaining")
    ax.set_ylabel("Columns")
    ax.legend()
    _safe_save(fig, out_dir / f"04_leakage_drop_vs_remaining.{fmt}", dpi)


def _eda_df(meta: List[Tuple[Path, Dict[str, object]]]) -> pd.DataFrame:
    rows = []
    for _, d in meta:
        rows.append(
            {
                "dataset": d.get("dataset"),
                "variant": d.get("variant"),
                "feature_count_numeric": d.get("feature_count_numeric"),
                "training_feature_count": d.get("training_feature_count"),
                "log_target": d.get("log_target"),
                "val_rmse": d.get("metrics", {}).get("val", {}).get("rmse"),
                "val_r2": d.get("metrics", {}).get("val", {}).get("r2"),
                "test_rmse": d.get("metrics", {}).get("test", {}).get("rmse"),
                "test_r2": d.get("metrics", {}).get("test", {}).get("r2"),
            }
        )
    return pd.DataFrame(rows)


def _plot_eda_stage(
    df: pd.DataFrame, eda_root: Path, out_dir: Path, fmt: str, dpi: int
) -> None:
    if df.empty:
        return
    df["run"] = df["dataset"].astype(str) + "/" + df["variant"].astype(str)
    df = df.sort_values("run")
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.2), 5))
    ax.bar(x - width / 2, df["val_rmse"], width=width, label="val_rmse")
    ax.bar(x + width / 2, df["test_rmse"], width=width, label="test_rmse")
    ax.set_xticks(x)
    ax.set_xticklabels(df["run"], rotation=45, ha="right")
    ax.set_title("EDA Stage: Baseline Model RMSE")
    ax.set_ylabel("RMSE")
    ax.legend()
    _safe_save(fig, out_dir / f"05_eda_rmse.{fmt}", dpi)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.2), 5))
    ax.bar(x - width / 2, df["feature_count_numeric"], width=width, label="numeric features")
    ax.bar(
        x + width / 2,
        df["training_feature_count"],
        width=width,
        label="selected training features",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["run"], rotation=45, ha="right")
    ax.set_title("EDA Stage: Feature Reduction")
    ax.set_ylabel("Count")
    ax.legend()
    _safe_save(fig, out_dir / f"06_eda_feature_reduction.{fmt}", dpi)

    # Top feature importance charts for each run.
    for run in df["run"]:
        dataset, variant = run.split("/", 1)
        imp_path = eda_root / dataset / variant / "feature_importance.csv"
        if not imp_path.exists():
            continue
        imp = pd.read_csv(imp_path)
        if imp.empty or "feature" not in imp.columns:
            continue
        top = imp.head(15)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(top["feature"][::-1], top["score_combined"][::-1])
        ax.set_title(f"EDA Top Features ({run})")
        ax.set_xlabel("Combined importance score")
        _safe_save(
            fig, out_dir / f"07_eda_top_features_{dataset}_{variant}.{fmt}", dpi
        )


def _models_df(meta: List[Tuple[Path, Dict[str, object]]]) -> pd.DataFrame:
    rows = []
    for meta_path, d in meta:
        rows.append(
            {
                "dataset": d.get("dataset"),
                "variant": d.get("variant"),
                "target_mode": d.get("target_mode"),
                "sweep_profile": d.get("sweep_profile"),
                "trials": d.get("n_trials_completed"),
                "elapsed_sec": d.get("elapsed_sec"),
                "val_rmse": d.get("metrics", {}).get("val", {}).get("rmse"),
                "test_rmse": d.get("metrics", {}).get("test", {}).get("rmse"),
                "val_r2": d.get("metrics", {}).get("val", {}).get("r2"),
                "test_r2": d.get("metrics", {}).get("test", {}).get("r2"),
                "baseline_val_rmse": d.get("baseline_metrics", {}).get("val", {}).get("rmse"),
                "baseline_test_rmse": d.get("baseline_metrics", {}).get("test", {}).get("rmse"),
                "trials_csv": d.get("outputs", {}).get("trials_csv"),
                "meta_path": str(meta_path.resolve()),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["val_gain"] = df["baseline_val_rmse"] - df["val_rmse"]
        df["test_gain"] = df["baseline_test_rmse"] - df["test_rmse"]
    return df


def _resolve_trials_path(row: pd.Series) -> Optional[Path]:
    p = row.get("trials_csv")
    if isinstance(p, str) and p:
        trial_path = Path(p)
        if trial_path.exists():
            return trial_path
    meta_path = row.get("meta_path")
    if isinstance(meta_path, str) and meta_path:
        local_candidate = Path(meta_path).with_name("optuna_trials.csv")
        if local_candidate.exists():
            return local_candidate
    return None


def _plot_models_stage(df: pd.DataFrame, out_dir: Path, fmt: str, dpi: int) -> None:
    if df.empty:
        return
    df["run"] = (
        df["dataset"].astype(str)
        + "/"
        + df["variant"].astype(str)
        + "/"
        + df["target_mode"].astype(str)
    )
    df = df.sort_values("run")
    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.2), 5))
    ax.bar(x - width / 2, df["val_rmse"], width=width, label="val_rmse")
    ax.bar(x + width / 2, df["test_rmse"], width=width, label="test_rmse")
    ax.set_xticks(x)
    ax.set_xticklabels(df["run"], rotation=45, ha="right")
    ax.set_title("Model Stage: Validation vs Test RMSE")
    ax.set_ylabel("RMSE")
    ax.legend()
    _safe_save(fig, out_dir / f"08_models_rmse.{fmt}", dpi)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 1.2), 5))
    ax.bar(x - width / 2, df["val_gain"], width=width, label="val gain vs baseline")
    ax.bar(x + width / 2, df["test_gain"], width=width, label="test gain vs baseline")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["run"], rotation=45, ha="right")
    ax.set_title("Model Stage: RMSE Gain vs Baseline")
    ax.set_ylabel("baseline_rmse - model_rmse")
    ax.legend()
    _safe_save(fig, out_dir / f"09_models_baseline_gain.{fmt}", dpi)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    for mode in sorted(df["target_mode"].dropna().unique()):
        part = df[df["target_mode"] == mode]
        ax.scatter(part["val_rmse"], part["test_rmse"], s=90, label=mode, alpha=0.85)
        for _, r in part.iterrows():
            ax.annotate(r["dataset"], (r["val_rmse"], r["test_rmse"]), fontsize=8)
    ax.set_xlabel("Validation RMSE")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Model Stage: Generalization Scatter")
    ax.legend(title="target_mode")
    _safe_save(fig, out_dir / f"10_models_scatter.{fmt}", dpi)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    order = df.sort_values("elapsed_sec")
    ax.plot(order["elapsed_sec"], order["val_rmse"], marker="o")
    for _, r in order.iterrows():
        ax.annotate(r["run"], (r["elapsed_sec"], r["val_rmse"]), fontsize=7)
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Validation RMSE")
    ax.set_title("Model Stage: Runtime vs Validation RMSE")
    _safe_save(fig, out_dir / f"11_models_runtime_vs_rmse.{fmt}", dpi)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    plotted = 0
    for _, r in df.iterrows():
        p = _resolve_trials_path(r)
        if not p:
            continue
        trials = pd.read_csv(p)
        if "value" not in trials.columns or trials.empty:
            continue
        y = trials["value"].cummin().to_numpy()
        ax.plot(np.arange(1, len(y) + 1), y, label=r["run"])
        plotted += 1
    if plotted:
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best objective so far (val RMSE)")
        ax.set_title("Model Stage: Optuna Convergence Curves")
        ax.legend(fontsize=7)
        _safe_save(fig, out_dir / f"12_models_optuna_convergence.{fmt}", dpi)
    else:
        plt.close(fig)


def _load_join_stats(paths: Iterable[Path], suite_name: str) -> pd.DataFrame:
    rows = []
    for p in paths:
        run = p.parent.parent.name
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        row = {
            "suite": suite_name,
            "run": run,
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "detail_used_true": int(df["detail_used"].astype(bool).sum())
            if "detail_used" in df.columns
            else np.nan,
            "detail_used_false": int((~df["detail_used"].astype(bool)).sum())
            if "detail_used" in df.columns
            else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_sweeps_stage(
    sweeps_root: Path, sweeps_detail_root: Path, out_dir: Path, fmt: str, dpi: int
) -> None:
    sweep_paths = sorted(sweeps_root.glob("**/eda/join_stats.csv")) if sweeps_root.exists() else []
    detail_paths = (
        sorted(sweeps_detail_root.glob("**/eda/join_stats.csv"))
        if sweeps_detail_root.exists()
        else []
    )
    df = pd.concat(
        [
            _load_join_stats(sweep_paths, "sweeps"),
            _load_join_stats(detail_paths, "sweeps_detail"),
        ],
        ignore_index=True,
    )
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    summary = df.groupby("suite", as_index=False).agg(
        runs=("run", "count"), avg_columns=("columns", "mean")
    )
    ax.bar(summary["suite"], summary["runs"])
    ax.set_title("Sweep Stage: Number of Sweep Configurations")
    ax.set_ylabel("Runs with join_stats.csv")
    for i, v in enumerate(summary["runs"]):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    _safe_save(fig, out_dir / f"13_sweeps_run_counts.{fmt}", dpi)

    if df["detail_used_true"].notna().any():
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        grp = df.groupby("suite", as_index=False).agg(
            detail_true=("detail_used_true", "sum"),
            detail_false=("detail_used_false", "sum"),
        )
        x = np.arange(len(grp))
        ax.bar(x, grp["detail_true"], label="detail_used=True")
        ax.bar(x, grp["detail_false"], bottom=grp["detail_true"], label="detail_used=False")
        ax.set_xticks(x)
        ax.set_xticklabels(grp["suite"])
        ax.set_ylabel("Count")
        ax.set_title("Sweep Stage: Detail Usage Flags")
        ax.legend()
        _safe_save(fig, out_dir / f"14_sweeps_detail_usage.{fmt}", dpi)


def _write_layman_summary(
    output_dir: Path,
    counts: Dict[str, int],
    features: pd.DataFrame,
    leakage: pd.DataFrame,
    eda: pd.DataFrame,
    models: pd.DataFrame,
) -> None:
    lines = [
        "# Pipeline Visualization Guide",
        "",
        "This folder contains simple visuals for each pipeline stage.",
        "",
        "## Stage Coverage",
    ]
    for stage, n in counts.items():
        lines.append(f"- `{stage}`: {n} metadata files detected")

    lines += ["", "## How to Read the Plots", ""]
    lines += [
        "- `00_stage_coverage`: confirms which stages produced artifacts.",
        "- `01-03_features_*`: data volume, invalid targets, and runtime scaling setup.",
        "- `04_leakage_*`: how many columns were removed vs retained after leakage policy.",
        "- `05-07_eda_*`: predictive quality and top explanatory features.",
        "- `08-12_models_*`: model quality, baseline gain, optimization progress, and runtime tradeoffs.",
        "- `13-14_sweeps_*`: sweep configuration inventory and detail-usage behavior.",
    ]

    if not models.empty:
        best_idx = models["val_rmse"].astype(float).idxmin()
        best = models.loc[best_idx]
        lines += [
            "",
            "## Best Validation Model (from metadata)",
            f"- Run: `{best['dataset']}/{best['variant']}/{best['target_mode']}`",
            f"- val_rmse: `{best['val_rmse']:.6g}`",
            f"- test_rmse: `{best['test_rmse']:.6g}`",
            f"- val_r2: `{best['val_r2']:.4f}`",
            f"- test_r2: `{best['test_r2']:.4f}`",
        ]

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    args = _parse_args()
    root = args.input_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    features_meta = _collect_stage_meta(root / "features")
    leakage_meta = _collect_stage_meta(root / "leakage")
    eda_meta = _collect_stage_meta(root / "eda")
    models_meta = _collect_stage_meta(root / "models")

    counts = {
        "features": len(features_meta),
        "leakage": len(leakage_meta),
        "eda": len(eda_meta),
        "models": len(models_meta),
        "sweeps_join_stats": len(list((root / "sweeps").glob("**/eda/join_stats.csv")))
        if (root / "sweeps").exists()
        else 0,
        "sweeps_detail_join_stats": len(
            list((root / "sweeps_detail").glob("**/eda/join_stats.csv"))
        )
        if (root / "sweeps_detail").exists()
        else 0,
    }

    features_df = _features_df(features_meta)
    leakage_df = _leakage_df(leakage_meta)
    eda_df = _eda_df(eda_meta)
    models_df = _models_df(models_meta)

    _plot_stage_overview(counts, output_dir, args.format, args.dpi)
    _plot_features_stage(features_df, output_dir, args.format, args.dpi)
    _plot_leakage_stage(leakage_df, output_dir, args.format, args.dpi)
    _plot_eda_stage(eda_df, root / "eda" / "totals", output_dir, args.format, args.dpi)
    _plot_models_stage(models_df, output_dir, args.format, args.dpi)
    _plot_sweeps_stage(root / "sweeps", root / "sweeps_detail", output_dir, args.format, args.dpi)

    if not models_df.empty:
        models_df.to_csv(output_dir / "models_metrics_summary.csv", index=False)
    if not eda_df.empty:
        eda_df.to_csv(output_dir / "eda_metrics_summary.csv", index=False)
    if not leakage_df.empty:
        leakage_df.to_csv(output_dir / "leakage_summary.csv", index=False)
    if not features_df.empty:
        features_df.to_csv(output_dir / "features_summary.csv", index=False)

    _write_layman_summary(output_dir, counts, features_df, leakage_df, eda_df, models_df)

    print("Stage visualization complete.")
    print(f"Input root: {root.resolve()}")
    print(f"Output dir: {output_dir.resolve()}")


if __name__ == "__main__":
    run()
