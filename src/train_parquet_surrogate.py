#!/usr/bin/env python3
"""
Train POSIX I/O intensity surrogate from darshan_total Parquet (specs/design.md).

Run from repo root with conda env:
  conda activate ml_base
  pip install -r requirements.txt
  python src/train_parquet_surrogate.py --total-glob "data/App10.parquet" --no-detail
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from darshan_surrogate.artifacts import save_join_stats, save_metrics, save_y_eda
from darshan_surrogate.detail_join import join_detail_features
from darshan_surrogate.features import apply_group_filters, build_feature_matrix
from darshan_surrogate.leakage import (
    apply_correlation_leakage_filter,
    apply_name_substring_prefilter,
    apply_timestamp_prefilter,
    assert_no_forbidden_columns,
)
from darshan_surrogate.io import load_totals_with_app_id
from darshan_surrogate.paths import ensure_out_dirs, project_root
from darshan_surrogate.split import time_ordered_split
from darshan_surrogate.targets import log1p_target, posix_intensity
from darshan_surrogate.training import metrics_on_original_scale, train_surrogate


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Darshan Parquet I/O intensity surrogate.")
    p.add_argument(
        "--total-glob",
        nargs="+",
        default=["data/App10.parquet"],
        help="One or more App*.parquet paths (relative to repo root or absolute).",
    )
    p.add_argument("--detail-root", default="data/2021", help="Root for darshan_detail hierarchy.")
    p.add_argument("--no-detail", action="store_true", help="Skip detail join; totals only.")
    p.add_argument("--train-frac", type=float, default=0.8, help="Early fraction for time-ordered train.")
    p.add_argument("--missing-threshold", type=float, default=0.2, help="Detail missing-rate fallback (design §3.3).")
    p.add_argument("--optuna-trials", type=int, default=0, help="Optuna trials; 0 = fixed default hyperparameters.")
    p.add_argument("--out", default="out", help="Output root (eda/, train/, models/).")
    p.add_argument(
        "--max-abs-corr-with-y",
        type=float,
        default=0.95,
        help="Drop total_/detail_ train features with |corr(x,y)| above this (proxy leakage; train split only).",
    )
    p.add_argument(
        "--no-corr-filter",
        action="store_true",
        help="Disable correlation-based proxy-leakage filter (not recommended).",
    )
    p.add_argument(
        "--drop-name-substrings",
        nargs="+",
        default=[],
        help=(
            "Optional coarse prefilter: drop feature columns containing any of these "
            "substrings (case-insensitive), e.g. BYTES TIME."
        ),
    )
    p.add_argument(
        "--include-groups",
        nargs="+",
        default=[],
        help=(
            "Optional coarse feature groups to keep. Allowed: "
            "posix mpiio h5 stdio lustre bgq pnetcdf detail calendar app other."
        ),
    )
    p.add_argument(
        "--exclude-groups",
        nargs="+",
        default=[],
        help=(
            "Optional coarse feature groups to drop. Allowed: "
            "posix mpiio h5 stdio lustre bgq pnetcdf detail calendar app other."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo = project_root()
    out_name = args.out
    if Path(out_name).is_absolute():
        out_root = Path(out_name)
        out = ensure_out_dirs(out_root.parent, out_root.name)
    else:
        out = ensure_out_dirs(repo, out_name)

    total_paths: list[Path] = []
    for p in args.total_glob:
        token = str(p)
        # Allow shell-style wildcards passed as literal args (e.g. App*.parquet).
        if any(ch in token for ch in "*?[]"):
            pattern = (repo / token).as_posix() if not Path(token).is_absolute() else token
            for m in sorted(glob.glob(pattern)):
                total_paths.append(Path(m))
        else:
            total_paths.append((repo / token).resolve() if not Path(token).is_absolute() else Path(token))

    # stable de-dup preserving order
    seen: set[Path] = set()
    deduped: list[Path] = []
    for tp in total_paths:
        rp = tp.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        deduped.append(rp)
    total_paths = deduped
    for tp in total_paths:
        if not tp.exists():
            raise FileNotFoundError(f"Total parquet not found: {tp}")

    df = load_totals_with_app_id([str(p) for p in total_paths])
    join_stats: dict = {"detail_used": False}

    if not args.no_detail:
        detail_root = (repo / args.detail_root).resolve() if not Path(args.detail_root).is_absolute() else Path(args.detail_root)
        if not detail_root.is_dir():
            print(f"Warning: detail root missing {detail_root}; proceeding with --no-detail behaviour.")
            join_stats["detail_used"] = False
        else:
            df, join_stats = join_detail_features(df, detail_root, missing_threshold=args.missing_threshold)
            join_stats["detail_used"] = True
            print("Join stats:", join_stats)

    save_join_stats(out["eda"] / "join_stats.csv", join_stats)

    df = df.reset_index(drop=True)
    y = posix_intensity(df)
    valid = np.isfinite(y) & (y >= 0)
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(float).reset_index(drop=True)
    if len(df) < 10:
        raise RuntimeError(f"Too few rows after filtering ({len(df)}); check data and detail join.")

    save_y_eda(out["eda"] / "y_percentiles.csv", y.values)

    train_pos, val_pos = time_ordered_split(df, train_frac=args.train_frac)
    y_train, y_val = y.iloc[train_pos], y.iloc[val_pos]
    z_train = log1p_target(y_train.values)
    z_val = log1p_target(y_val.values)

    X_all, le, _ = build_feature_matrix(
        df,
        train_pos,
        drop_sparse_frac=None,
    )
    X_all, timestamp_prefilter_drops = apply_timestamp_prefilter(X_all)
    if timestamp_prefilter_drops:
        pd.DataFrame({"column": timestamp_prefilter_drops}).to_csv(
            out["eda"] / "timestamp_prefilter_drops.csv",
            index=False,
        )
        print(f"Timestamp prefilter: dropped {len(timestamp_prefilter_drops)} features.")

    name_prefilter_drops: list[str] = []
    if args.drop_name_substrings:
        X_all, name_prefilter_drops = apply_name_substring_prefilter(
            X_all,
            args.drop_name_substrings,
        )
        if name_prefilter_drops:
            pd.DataFrame({"column": name_prefilter_drops}).to_csv(
                out["eda"] / "name_prefilter_drops.csv",
                index=False,
            )
            print(
                f"Name prefilter: dropped {len(name_prefilter_drops)} features "
                f"for substrings={args.drop_name_substrings}."
            )
    group_filter_drops: list[str] = []
    if args.include_groups or args.exclude_groups:
        X_all, group_filter_drops = apply_group_filters(
            X_all,
            include_groups=args.include_groups,
            exclude_groups=args.exclude_groups,
        )
        if group_filter_drops:
            pd.DataFrame({"column": group_filter_drops}).to_csv(
                out["eda"] / "group_filter_drops.csv",
                index=False,
            )
        print(
            "Group filter:",
            {
                "include": args.include_groups,
                "exclude": args.exclude_groups,
                "n_dropped": len(group_filter_drops),
            },
        )
    assert_no_forbidden_columns(X_all.columns)

    corr_records: list[dict] = []
    if not args.no_corr_filter:
        X_all, corr_records = apply_correlation_leakage_filter(
            X_all,
            train_pos,
            y.values,
            max_abs_corr=args.max_abs_corr_with_y,
        )
        assert_no_forbidden_columns(X_all.columns)
        if corr_records:
            pd.DataFrame(corr_records).to_csv(out["eda"] / "correlation_leakage_drops.csv", index=False)
            print(
                f"Proxy-leakage filter: dropped {len(corr_records)} features "
                f"(|corr|>{args.max_abs_corr_with_y} on train vs y)."
            )
    if X_all.shape[1] < 8:
        print(
            "Warning: very few features after leakage filters — "
            "consider adjusting --max-abs-corr-with-y or reviewing EDA.",
        )

    X_train = X_all.iloc[train_pos]
    X_val = X_all.iloc[val_pos]

    model, metrics_z = train_surrogate(
        X_train,
        z_train,
        X_val,
        z_val,
        n_optuna_trials=args.optuna_trials,
    )
    z_pred = model.predict(X_val)
    metrics_y = metrics_on_original_scale(y_val.values, z_pred)

    med = float(np.median(y_train.values))
    baseline_rmse = float(np.sqrt(np.mean((y_val.values - med) ** 2)))

    row = {
        **metrics_y,
        **{f"val_{k}": v for k, v in metrics_z.items()},
        "baseline_rmse_y_median_train": baseline_rmse,
        "n_train": len(train_pos),
        "n_val": len(val_pos),
        "n_features": X_all.shape[1],
        "max_abs_corr_with_y": args.max_abs_corr_with_y,
        "corr_filter_disabled": bool(args.no_corr_filter),
        "n_corr_leakage_drops": len(corr_records),
        "n_name_prefilter_drops": len(name_prefilter_drops),
        "n_timestamp_prefilter_drops": len(timestamp_prefilter_drops),
        "n_group_filter_drops": len(group_filter_drops),
    }
    save_metrics(out["train"] / "metrics.csv", row)
    print("Validation (original y):", metrics_y)
    print("Baseline RMSE (median y_train):", baseline_rmse)

    fi = pd.Series(model.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    fi.to_frame("importance").to_csv(out["train"] / "feature_importance.csv")

    model_path = out["models"] / "xgboost_surrogate.json"
    model.save_model(str(model_path))
    meta = {
        "feature_columns": list(X_all.columns),
        "app_id_classes": le.classes_.tolist(),
        "train_frac": args.train_frac,
        "detail_used": join_stats.get("detail_used", False),
        "max_abs_corr_with_y": args.max_abs_corr_with_y,
        "corr_filter_disabled": args.no_corr_filter,
        "drop_name_substrings": args.drop_name_substrings,
        "include_groups": args.include_groups,
        "exclude_groups": args.exclude_groups,
        "timestamp_prefilter_drops": timestamp_prefilter_drops,
        "name_prefilter_drops": name_prefilter_drops,
        "group_filter_drops": group_filter_drops,
        "correlation_leakage_drops": [r["column"] for r in corr_records],
    }
    with open(out["models"] / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model -> {model_path}")


if __name__ == "__main__":
    main()
