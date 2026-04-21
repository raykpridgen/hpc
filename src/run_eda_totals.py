#!/usr/bin/env python3
"""Run EDA summaries and feature importance on leakage-safe totals splits."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - fallback path
    XGBRegressor = None
    from sklearn.ensemble import RandomForestRegressor


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class EDAConfig:
    input_root: Path
    output_root: Path
    dataset_glob: str
    variant_glob: str
    target_col: str
    top_k: int
    train_feature_count: int
    seed: int
    log_target: bool
    drop_invalid_target: bool
    export_training_dataset: bool


def _parse_args() -> EDAConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Compute EDA summaries and model-based feature importance for leakage-safe "
            "totals datasets."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("out/leakage/totals"),
        help="Root containing leakage variants by dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("out/eda/totals"),
        help="Root for EDA artifacts.",
    )
    parser.add_argument(
        "--dataset-glob",
        type=str,
        default="*",
        help="Dataset glob under input root.",
    )
    parser.add_argument(
        "--variant-glob",
        type=str,
        default="balanced",
        help="Variant glob under each dataset (e.g. balanced, ablation_*).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="io_intensity",
        help="Target column.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top features to include in shortlist.",
    )
    parser.add_argument(
        "--train-feature-count",
        type=int,
        default=30,
        help="Number of top-ranked features to export for training dataset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for EDA model fitting.",
    )
    parser.add_argument(
        "--log-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train/evaluate model on log1p(target) and invert predictions for metrics.",
    )
    parser.add_argument(
        "--drop-invalid-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop invalid target rows (non-finite and non-positive).",
    )
    parser.add_argument(
        "--export-training-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export a top-N feature training dataset parquet bundle.",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0.")
    if args.train_feature_count <= 0:
        raise ValueError("--train-feature-count must be > 0.")
    return EDAConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        dataset_glob=args.dataset_glob,
        variant_glob=args.variant_glob,
        target_col=args.target_col,
        top_k=args.top_k,
        train_feature_count=args.train_feature_count,
        seed=args.seed,
        log_target=args.log_target,
        drop_invalid_target=args.drop_invalid_target,
        export_training_dataset=args.export_training_dataset,
    )


def _discover_targets(root: Path, dataset_glob: str, variant_glob: str) -> List[Tuple[str, str, Path]]:
    targets: List[Tuple[str, str, Path]] = []
    for dataset_dir in sorted(p for p in root.glob(dataset_glob) if p.is_dir()):
        for variant_dir in sorted(p for p in dataset_dir.glob(variant_glob) if p.is_dir()):
            data_dir = variant_dir / "data"
            if all((data_dir / f"{split}.parquet").exists() for split in SPLITS):
                targets.append((dataset_dir.name, variant_dir.name, data_dir))
    if not targets:
        raise FileNotFoundError(
            f"No leakage datasets found under '{root}' with dataset_glob='{dataset_glob}' "
            f"and variant_glob='{variant_glob}'."
        )
    return targets


def _load_splits(data_dir: Path) -> Dict[str, pd.DataFrame]:
    return {split: pd.read_parquet(data_dir / f"{split}.parquet") for split in SPLITS}


def _numeric_features(df: pd.DataFrame, target_col: str) -> List[str]:
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in num_cols if c != target_col]


def _fit_model(seed: int):
    if XGBRegressor is not None:
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=4,
        )
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=seed,
        n_jobs=4,
    )


def _importance_from_model(model, feature_names: List[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        scores = np.asarray(model.feature_importances_, dtype=float)
    else:
        scores = np.zeros(len(feature_names), dtype=float)
    df = pd.DataFrame({"feature": feature_names, "importance_model": scores})
    df = df.sort_values("importance_model", ascending=False).reset_index(drop=True)
    return df


def _corr_with_target(x: pd.DataFrame, y: pd.Series, features: List[str]) -> pd.DataFrame:
    corr_df = x[features].copy()
    corr_df["__target"] = y
    corr = corr_df.corr(numeric_only=True)["__target"]
    corr = corr.drop(labels=["__target"], errors="ignore")
    out = corr.abs().sort_values(ascending=False).rename("abs_corr_target").reset_index()
    out.columns = ["feature", "abs_corr_target"]
    return out


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _distribution_summary(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(columns=["feature", "mean", "std", "q05", "q50", "q95"])
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "mean": float(s.mean()),
                "std": float(s.std()),
                "q05": float(s.quantile(0.05)),
                "q50": float(s.quantile(0.50)),
                "q95": float(s.quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def _valid_target_mask(
    y_raw: pd.Series, drop_invalid_target: bool, log_target: bool
) -> pd.Series:
    valid = y_raw.notna() & np.isfinite(y_raw)
    if drop_invalid_target:
        valid &= y_raw > 0.0
    if log_target:
        valid &= y_raw > -1.0
    return valid


def _transform_target(y_raw: pd.Series, log_target: bool) -> pd.Series:
    return np.log1p(y_raw) if log_target else y_raw


def _inverse_transform_pred(y_pred_model: np.ndarray, log_target: bool) -> np.ndarray:
    return np.expm1(y_pred_model) if log_target else y_pred_model


def _run_one(config: EDAConfig, dataset: str, variant: str, data_dir: Path) -> Dict[str, object]:
    splits = _load_splits(data_dir)
    for split in SPLITS:
        if config.target_col not in splits[split].columns:
            raise KeyError(
                f"Split '{split}' in {dataset}/{variant} missing target '{config.target_col}'."
            )

    features = _numeric_features(splits["train"], config.target_col)
    if not features:
        raise ValueError(f"No numeric features found for {dataset}/{variant}.")

    target_raw = {
        split: pd.to_numeric(splits[split][config.target_col], errors="coerce")
        for split in SPLITS
    }
    valid_masks = {
        split: _valid_target_mask(
            target_raw[split],
            drop_invalid_target=config.drop_invalid_target,
            log_target=config.log_target,
        )
        for split in SPLITS
    }
    invalid_counts = {
        split: int((~valid_masks[split]).sum())
        for split in SPLITS
    }

    x_train = splits["train"].loc[valid_masks["train"], features]
    y_train_raw = target_raw["train"].loc[valid_masks["train"]]
    y_train_model = _transform_target(y_train_raw, log_target=config.log_target)
    if len(y_train_model) == 0:
        raise ValueError(
            f"No valid training target rows after filtering for {dataset}/{variant}."
        )

    model = _fit_model(config.seed)
    model.fit(x_train, y_train_model)

    importance_model = _importance_from_model(model, features)
    corr_table = _corr_with_target(x_train, y_train_model, features)
    importance = importance_model.merge(corr_table, on="feature", how="left")
    importance["score_combined"] = (
        0.7 * importance["importance_model"].fillna(0.0)
        + 0.3 * importance["abs_corr_target"].fillna(0.0)
    )
    importance = importance.sort_values("score_combined", ascending=False).reset_index(drop=True)

    top_k = min(config.top_k, len(importance))
    shortlist = importance["feature"].head(top_k).tolist()
    train_feature_count = min(config.train_feature_count, len(importance))
    training_features = importance["feature"].head(train_feature_count).tolist()

    metrics: Dict[str, Dict[str, float]] = {}
    for split in ("train", "val", "test"):
        x = splits[split].loc[valid_masks[split], features]
        y = target_raw[split].loc[valid_masks[split]]
        if len(y) == 0:
            metrics[split] = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
            continue
        y_pred_model = model.predict(x)
        y_pred = _inverse_transform_pred(np.asarray(y_pred_model), log_target=config.log_target)
        metrics[split] = _evaluate_regression(y.to_numpy(), y_pred)

    out_dir = config.output_root / dataset / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    importance_path = out_dir / "feature_importance.csv"
    corr_path = out_dir / "feature_corr_target.csv"
    shortlist_path = out_dir / "shortlist_top_features.txt"
    dist_path = out_dir / "distribution_summary_train.csv"
    metadata_path = out_dir / "metadata.json"

    importance.to_csv(importance_path, index=False)
    corr_table.to_csv(corr_path, index=False)
    _distribution_summary(
        splits["train"].loc[valid_masks["train"]], shortlist
    ).to_csv(dist_path, index=False)
    with shortlist_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(shortlist) + "\n")

    training_dataset_outputs: Dict[str, str] = {}
    if config.export_training_dataset:
        train_ds_dir = out_dir / f"training_dataset_{train_feature_count}"
        train_ds_dir.mkdir(parents=True, exist_ok=True)
        target_model_col = f"{config.target_col}_model"
        for split in SPLITS:
            out_df = splits[split].loc[
                valid_masks[split], training_features + [config.target_col]
            ].copy()
            if config.log_target:
                out_df[target_model_col] = _transform_target(
                    pd.to_numeric(out_df[config.target_col], errors="coerce"),
                    log_target=True,
                )
            out_path = train_ds_dir / f"{split}.parquet"
            out_df.to_parquet(out_path, index=False)
            training_dataset_outputs[split] = str(out_path.resolve())

    metadata = {
        "dataset": dataset,
        "variant": variant,
        "input_data_dir": str(data_dir.resolve()),
        "target_col": config.target_col,
        "model_family": "xgboost" if XGBRegressor is not None else "random_forest",
        "feature_count_numeric": len(features),
        "shortlist_count": len(shortlist),
        "top_k_requested": config.top_k,
        "train_feature_count_requested": config.train_feature_count,
        "training_feature_count": train_feature_count,
        "training_features": training_features,
        "log_target": config.log_target,
        "drop_invalid_target": config.drop_invalid_target,
        "invalid_target_rows_dropped": invalid_counts,
        "metrics": metrics,
        "outputs": {
            "feature_importance_csv": str(importance_path.resolve()),
            "feature_corr_target_csv": str(corr_path.resolve()),
            "distribution_summary_train_csv": str(dist_path.resolve()),
            "shortlist_txt": str(shortlist_path.resolve()),
        },
    }
    if training_dataset_outputs:
        metadata["outputs"]["training_dataset"] = training_dataset_outputs
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def run(config: EDAConfig) -> None:
    targets = _discover_targets(config.input_root, config.dataset_glob, config.variant_glob)
    summaries = [_run_one(config, d, v, p) for d, v, p in targets]
    print("EDA complete.")
    for s in summaries:
        val = s["metrics"]["val"]
        print(
            f"{s['dataset']}/{s['variant']} "
            f"features={s['feature_count_numeric']} "
            f"shortlist={s['shortlist_count']} "
            f"val_rmse={val['rmse']:.6g} "
            f"out={config.output_root / s['dataset'] / s['variant']}"
        )


if __name__ == "__main__":
    run(_parse_args())
