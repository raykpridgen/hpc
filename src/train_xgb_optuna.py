#!/usr/bin/env python3
"""Train XGBoost surrogate models with Optuna tuning."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


SPLITS = ("train", "val", "test")
TARGET_MODES = ("raw", "log1p")


@dataclass(frozen=True)
class TrainConfig:
    input_root: Path
    output_root: Path
    dataset_glob: str
    variants: Tuple[str, ...]
    target_modes: Tuple[str, ...]
    target_col: str
    train_feature_count: int
    n_trials: int
    timeout_sec: int
    seed: int
    early_stopping_rounds: int
    n_jobs: int
    sweep_profile: str
    fast_mode: bool
    skip_missing_variants: bool


def _parse_csv(raw: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train/tune XGBoost models from EDA-exported training datasets with "
            "support for raw and log1p target modes."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("out/eda/totals"),
        help="Root containing EDA outputs by dataset/variant.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("out/models/totals"),
        help="Root to write model artifacts.",
    )
    parser.add_argument(
        "--dataset-glob",
        type=str,
        default="*",
        help="Dataset glob under input root.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="balanced",
        help="Comma-separated variants to train.",
    )
    parser.add_argument(
        "--target-modes",
        type=str,
        default="log1p",
        help="Comma-separated target modes: raw,log1p.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="io_intensity",
        help="Target column.",
    )
    parser.add_argument(
        "--train-feature-count",
        type=int,
        default=30,
        help="Feature count for training_dataset_<N> discovery.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=40,
        help="Optuna trials per dataset/variant/target-mode run.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=2400,
        help="Max optimization walltime per run (seconds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=80,
        help="Patience for early stopping on validation set.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="XGBoost parallel thread count.",
    )
    parser.add_argument(
        "--sweep-profile",
        type=str,
        default="expanded_safe",
        choices=("baseline", "expanded_safe"),
        help=(
            "Hyperparameter search profile. expanded_safe widens space while "
            "remaining compute-conscious."
        ),
    )
    parser.add_argument(
        "--fast-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run only balanced + log1p with reduced trial count for faster turnaround."
        ),
    )
    parser.add_argument(
        "--skip-missing-variants",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip missing dataset/variant folders instead of failing.",
    )
    args = parser.parse_args()

    variants = _parse_csv(args.variants)
    target_modes = _parse_csv(args.target_modes)
    if args.fast_mode:
        variants = ("balanced",)
        target_modes = ("log1p",)
        n_trials = min(args.n_trials, 12)
        timeout_sec = min(args.timeout_sec, 900)
        early_stopping_rounds = min(args.early_stopping_rounds, 50)
        sweep_profile = "baseline"
    else:
        n_trials = args.n_trials
        timeout_sec = args.timeout_sec
        early_stopping_rounds = args.early_stopping_rounds
        sweep_profile = args.sweep_profile

    for mode in target_modes:
        if mode not in TARGET_MODES:
            raise ValueError(f"Unsupported target mode '{mode}'. Use raw/log1p.")
    if args.train_feature_count <= 0:
        raise ValueError("--train-feature-count must be > 0.")
    if n_trials <= 0:
        raise ValueError("--n-trials must be > 0.")
    if timeout_sec <= 0:
        raise ValueError("--timeout-sec must be > 0.")
    if early_stopping_rounds <= 0:
        raise ValueError("--early-stopping-rounds must be > 0.")
    if args.n_jobs <= 0:
        raise ValueError("--n-jobs must be > 0.")

    return TrainConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        dataset_glob=args.dataset_glob,
        variants=variants,
        target_modes=target_modes,
        target_col=args.target_col,
        train_feature_count=args.train_feature_count,
        n_trials=n_trials,
        timeout_sec=timeout_sec,
        seed=args.seed,
        early_stopping_rounds=early_stopping_rounds,
        n_jobs=args.n_jobs,
        sweep_profile=sweep_profile,
        fast_mode=args.fast_mode,
        skip_missing_variants=args.skip_missing_variants,
    )


def _discover_runs(config: TrainConfig) -> List[Tuple[str, str, Path]]:
    runs: List[Tuple[str, str, Path]] = []
    for dataset_dir in sorted(p for p in config.input_root.glob(config.dataset_glob) if p.is_dir()):
        for variant in config.variants:
            var_dir = dataset_dir / variant
            if not var_dir.exists():
                if config.skip_missing_variants:
                    continue
                raise FileNotFoundError(f"Missing variant dir: {var_dir}")
            data_dir = var_dir / f"training_dataset_{config.train_feature_count}"
            if not data_dir.exists():
                if config.skip_missing_variants:
                    continue
                raise FileNotFoundError(f"Missing training dataset dir: {data_dir}")
            if all((data_dir / f"{s}.parquet").exists() for s in SPLITS):
                runs.append((dataset_dir.name, variant, data_dir))
    if not runs:
        raise FileNotFoundError(
            "No trainable datasets found. Ensure EDA exported training_dataset_<N> outputs."
        )
    return runs


def _load_splits(data_dir: Path) -> Dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(data_dir / f"{s}.parquet") for s in SPLITS}


def _feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    drop = {target_col, f"{target_col}_model"}
    return [c for c in numeric if c not in drop]


def _transform_target(y_raw: pd.Series, mode: str) -> pd.Series:
    if mode == "log1p":
        return np.log1p(y_raw)
    return y_raw


def _inverse_target(y_model: np.ndarray, mode: str) -> np.ndarray:
    if mode == "log1p":
        return np.expm1(y_model)
    return y_model


def _valid_mask(y: pd.Series, mode: str) -> pd.Series:
    valid = y.notna() & np.isfinite(y)
    if mode == "log1p":
        valid &= y > 0.0
    return valid


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_model(
    params: Dict[str, float | int | str],
    seed: int,
    n_jobs: int,
    early_stopping_rounds: int,
) -> XGBRegressor:
    tree_method = str(params.get("tree_method", "hist"))
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method=tree_method,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
        min_child_weight=float(params["min_child_weight"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        gamma=float(params["gamma"]),
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
        n_jobs=n_jobs,
    )


def _optuna_params(trial: optuna.Trial, profile: str) -> Dict[str, float | int | str]:
    if profile == "baseline":
        return {
            "tree_method": "hist",
            "n_estimators": trial.suggest_int("n_estimators", 250, 1400),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 25.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

    # Expanded but compute-safe profile.
    return {
        "tree_method": "hist",
        "n_estimators": trial.suggest_int("n_estimators", 400, 2400),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.25, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 11),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 30.0),
        "subsample": trial.suggest_float("subsample", 0.55, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-9, 30.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-9, 60.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
    }


def _fit_one(
    config: TrainConfig,
    dataset: str,
    variant: str,
    data_dir: Path,
    mode: str,
) -> Dict[str, object]:
    splits = _load_splits(data_dir)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    if config.target_col not in train_df.columns:
        raise KeyError(f"Target '{config.target_col}' missing in {data_dir}/train.parquet")

    features = _feature_columns(train_df, config.target_col)
    if not features:
        raise ValueError(f"No numeric feature columns in {data_dir}")

    y_train_raw = pd.to_numeric(train_df[config.target_col], errors="coerce")
    y_val_raw = pd.to_numeric(val_df[config.target_col], errors="coerce")
    y_test_raw = pd.to_numeric(test_df[config.target_col], errors="coerce")

    mask_train = _valid_mask(y_train_raw, mode)
    mask_val = _valid_mask(y_val_raw, mode)
    mask_test = _valid_mask(y_test_raw, mode)
    dropped_counts = {
        "train": int((~mask_train).sum()),
        "val": int((~mask_val).sum()),
        "test": int((~mask_test).sum()),
    }

    x_train = train_df.loc[mask_train, features]
    x_val = val_df.loc[mask_val, features]
    x_test = test_df.loc[mask_test, features]

    y_train_fit = _transform_target(y_train_raw.loc[mask_train], mode)
    y_val_fit = _transform_target(y_val_raw.loc[mask_val], mode)

    y_train_eval = y_train_raw.loc[mask_train].to_numpy()
    y_val_eval = y_val_raw.loc[mask_val].to_numpy()
    y_test_eval = y_test_raw.loc[mask_test].to_numpy()

    if len(y_train_fit) == 0 or len(y_val_fit) == 0:
        raise ValueError(f"Insufficient valid train/val rows for {dataset}/{variant}/{mode}.")

    start = time.time()
    sampler = optuna.samplers.TPESampler(seed=config.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        params = _optuna_params(trial, config.sweep_profile)
        model = _build_model(
            params,
            seed=config.seed,
            n_jobs=config.n_jobs,
            early_stopping_rounds=config.early_stopping_rounds,
        )
        model.fit(
            x_train,
            y_train_fit,
            eval_set=[(x_val, y_val_fit)],
            verbose=False,
        )
        pred_model = model.predict(x_val)
        pred = _inverse_target(np.asarray(pred_model), mode)
        return float(np.sqrt(mean_squared_error(y_val_eval, pred)))

    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout_sec)
    best = study.best_params
    final_model = _build_model(
        best,
        seed=config.seed,
        n_jobs=config.n_jobs,
        early_stopping_rounds=config.early_stopping_rounds,
    )
    final_model.fit(
        x_train,
        y_train_fit,
        eval_set=[(x_val, y_val_fit)],
        verbose=False,
    )

    preds = {
        "train": _inverse_target(np.asarray(final_model.predict(x_train)), mode),
        "val": _inverse_target(np.asarray(final_model.predict(x_val)), mode),
        "test": _inverse_target(np.asarray(final_model.predict(x_test)), mode),
    }
    metrics = {
        "train": _metrics(y_train_eval, preds["train"]),
        "val": _metrics(y_val_eval, preds["val"]),
        "test": _metrics(y_test_eval, preds["test"]),
    }

    baseline_ref = float(np.nanmean(y_train_eval))
    baseline_metrics = {
        "val": _metrics(y_val_eval, np.full_like(y_val_eval, baseline_ref, dtype=float)),
        "test": _metrics(y_test_eval, np.full_like(y_test_eval, baseline_ref, dtype=float)),
        "baseline_reference": baseline_ref,
    }

    out_dir = config.output_root / dataset / variant / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    final_model.get_booster().save_model(str(out_dir / "xgboost_model.json"))
    trials_df = study.trials_dataframe()
    trials_df.to_csv(out_dir / "optuna_trials.csv", index=False)
    with (out_dir / "best_params.json").open("w", encoding="utf-8") as handle:
        json.dump(best, handle, indent=2)

    for split_name, y_true, y_pred in (
        ("train", y_train_eval, preds["train"]),
        ("val", y_val_eval, preds["val"]),
        ("test", y_test_eval, preds["test"]),
    ):
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_parquet(
            out_dir / f"predictions_{split_name}.parquet",
            index=False,
        )

    elapsed = time.time() - start
    metadata = {
        "dataset": dataset,
        "variant": variant,
        "target_mode": mode,
        "input_data_dir": str(data_dir.resolve()),
        "target_col": config.target_col,
        "feature_count": len(features),
        "features": features,
        "valid_rows": {
            "train": int(len(y_train_eval)),
            "val": int(len(y_val_eval)),
            "test": int(len(y_test_eval)),
        },
        "invalid_rows_dropped": dropped_counts,
        "n_trials_requested": config.n_trials,
        "n_trials_completed": int(len(study.trials)),
        "timeout_sec": config.timeout_sec,
        "seed": config.seed,
        "early_stopping_rounds": config.early_stopping_rounds,
        "n_jobs": config.n_jobs,
        "sweep_profile": config.sweep_profile,
        "elapsed_sec": elapsed,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "outputs": {
            "model_json": str((out_dir / "xgboost_model.json").resolve()),
            "best_params_json": str((out_dir / "best_params.json").resolve()),
            "trials_csv": str((out_dir / "optuna_trials.csv").resolve()),
            "pred_train": str((out_dir / "predictions_train.parquet").resolve()),
            "pred_val": str((out_dir / "predictions_val.parquet").resolve()),
            "pred_test": str((out_dir / "predictions_test.parquet").resolve()),
        },
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def run(config: TrainConfig) -> None:
    runs = _discover_runs(config)
    results: List[Dict[str, object]] = []
    for dataset, variant, data_dir in runs:
        for mode in config.target_modes:
            results.append(_fit_one(config, dataset, variant, data_dir, mode))

    print("Training complete.")
    for r in results:
        print(
            f"{r['dataset']}/{r['variant']}/{r['target_mode']} "
            f"val_rmse={r['metrics']['val']['rmse']:.6g} "
            f"test_rmse={r['metrics']['test']['rmse']:.6g} "
            f"trials={r['n_trials_completed']} "
            f"time_sec={r['elapsed_sec']:.1f}"
        )


if __name__ == "__main__":
    run(_parse_args())
