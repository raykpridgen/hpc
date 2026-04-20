"""XGBoost on log1p(y) with optional Optuna (design.md §8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

try:
    import optuna

    _HAS_OPTUNA = True
except ImportError:
    _HAS_OPTUNA = False


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(mean_squared_error(a, b, squared=False))
    except TypeError:
        return float(np.sqrt(mean_squared_error(a, b)))


def train_surrogate(
    X_train: pd.DataFrame,
    z_train: np.ndarray,
    X_val: pd.DataFrame,
    z_val: np.ndarray,
    *,
    n_optuna_trials: int = 0,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
) -> tuple[xgb.XGBRegressor, dict]:
    """Fit XGBRegressor on log1p target `z`. Returns model and validation metrics on `z` space."""
    if n_optuna_trials > 0:
        if not _HAS_OPTUNA:
            raise ImportError("optuna is required when n_optuna_trials > 0")
        return _train_with_optuna(
            X_train,
            z_train,
            X_val,
            z_val,
            n_trials=n_optuna_trials,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
        )

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",
    )
    model.fit(
        X_train,
        z_train,
        eval_set=[(X_val, z_val)],
        verbose=False,
    )
    z_pred = model.predict(X_val)
    metrics_z = {
        "rmse_z": _rmse(z_val, z_pred),
        "mae_z": float(mean_absolute_error(z_val, z_pred)),
        "r2_z": float(r2_score(z_val, z_pred)),
    }
    return model, metrics_z


def _train_with_optuna(
    X_train: pd.DataFrame,
    z_train: np.ndarray,
    X_val: pd.DataFrame,
    z_val: np.ndarray,
    *,
    n_trials: int,
    early_stopping_rounds: int,
    random_state: int,
) -> tuple[xgb.XGBRegressor, dict]:
    def objective(trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "random_state": random_state,
            "early_stopping_rounds": early_stopping_rounds,
            "eval_metric": "rmse",
        }
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            z_train,
            eval_set=[(X_val, z_val)],
            verbose=False,
        )
        z_pred = model.predict(X_val)
        return _rmse(z_val, z_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params.copy()
    best_params.update(
        {
            "random_state": random_state,
            "early_stopping_rounds": early_stopping_rounds,
            "eval_metric": "rmse",
        }
    )
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, z_train, eval_set=[(X_val, z_val)], verbose=False)
    z_pred = model.predict(X_val)
    metrics_z = {
        "rmse_z": _rmse(z_val, z_pred),
        "mae_z": float(mean_absolute_error(z_val, z_pred)),
        "r2_z": float(r2_score(z_val, z_pred)),
        "optuna_best_rmse": study.best_value,
        "n_trials": n_trials,
    }
    best_state["study"] = study
    return model, metrics_z


def metrics_on_original_scale(
    y_true: np.ndarray,
    z_pred: np.ndarray,
) -> dict[str, float]:
    """RMSE / MAE / R² on original intensity `y` given predictions in log1p space `z_pred`."""
    from .targets import expm1_predictions

    yp = expm1_predictions(z_pred)
    yt = np.asarray(y_true, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    return {
        "rmse_y": _rmse(yt, yp),
        "mae_y": float(mean_absolute_error(yt, yp)),
        "r2_y": float(r2_score(yt, yp)),
    }
