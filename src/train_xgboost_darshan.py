'''

# On an Expanse login or compute node
module load cpu
module load anaconda3

conda activate darshan_ml          # (the env you created earlier)

pip install darshan optuna         # PyDarshan + hyperparameter optimizer
pip install pyarrow                # faster Parquet I/O (optional but nice)


'''


import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
from pathlib import Path

# ====================== CONFIG ======================
LOG_DIR = "/path/to/your/unpacked/darshan/logs"   # ← CHANGE THIS (e.g. after unpacking one day's .tar.gz)
MAX_LOGS = 500                                     # start small for testing; increase later
RANDOM_SEED = 42
# ===================================================

def extract_basic_features(log_path: str) -> dict:
    """Extract job-level features from a single .darshan file.
    Expand this function later with stride patterns, DXT traces, etc."""
    try:
        with darshan.DarshanLog(log_path) as log:
            # Job metadata
            job = log.job_data
            features = {
                "nprocs": int(job.get("nprocs", 0)),
                "runtime": float(job.get("runtime", 0.0)),
                "num_files": 0,
                "total_opens": 0,
                "total_reads": 0,
                "total_writes": 0,
                "total_read_bytes": 0,
                "total_write_bytes": 0,
            }

            # POSIX module (main source of filesystem counters)
            if "POSIX" in log.modules:
                # Modern PyDarshan supports format="pandas"
                posix_df = log.get_module_records("POSIX", format="pandas")
                if isinstance(posix_df, pd.DataFrame) and not posix_df.empty:
                    features["num_files"] = int(posix_df["id"].nunique())
                    features["total_opens"] = int(posix_df["POSIX_OPENS"].sum())
                    features["total_reads"] = int(posix_df["POSIX_READS"].sum())
                    features["total_writes"] = int(posix_df["POSIX_WRITES"].sum())
                    features["total_read_bytes"] = float(posix_df["POSIX_READ_BYTES"].sum())
                    features["total_write_bytes"] = float(posix_df["POSIX_WRITE_BYTES"].sum())

            # Derived surrogate TARGET: I/O intensity (bytes per second of runtime)
            # This is what the model will learn to predict
            total_bytes = features["total_read_bytes"] + features["total_write_bytes"]
            runtime = features["runtime"]
            features["io_intensity"] = total_bytes / runtime if runtime > 0 else 0.0

            return features
    except Exception as e:
        print(f"Skipping {log_path}: {e}")
        return None

# ====================== DATA LOADING ======================
print("Loading Darshan logs...")
log_files = sorted(glob.glob(os.path.join(LOG_DIR, "**/*.darshan"), recursive=True))[:MAX_LOGS]
print(f"Found {len(log_files)} logs — using first {len(log_files)}")

records = []
for path in log_files:
    feat = extract_basic_features(path)
    if feat is not None:
        records.append(feat)

df = pd.DataFrame(records)
print(f"Final dataset: {len(df)} jobs × {len(df.columns)} columns")
print(df.head())

# ====================== TRAIN / VAL SPLIT ======================
# For real surrogate work you should use TIME-ORDERED split (earlier → later logs)
# Here we use random for quick demo; replace with chronological when ready
X = df.drop(columns=["io_intensity"])          # all features
y = df["io_intensity"]                         # target we want to predict

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# ====================== BASELINE MODEL ======================
print("\nTraining baseline XGBoost...")
baseline = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    early_stopping_rounds=50,
    eval_metric="rmse"
)

baseline.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

pred = baseline.predict(X_val)
print(f"Baseline RMSE: {mean_squared_error(y_val, pred, squared=False):.4f}")
print(f"Baseline R²:   {r2_score(y_val, pred):.4f}")

# ====================== OPTUNA HYPERPARAMETER SWEEP ======================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": RANDOM_SEED,
        "early_stopping_rounds": 50,
        "eval_metric": "rmse",
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_val = model.predict(X_val)
    return mean_squared_error(y_val, pred_val, squared=False)

print("\nStarting Optuna sweep (20 trials)...")
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20, show_progress_bar=True)

print(f"\nBest hyperparameters: {study.best_params}")
print(f"Best RMSE: {study.best_value:.4f}")

# Train final model with best params
best_params = study.best_params
best_params.update({
    "random_state": RANDOM_SEED,
    "early_stopping_rounds": 50,
    "eval_metric": "rmse"
})
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# ====================== SAVE & INSPECT ======================
final_model.save_model("polaris_xgboost_surrogate.json")
print("\nFinal model saved as polaris_xgboost_surrogate.json")
print("Top 10 most important features:")
importances = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(importances.head(10))

# Quick prediction example
print("\nExample prediction (first validation job):")
print(f"True io_intensity: {y_val.iloc[0]:.2f}")
print(f"Predicted:         {final_model.predict(X_val.iloc[[0]])[0]:.2f}")
