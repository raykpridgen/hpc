"""Feature matrix: totals + calendar one-hot + app_id encoding (design.md §5–§6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .leakage import LEAKAGE_TOTAL_COUNTERS

# Alias for feature selection: drop these POSIX total counters from total_* (design.md §6).
LEAKAGE_COLS = LEAKAGE_TOTAL_COUNTERS

META_EXCLUDE = frozenset({"uid", "exe", "jobid"})


def add_calendar_one_hot(df: pd.DataFrame, time_col: str = "start_time") -> pd.DataFrame:
    """One-hot encode calendar month (1–12) as month_1 … month_12."""
    out = df.copy()
    ts = pd.to_datetime(out[time_col], unit="s")
    month = ts.dt.month.astype(int)
    for m in range(1, 13):
        out[f"month_{m}"] = (month == m).astype(int)
    return out


def candidate_total_columns(df: pd.DataFrame) -> list[str]:
    """All total_* columns minus leakage set."""
    return [c for c in df.columns if c.startswith("total_") and c not in LEAKAGE_COLS]


def drop_constant_columns(train: pd.DataFrame, cols: list[str], eps: float = 1e-18) -> list[str]:
    """Keep numeric columns with variance > eps on training split."""
    keep: list[str] = []
    for c in cols:
        if c not in train.columns:
            continue
        s = train[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        v = s.var(ddof=0)
        if pd.isna(v) or v > eps:
            keep.append(c)
    return keep


def build_feature_matrix(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    *,
    drop_sparse_frac: float | None = None,
) -> tuple[pd.DataFrame, LabelEncoder, list[str]]:
    """
    Returns X (numeric + month one-hot + app_id encoded), fitted LabelEncoder for app_id,
    and list of feature column names used.
    """
    train = df.iloc[train_mask]
    df2 = add_calendar_one_hot(df)

    totals = candidate_total_columns(df2)
    totals = drop_constant_columns(train, totals)

    if drop_sparse_frac is not None:
        n = len(train)
        sparse = []
        for c in totals:
            z = (train[c].fillna(0) == 0).mean()
            if z > drop_sparse_frac:
                sparse.append(c)
        totals = [c for c in totals if c not in sparse]

    extra_num = [c for c in df2.columns if c.startswith("detail_") and c != "missing_detail"]
    if "missing_detail" in df2.columns:
        extra_num.append("missing_detail")

    month_cols = [f"month_{m}" for m in range(1, 13)]

    le = LabelEncoder()
    le.fit(df2["app_id"].astype(str))
    app_code = le.transform(df2["app_id"].astype(str)).astype(np.float64)

    use_cols = [c for c in totals + month_cols + extra_num if c in df2.columns]
    X = df2[use_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    X = pd.concat(
        [X, pd.Series(app_code, index=X.index, name="app_id_code")],
        axis=1,
    )

    feature_names = list(X.columns)
    return X, le, feature_names
