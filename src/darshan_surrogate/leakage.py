"""
Leakage guards for I/O intensity surrogate (specs/design.md §6).

Direct leakage: target components and any column that fixes job duration without modelling.
Proxy leakage: other features nearly collinear with ``y`` on the training split (fit on train only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# `total_*` counters that define the target numerator — never in X (design.md §6).
LEAKAGE_TOTAL_COUNTERS: frozenset[str] = frozenset(
    {
        "total_POSIX_BYTES_READ",
        "total_POSIX_BYTES_WRITTEN",
    }
)

# Never allow these as columns in X (target parts, duration, or raw time that encodes duration/order).
FORBIDDEN_IN_X: frozenset[str] = frozenset(
    {
        "total_POSIX_BYTES_READ",
        "total_POSIX_BYTES_WRITTEN",
        "runtime",
        "end_time",
        "start_time",
    }
)

# Prefixes for correlation-based proxy-leakage drop (train fit only).
_CORR_FILTER_PREFIXES = ("total_", "detail_")


def assert_no_forbidden_columns(columns: pd.Index | list[str]) -> None:
    """Raise if any forbidden column name appears in the feature matrix."""
    cols = set(columns)
    bad = FORBIDDEN_IN_X & cols
    if bad:
        raise ValueError(f"Leakage: forbidden columns present in X: {sorted(bad)}")


def columns_for_correlation_filter(columns: pd.Index | list[str]) -> list[str]:
    """Numeric total_/detail_ candidates; exclude month_*, app_id_code, missing_detail if needed."""
    out: list[str] = []
    for c in columns:
        if c.startswith("month_") or c == "app_id_code":
            continue
        if c.startswith(_CORR_FILTER_PREFIXES):
            out.append(c)
    return out


def drop_correlated_with_y(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    candidate_cols: list[str],
    max_abs_corr: float,
) -> tuple[list[str], list[dict]]:
    """
    On the **training** split only, drop candidates with |corr(x, y)| > max_abs_corr.

    Returns (columns_to_drop, records for CSV).
    """
    y = np.asarray(y_train, dtype=float)
    mask = np.isfinite(y)
    y_fit = y[mask]
    to_drop: list[str] = []
    records: list[dict] = []

    for c in candidate_cols:
        if c not in X_train.columns:
            continue
        x = pd.to_numeric(X_train[c], errors="coerce").fillna(0.0).values[mask]
        if np.std(x) < 1e-18:
            continue
        if len(y_fit) < 3:
            continue
        r = np.corrcoef(x.astype(float), y_fit)[0, 1]
        if not np.isfinite(r):
            continue
        if abs(r) > max_abs_corr:
            to_drop.append(c)
            records.append({"column": c, "abs_corr_with_y_train": abs(float(r)), "corr_with_y_train": float(r)})

    return to_drop, records


def apply_correlation_leakage_filter(
    X: pd.DataFrame,
    train_mask: np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    max_abs_corr: float,
) -> tuple[pd.DataFrame, list[dict]]:
    """Drop proxy-leakage columns from all rows; statistics from train only."""
    y_train = np.asarray(y)[train_mask]
    X_train = X.iloc[train_mask]
    candidates = columns_for_correlation_filter(X.columns)
    to_drop, records = drop_correlated_with_y(
        X_train,
        y_train,
        candidate_cols=candidates,
        max_abs_corr=max_abs_corr,
    )
    if not to_drop:
        return X, records
    return X.drop(columns=to_drop, errors="ignore"), records


def apply_name_substring_prefilter(
    X: pd.DataFrame,
    substrings: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop feature columns whose names contain any provided substring (case-insensitive).

    Intended for coarse EDA passes (e.g., dropping '*BYTES*'/'*TIME*' families)
    before model fitting.
    """
    keys = [s.upper() for s in substrings if s and s.strip()]
    if not keys:
        return X, []

    to_drop: list[str] = []
    for c in X.columns:
        cu = c.upper()
        if any(k in cu for k in keys):
            to_drop.append(c)

    if not to_drop:
        return X, []
    return X.drop(columns=to_drop, errors="ignore"), to_drop


def apply_timestamp_prefilter(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Always drop timestamp-like feature columns.

    This is a conservative guardrail against direct/near-direct temporal leakage
    from module timestamp counters (e.g. *_TIMESTAMP family), while keeping other
    non-target counters (including non-POSIX BYTES/TIME aggregates).
    """
    return apply_name_substring_prefilter(X, ["TIMESTAMP"])
