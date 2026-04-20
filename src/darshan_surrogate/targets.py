"""POSIX I/O intensity target (design.md §4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

POSIX_READ = "total_POSIX_BYTES_READ"
POSIX_WRITE = "total_POSIX_BYTES_WRITTEN"


def posix_intensity(df: pd.DataFrame) -> pd.Series:
    """y = (POSIX read + POSIX write) / runtime [bytes/s]. Invalid rows → NaN."""
    rb = df[POSIX_READ].fillna(0.0).astype(float)
    wb = df[POSIX_WRITE].fillna(0.0).astype(float)
    posix = rb + wb
    rt = df["runtime"].astype(float)
    y = posix / rt.replace(0, np.nan)
    return y


def log1p_target(y: pd.Series | np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(y, dtype=float))


def expm1_predictions(z: np.ndarray) -> np.ndarray:
    out = np.expm1(np.asarray(z, dtype=float))
    return np.maximum(out, 0.0)
