"""Time-ordered train/validation split (design.md §6)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def time_ordered_split(
    df: pd.DataFrame,
    train_frac: float = 0.8,
    time_col: str = "start_time",
) -> tuple[np.ndarray, np.ndarray]:
    """Return train and validation **positional** index arrays (sorted by time)."""
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    order = np.argsort(df[time_col].values)
    n = len(order)
    cut = int(n * train_frac)
    if cut < 1 or cut >= n:
        raise ValueError("Not enough rows for time split")
    train_pos = order[:cut]
    val_pos = order[cut:]
    return train_pos, val_pos
