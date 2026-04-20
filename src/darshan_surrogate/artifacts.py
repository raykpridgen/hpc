"""CSV exports under `out/` (design.md §11.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_join_stats(path: Path, stats: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(path, index=False)


def save_y_eda(path: Path, y: np.ndarray | pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yv = np.asarray(y, dtype=float)
    yv = yv[np.isfinite(yv)]
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    pct = np.percentile(yv, qs)
    out = pd.DataFrame({"percentile": qs, "y_value": pct})
    out.to_csv(path, index=False)


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)
