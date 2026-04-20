"""Load `darshan_total` Parquet files with pandas only (design.md §13)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_totals_with_app_id(paths: list[Path | str]) -> pd.DataFrame:
    """Read each App*.parquet and add `app_id` from filename stem (e.g. App10)."""
    parts = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(p)
        d = pd.read_parquet(p)
        d = d.copy()
        d["app_id"] = p.stem
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
