"""Join `darshan_detail` aggregates onto total rows (design.md §3.3, §5.5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _detail_day_dir(detail_root: Path, start_time: int) -> Path:
    t = pd.Timestamp(start_time, unit="s")
    return detail_root / str(t.year) / str(t.month) / str(t.day)


def list_detail_shards(detail_root: Path, start_time: int, jobid: int) -> list[Path]:
    day = _detail_day_dir(detail_root, start_time)
    if not day.is_dir():
        return []
    return sorted(day.glob(f"{jobid}-*.parquet"))


def _aggregate_detail_frames(paths: list[Path]) -> dict[str, float]:
    if not paths:
        return {
            "detail_n_files": np.nan,
            "detail_posix_cv": np.nan,
            "detail_posix_var_mean": np.nan,
        }
    dfs = [pd.read_parquet(p) for p in paths]
    d = pd.concat(dfs, ignore_index=True)
    n_files = float(d["filename"].nunique()) if "filename" in d.columns else np.nan
    br = d["POSIX_BYTES_READ"].fillna(0.0) if "POSIX_BYTES_READ" in d.columns else pd.Series(0.0, index=d.index)
    bw = d["POSIX_BYTES_WRITTEN"].fillna(0.0) if "POSIX_BYTES_WRITTEN" in d.columns else pd.Series(0.0, index=d.index)
    rec = br.astype(float) + bw.astype(float)
    mean = float(rec.mean()) if len(rec) else 0.0
    std = float(rec.std(ddof=0)) if len(rec) else 0.0
    cv = std / mean if mean > 1e-12 else 0.0
    var_col = "POSIX_F_VARIANCE_RANK_BYTES"
    var_mean = float(d[var_col].mean()) if var_col in d.columns else np.nan
    return {
        "detail_n_files": n_files,
        "detail_posix_cv": float(cv),
        "detail_posix_var_mean": var_mean,
    }


def join_detail_features(
    df: pd.DataFrame,
    detail_root: Path | str,
    missing_threshold: float = 0.2,
) -> tuple[pd.DataFrame, dict]:
    """
    Add detail_* columns. If missing-detail rate >= missing_threshold, impute zeros
    and set missing_detail=1; else drop rows with no detail (design.md §3.3).
    """
    detail_root = Path(detail_root)
    out = df.copy()
    out["_dt"] = pd.to_datetime(out["start_time"], unit="s")
    out["_date"] = out["_dt"].dt.date

    detail_blocks: list[pd.DataFrame] = []
    for _, g in out.groupby(["jobid", "_date"], sort=False):
        st = int(g["start_time"].iloc[0])
        jid = int(g["jobid"].iloc[0])
        paths = list_detail_shards(detail_root, st, jid)
        feats = _aggregate_detail_frames(paths)
        sub = pd.DataFrame(index=g.index)
        for k, v in feats.items():
            sub[k] = v
        sub["_has_detail"] = len(paths) > 0
        detail_blocks.append(sub)

    extra = pd.concat(detail_blocks, axis=0).sort_index()
    out = pd.concat([out, extra], axis=1)

    no_detail = ~out["_has_detail"].astype(bool)
    missing_rate = float(no_detail.mean())

    join_stats: dict = {
        "n_rows": len(out),
        "n_missing_detail": int(no_detail.sum()),
        "missing_detail_rate": missing_rate,
        "fallback_impute": False,
    }

    if missing_rate >= missing_threshold:
        join_stats["fallback_impute"] = True
        out["missing_detail"] = no_detail.astype(int)
        for c in ["detail_n_files", "detail_posix_cv", "detail_posix_var_mean"]:
            out[c] = out[c].fillna(0.0)
    else:
        out["missing_detail"] = no_detail.astype(int)
        dropped = int(no_detail.sum())
        out = out.loc[~no_detail].copy()
        join_stats["n_dropped"] = dropped
        join_stats["rows_after_drop"] = len(out)

    out = out.drop(columns=["_dt", "_date", "_has_detail"], errors="ignore")
    return out, join_stats
