"""Join `darshan_detail` aggregates onto total rows (design.md §3.3, §5.5)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None


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
    # Stream batches per shard to keep peak memory bounded.
    file_set: set[str] = set()
    n_rec = 0
    rec_sum = 0.0
    rec_sq_sum = 0.0
    var_sum = 0.0
    var_n = 0

    need_cols = ["filename", "POSIX_BYTES_READ", "POSIX_BYTES_WRITTEN", "POSIX_F_VARIANCE_RANK_BYTES"]

    for p in paths:
        if pq is None:
            d = pd.read_parquet(p)
            batches = [d]
        else:
            pf = pq.ParquetFile(p)
            avail = set(pf.schema.names)
            cols = [c for c in need_cols if c in avail]
            batches = (b.to_pandas() for b in pf.iter_batches(columns=cols, batch_size=65536))

        for b in batches:
            if "filename" in b.columns:
                fn = b["filename"].dropna()
                if len(fn):
                    file_set.update(fn.astype(str).tolist())

            br = pd.to_numeric(b.get("POSIX_BYTES_READ", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            bw = pd.to_numeric(b.get("POSIX_BYTES_WRITTEN", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            rec = br + bw
            if rec.size:
                n_rec += int(rec.size)
                rec_sum += float(rec.sum())
                rec_sq_sum += float(np.square(rec).sum())

            if "POSIX_F_VARIANCE_RANK_BYTES" in b.columns:
                vv = pd.to_numeric(b["POSIX_F_VARIANCE_RANK_BYTES"], errors="coerce").dropna().to_numpy(dtype=float)
                if vv.size:
                    var_sum += float(vv.sum())
                    var_n += int(vv.size)

    n_files = float(len(file_set)) if file_set else np.nan
    mean = (rec_sum / n_rec) if n_rec else 0.0
    var = max((rec_sq_sum / n_rec) - (mean * mean), 0.0) if n_rec else 0.0
    std = float(np.sqrt(var))
    cv = std / mean if mean > 1e-12 else 0.0
    var_mean = (var_sum / var_n) if var_n else np.nan
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
