"""
Crawl Darshan parquet datasets and report feature support statistics.

This script is intended for EDA in `specs/eda.md`:
- account for every parquet file under totals/detail roots
- summarize features seen
- count filled (non-NaN) entries
- report majority value when feasible (low-cardinality columns)

Usage examples:
  python src/explore_feature_support.py --dataset-root /path/to/darshan_share
  python src/explore_feature_support.py --total-root data --detail-root data/2021
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ColumnStats:
    files_seen: int = 0
    rows_seen: int = 0
    non_null: int = 0
    null: int = 0
    dtype_counts: Counter[str] = field(default_factory=Counter)
    # Track exact value counts only while cardinality stays manageable.
    value_counts: Counter[str] = field(default_factory=Counter)
    value_count_capped: bool = False


def _collect_parquets_total(root: Path) -> list[Path]:
    if not root.exists():
        return []
    # Typical layout: darshan_total/App*.parquet
    app_files = sorted(root.glob("App*.parquet"))
    if app_files:
        return app_files
    # Fallback: any parquet in the directory.
    return sorted(root.glob("*.parquet"))


def _collect_parquets_detail(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob("**/*.parquet"))


def _drop_for_leakage(name: str, drop_time_and_bytes: bool) -> bool:
    if not drop_time_and_bytes:
        return False
    up = name.upper()
    return ("BYTES" in up) or ("TIME" in up)


def _update_value_counts(
    stats: ColumnStats,
    s_non_null: pd.Series,
    max_distinct_values: int,
) -> None:
    if stats.value_count_capped:
        return
    vc = s_non_null.value_counts(dropna=True)
    # Estimate whether adding all values would exceed cardinality budget.
    unseen = [k for k in vc.index if str(k) not in stats.value_counts]
    if len(stats.value_counts) + len(unseen) > max_distinct_values:
        stats.value_count_capped = True
        stats.value_counts.clear()
        return
    for k, v in vc.items():
        stats.value_counts[str(k)] += int(v)


def _scan_files(
    paths: list[Path],
    source: str,
    drop_time_and_bytes: bool,
    max_distinct_values: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    per_col: dict[str, ColumnStats] = defaultdict(ColumnStats)
    files_scanned = 0
    rows_total = 0

    for p in paths:
        df = pd.read_parquet(p)
        files_scanned += 1
        n_rows = len(df)
        rows_total += n_rows

        for col in df.columns:
            if _drop_for_leakage(col, drop_time_and_bytes):
                continue

            s = df[col]
            cs = per_col[col]
            cs.files_seen += 1
            cs.rows_seen += n_rows
            cs.dtype_counts[str(s.dtype)] += 1

            non_null = int(s.notna().sum())
            cs.non_null += non_null
            cs.null += int(n_rows - non_null)

            if non_null > 0:
                _update_value_counts(cs, s.dropna(), max_distinct_values=max_distinct_values)

    records: list[dict[str, Any]] = []
    for col, cs in per_col.items():
        if cs.value_count_capped:
            majority_value = "<capped_high_cardinality>"
            majority_count = None
        elif cs.value_counts:
            majority_value, majority_count_raw = cs.value_counts.most_common(1)[0]
            majority_count = int(majority_count_raw)
        else:
            majority_value = None
            majority_count = 0

        fill_rate = (cs.non_null / cs.rows_seen) if cs.rows_seen else 0.0
        coverage_rate_global = (cs.non_null / rows_total) if rows_total else 0.0
        majority_share = (
            (majority_count / cs.non_null)
            if (majority_count is not None and cs.non_null > 0)
            else None
        )

        records.append(
            {
                "source": source,
                "feature": col,
                "files_seen": cs.files_seen,
                "rows_seen_where_column_present": cs.rows_seen,
                "non_null_count": cs.non_null,
                "null_count": cs.null,
                "fill_rate_when_present": fill_rate,
                "coverage_rate_global_rows": coverage_rate_global,
                "dtype_observed": ";".join(
                    f"{k}:{v}" for k, v in cs.dtype_counts.most_common()
                ),
                "majority_value": majority_value,
                "majority_count": majority_count,
                "majority_share_of_non_null": majority_share,
                "majority_value_capped": cs.value_count_capped,
            }
        )

    out_df = pd.DataFrame.from_records(records)
    if not out_df.empty:
        out_df = out_df.sort_values(
            by=["non_null_count", "fill_rate_when_present", "feature"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    meta = {
        "source": source,
        "files_scanned": files_scanned,
        "rows_scanned": rows_total,
        "features_reported": int(len(out_df)),
        "drop_time_and_bytes": drop_time_and_bytes,
        "max_distinct_values_for_majority": max_distinct_values,
    }
    return out_df, meta


def _resolve_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.dataset_root is not None:
        base = args.dataset_root
        return base / "darshan_total", base / "darshan_detail"
    if args.total_root is None or args.detail_root is None:
        raise SystemExit(
            "Provide either --dataset-root OR both --total-root and --detail-root."
        )
    return args.total_root, args.detail_root


def main() -> None:
    p = argparse.ArgumentParser(description="Crawl parquet feature support statistics.")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Root with darshan_total/ and darshan_detail/ subdirs.",
    )
    p.add_argument(
        "--total-root",
        type=Path,
        default=None,
        help="Path to totals directory (App*.parquet).",
    )
    p.add_argument(
        "--detail-root",
        type=Path,
        default=None,
        help="Path to detail directory (YYYY/M/D/jobid-*.parquet).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out") / "eda",
        help="Directory for CSV/JSON outputs.",
    )
    p.add_argument(
        "--drop-time-and-bytes",
        action="store_true",
        help="Drop columns whose names contain TIME or BYTES (leakage heuristic in eda.md).",
    )
    p.add_argument(
        "--max-distinct-values-for-majority",
        type=int,
        default=1000,
        help="Cap for exact majority-value tracking per feature.",
    )
    args = p.parse_args()

    total_root, detail_root = _resolve_roots(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    total_files = _collect_parquets_total(total_root)
    detail_files = _collect_parquets_detail(detail_root)

    if not total_files:
        raise SystemExit(f"No total parquet files found under: {total_root}")
    if not detail_files:
        raise SystemExit(f"No detail parquet files found under: {detail_root}")

    total_df, total_meta = _scan_files(
        total_files,
        source="total",
        drop_time_and_bytes=args.drop_time_and_bytes,
        max_distinct_values=args.max_distinct_values_for_majority,
    )
    detail_df, detail_meta = _scan_files(
        detail_files,
        source="detail",
        drop_time_and_bytes=args.drop_time_and_bytes,
        max_distinct_values=args.max_distinct_values_for_majority,
    )

    combined_df = pd.concat([total_df, detail_df], ignore_index=True)
    if not combined_df.empty:
        combined_df = combined_df.sort_values(
            by=["non_null_count", "fill_rate_when_present", "feature"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    total_csv = out_dir / "feature_support_total.csv"
    detail_csv = out_dir / "feature_support_detail.csv"
    combined_csv = out_dir / "feature_support_all.csv"
    meta_json = out_dir / "feature_support_meta.json"

    total_df.to_csv(total_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)
    combined_df.to_csv(combined_csv, index=False)

    meta = {
        "total": total_meta,
        "detail": detail_meta,
        "paths": {
            "total_root": str(total_root),
            "detail_root": str(detail_root),
            "total_files_found": len(total_files),
            "detail_files_found": len(detail_files),
        },
        "outputs": {
            "total_csv": str(total_csv),
            "detail_csv": str(detail_csv),
            "combined_csv": str(combined_csv),
        },
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[ok] total files scanned:  {total_meta['files_scanned']}")
    print(f"[ok] detail files scanned: {detail_meta['files_scanned']}")
    print(f"[ok] wrote: {total_csv}")
    print(f"[ok] wrote: {detail_csv}")
    print(f"[ok] wrote: {combined_csv}")
    print(f"[ok] wrote: {meta_json}")


if __name__ == "__main__":
    main()
