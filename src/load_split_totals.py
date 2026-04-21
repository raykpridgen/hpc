#!/usr/bin/env python3
"""Load totals parquets and build chronological train/val/test splits.

This module implements the loading/split phase only. It reads parquet files
from the totals directory, computes split assignments, and writes split parquet
artifacts plus metadata under an output directory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLIT_ORDER = [SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST]


@dataclass(frozen=True)
class SplitConfig:
    input_dir: Path
    output_dir: Path
    file_pattern: str
    start_time_col: str
    train_frac: float
    seed: int
    allow_datetime_fallback: bool


def _parse_args() -> SplitConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Load darshan_totals parquet files and generate chronological "
            "80/10/10 train/val/test split artifacts."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("darshan_share/darshan_totals"),
        help="Directory containing totals parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/splits/totals"),
        help="Output directory for split artifacts.",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="App*.parquet",
        help="Glob pattern for totals parquet files.",
    )
    parser.add_argument(
        "--start-time-col",
        type=str,
        default="start_time",
        help="Column used for chronology sort key.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction of earliest rows assigned to train.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for val/test assignment in the holdout pool.",
    )
    parser.add_argument(
        "--allow-datetime-fallback",
        action="store_true",
        help=(
            "Allow datetime parsing when start_time is not numeric. "
            "Disabled by default to prefer unix-like numeric timestamps."
        ),
    )
    args = parser.parse_args()

    if not (0.0 < args.train_frac < 1.0):
        raise ValueError("--train-frac must be in the open interval (0, 1).")

    return SplitConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        file_pattern=args.file_pattern,
        start_time_col=args.start_time_col,
        train_frac=args.train_frac,
        seed=args.seed,
        allow_datetime_fallback=args.allow_datetime_fallback,
    )


def _discover_files(input_dir: Path, pattern: str) -> List[Path]:
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError(
            f"No parquet files found in '{input_dir}' using pattern '{pattern}'."
        )
    return files


def _coerce_start_time(values: pd.Series, allow_datetime_fallback: bool) -> pd.Series:
    """Convert start_time values to numeric chronology key.

    Returns float64 nanoseconds for datetime-like values or numeric values as-is.
    NaN values are preserved for rows that cannot be coerced.
    """

    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    if numeric.notna().any() or not allow_datetime_fallback:
        return numeric

    dt = pd.to_datetime(values, errors="coerce", utc=True)
    as_ns = dt.view("int64").astype("float64")
    as_ns[dt.isna()] = np.nan
    return as_ns


def _build_manifest(
    files: List[Path], start_time_col: str, allow_datetime_fallback: bool
) -> pd.DataFrame:
    manifest_frames: List[pd.DataFrame] = []
    for src_path in files:
        frame = pd.read_parquet(src_path, columns=[start_time_col])
        frame = frame.reset_index(drop=True)
        frame["__source_file"] = src_path.name
        frame["__row_in_file"] = np.arange(len(frame), dtype=np.int64)
        frame["__sort_time"] = _coerce_start_time(
            frame[start_time_col], allow_datetime_fallback=allow_datetime_fallback
        )
        manifest_frames.append(
            frame[["__source_file", "__row_in_file", start_time_col, "__sort_time"]]
        )

    manifest = pd.concat(manifest_frames, axis=0, ignore_index=True)
    manifest["__sort_time_missing"] = manifest["__sort_time"].isna()
    manifest = manifest.sort_values(
        by=["__sort_time_missing", "__sort_time", "__source_file", "__row_in_file"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return manifest


def _assign_splits(
    manifest: pd.DataFrame, train_frac: float, seed: int
) -> pd.DataFrame:
    n_rows = len(manifest)
    n_train = int(np.floor(n_rows * train_frac))
    holdout_start = n_train
    holdout_indices = np.arange(holdout_start, n_rows, dtype=np.int64)

    n_holdout = len(holdout_indices)
    n_val = n_holdout // 2
    n_test = n_holdout - n_val

    rng = np.random.default_rng(seed)
    shuffled_holdout = holdout_indices.copy()
    rng.shuffle(shuffled_holdout)
    val_indices = shuffled_holdout[:n_val]
    test_indices = shuffled_holdout[n_val:]

    split = np.full(n_rows, SPLIT_TRAIN, dtype=object)
    split[val_indices] = SPLIT_VAL
    split[test_indices] = SPLIT_TEST

    assigned = manifest.copy()
    assigned["split"] = split
    assigned["__is_holdout"] = np.arange(n_rows, dtype=np.int64) >= holdout_start
    assigned["__valtest_random_rank"] = -1
    assigned.loc[shuffled_holdout, "__valtest_random_rank"] = np.arange(
        n_holdout, dtype=np.int64
    )

    # Sanity checks.
    split_counts = assigned["split"].value_counts()
    if split_counts.get(SPLIT_TRAIN, 0) != n_train:
        raise RuntimeError("Train split count mismatch.")
    if split_counts.get(SPLIT_VAL, 0) != n_val:
        raise RuntimeError("Validation split count mismatch.")
    if split_counts.get(SPLIT_TEST, 0) != n_test:
        raise RuntimeError("Test split count mismatch.")

    return assigned


def _build_file_split_lookup(assigned_manifest: pd.DataFrame) -> Dict[str, np.ndarray]:
    lookup: Dict[str, np.ndarray] = {}
    for source_file, file_rows in assigned_manifest.groupby("__source_file"):
        max_row_idx = int(file_rows["__row_in_file"].max())
        split_arr = np.empty(max_row_idx + 1, dtype=object)
        split_arr[file_rows["__row_in_file"].to_numpy()] = file_rows["split"].to_numpy()
        lookup[source_file] = split_arr
    return lookup


def _arrow_type_family(arrow_type: pa.DataType) -> str:
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "string"
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return "string"
    if pa.types.is_floating(arrow_type):
        return "float"
    if pa.types.is_integer(arrow_type):
        return "int"
    if pa.types.is_boolean(arrow_type):
        return "bool"
    if pa.types.is_date(arrow_type) or pa.types.is_timestamp(arrow_type):
        return "datetime"
    return "other"


def _merge_type_families(left: str, right: str) -> str:
    if left == right:
        return left
    if "string" in (left, right):
        return "string"
    if "datetime" in (left, right):
        # Mixed datetime with numerics/other is safest as string for consistency.
        return "string"
    if {"int", "float"} == {left, right}:
        return "float"
    if "other" in (left, right):
        return "string"
    # e.g. bool + int/float => numeric
    if {"bool", "int"} == {left, right}:
        return "int"
    if {"bool", "float"} == {left, right}:
        return "float"
    return "string"


def _infer_unified_type_plan(files: List[Path]) -> Dict[str, str]:
    column_family: Dict[str, str] = {}
    for src_path in files:
        schema = pq.read_schema(src_path)
        for field in schema:
            family = _arrow_type_family(field.type)
            prior = column_family.get(field.name)
            column_family[field.name] = family if prior is None else _merge_type_families(
                prior, family
            )
    return column_family


def _cast_frame_to_unified_plan(
    frame: pd.DataFrame, type_plan: Dict[str, str]
) -> pd.DataFrame:
    out = frame.copy()
    for col, family in type_plan.items():
        if col not in out.columns:
            continue
        if family == "string":
            out[col] = out[col].astype("string")
        elif family == "float":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
        elif family == "int":
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
        elif family == "bool":
            out[col] = out[col].astype("boolean")
        elif family == "datetime":
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _write_split_parquets(
    files: List[Path],
    file_split_lookup: Dict[str, np.ndarray],
    output_data_dir: Path,
    unified_type_plan: Optional[Dict[str, str]] = None,
) -> None:
    output_data_dir.mkdir(parents=True, exist_ok=True)
    writers: Dict[str, pq.ParquetWriter] = {}
    output_paths = {
        SPLIT_TRAIN: output_data_dir / "train.parquet",
        SPLIT_VAL: output_data_dir / "val.parquet",
        SPLIT_TEST: output_data_dir / "test.parquet",
    }

    try:
        for src_path in files:
            frame = pd.read_parquet(src_path)
            # Normalize dtypes to pandas nullable types to reduce schema drift
            # across files (e.g., int columns with nulls in only some files).
            frame = frame.convert_dtypes()
            if unified_type_plan:
                frame = _cast_frame_to_unified_plan(frame, unified_type_plan)
            if len(frame) == 0:
                continue

            split_arr = file_split_lookup[src_path.name]
            if len(split_arr) != len(frame):
                raise RuntimeError(
                    f"Row count mismatch for '{src_path.name}': "
                    f"manifest has {len(split_arr)} rows, parquet has {len(frame)} rows."
                )

            for split_name in SPLIT_ORDER:
                mask = split_arr == split_name
                if not np.any(mask):
                    continue
                split_frame = frame.loc[mask].reset_index(drop=True)
                table = pa.Table.from_pandas(split_frame, preserve_index=False)
                if split_name not in writers:
                    writers[split_name] = pq.ParquetWriter(
                        where=output_paths[split_name], schema=table.schema
                    )
                writers[split_name].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()


def _split_time_summary(assigned_manifest: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for split_name in SPLIT_ORDER:
        part = assigned_manifest.loc[assigned_manifest["split"] == split_name]
        times = part["__sort_time"].dropna()
        if len(times) == 0:
            out[split_name] = {"min_sort_time": None, "max_sort_time": None}
        else:
            out[split_name] = {
                "min_sort_time": float(times.min()),
                "max_sort_time": float(times.max()),
            }
    return out


def _write_outputs(
    assigned_manifest: pd.DataFrame,
    files: List[Path],
    config: SplitConfig,
) -> None:
    manifest_dir = config.output_dir / "manifest"
    data_dir = config.output_dir / "data"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    manifest_to_save = assigned_manifest[
        [
            "__source_file",
            "__row_in_file",
            config.start_time_col,
            "__sort_time",
            "__sort_time_missing",
            "split",
            "__is_holdout",
            "__valtest_random_rank",
        ]
    ]
    manifest_path = manifest_dir / "split_manifest.parquet"
    manifest_to_save.to_parquet(manifest_path, index=False)

    lookup = _build_file_split_lookup(assigned_manifest)
    unified_type_plan = _infer_unified_type_plan(files)
    _write_split_parquets(files, lookup, data_dir, unified_type_plan=unified_type_plan)

    split_counts = assigned_manifest["split"].value_counts().to_dict()
    split_counts = {key: int(split_counts.get(key, 0)) for key in SPLIT_ORDER}
    metadata = {
        "input_dir": str(config.input_dir.resolve()),
        "file_pattern": config.file_pattern,
        "start_time_col": config.start_time_col,
        "train_frac": config.train_frac,
        "seed": config.seed,
        "allow_datetime_fallback": config.allow_datetime_fallback,
        "file_count": len(files),
        "row_count": int(len(assigned_manifest)),
        "rows_with_missing_start_time": int(
            assigned_manifest["__sort_time_missing"].sum()
        ),
        "split_counts": split_counts,
        "split_time_ranges": _split_time_summary(assigned_manifest),
        "outputs": {
            "train_parquet": str((data_dir / "train.parquet").resolve()),
            "val_parquet": str((data_dir / "val.parquet").resolve()),
            "test_parquet": str((data_dir / "test.parquet").resolve()),
            "manifest_parquet": str(manifest_path.resolve()),
        },
    }
    metadata_path = config.output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def run(config: SplitConfig) -> None:
    files = _discover_files(config.input_dir, config.file_pattern)
    assigned_manifest = _assign_splits(
        _build_manifest(
            files,
            config.start_time_col,
            allow_datetime_fallback=config.allow_datetime_fallback,
        ),
        train_frac=config.train_frac,
        seed=config.seed,
    )
    _write_outputs(assigned_manifest, files, config)

    split_counts = assigned_manifest["split"].value_counts().to_dict()
    print("Split generation complete.")
    print(f"Input files: {len(files)}")
    print(f"Rows: {len(assigned_manifest)}")
    print(
        "Counts: "
        f"train={split_counts.get(SPLIT_TRAIN, 0)}, "
        f"val={split_counts.get(SPLIT_VAL, 0)}, "
        f"test={split_counts.get(SPLIT_TEST, 0)}"
    )
    print(f"Artifacts written to: {config.output_dir.resolve()}")


if __name__ == "__main__":
    run(_parse_args())
