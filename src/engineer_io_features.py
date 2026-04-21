#!/usr/bin/env python3
"""Append runtime/IO target features to cleaned split parquet datasets."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class FeatureConfig:
    input_root: Path
    output_root: Path
    dataset_glob: str
    start_time_col: str
    end_time_col: str
    read_cols_override: Tuple[str, ...]
    write_cols_override: Tuple[str, ...]
    min_runtime: float


def _parse_csv(raw: Optional[str]) -> Tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_args() -> FeatureConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Append runtime, runtime_scaled, total_io, and io_intensity features "
            "to train/val/test parquet splits."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("out/preprocess/totals"),
        help=(
            "Input root. Can be a directory containing train/val/test parquet files, "
            "or a root with subdirectories like variance_*/data."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("out/features/totals"),
        help="Output root for parquet datasets with appended features.",
    )
    parser.add_argument(
        "--dataset-glob",
        type=str,
        default="variance_*",
        help="Glob used to discover dataset directories under input root.",
    )
    parser.add_argument(
        "--start-time-col",
        type=str,
        default="start_time",
        help="Start-time column name.",
    )
    parser.add_argument(
        "--end-time-col",
        type=str,
        default="end_time",
        help="End-time column name.",
    )
    parser.add_argument(
        "--read-cols",
        type=str,
        default="",
        help=(
            "Optional comma-separated override of read-volume columns used in total_io."
        ),
    )
    parser.add_argument(
        "--write-cols",
        type=str,
        default="",
        help=(
            "Optional comma-separated override of write-volume columns used in total_io."
        ),
    )
    parser.add_argument(
        "--min-runtime",
        type=float,
        default=1e-12,
        help="Minimum positive runtime used when determining valid io_intensity rows.",
    )
    args = parser.parse_args()
    if args.min_runtime <= 0:
        raise ValueError("--min-runtime must be > 0.")

    return FeatureConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        dataset_glob=args.dataset_glob,
        start_time_col=args.start_time_col,
        end_time_col=args.end_time_col,
        read_cols_override=_parse_csv(args.read_cols),
        write_cols_override=_parse_csv(args.write_cols),
        min_runtime=args.min_runtime,
    )


def _load_split_frames(data_dir: Path) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        split_path = data_dir / f"{split}.parquet"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split parquet: {split_path}")
        frames[split] = pd.read_parquet(split_path)
    return frames


def _discover_dataset_dirs(root: Path, dataset_glob: str) -> List[Tuple[str, Path]]:
    # Single-dataset mode: input root directly contains train/val/test parquet.
    if all((root / f"{split}.parquet").exists() for split in SPLITS):
        return [("dataset", root)]

    dataset_dirs: List[Tuple[str, Path]] = []
    for dataset_dir in sorted(p for p in root.glob(dataset_glob) if p.is_dir()):
        data_dir = dataset_dir / "data"
        if all((data_dir / f"{split}.parquet").exists() for split in SPLITS):
            dataset_dirs.append((dataset_dir.name, data_dir))
    if not dataset_dirs:
        raise FileNotFoundError(
            f"No datasets found under '{root}' using dataset glob '{dataset_glob}'."
        )
    return dataset_dirs


def _infer_io_columns(frame: pd.DataFrame) -> Tuple[List[str], List[str], str]:
    cols = list(frame.columns)

    # Preferred: byte-volume columns.
    read_bytes = sorted(c for c in cols if re.search(r"(^|_)BYTES_READ$", c))
    write_bytes = sorted(c for c in cols if re.search(r"(^|_)BYTES_WRITTEN$", c))
    if read_bytes and write_bytes:
        return read_bytes, write_bytes, "bytes"

    # Fallback: operation counters.
    read_counts = sorted(c for c in cols if re.search(r"(^|_)READS$", c))
    write_counts = sorted(c for c in cols if re.search(r"(^|_)WRITES$", c))
    if read_counts and write_counts:
        return read_counts, write_counts, "counts"

    raise ValueError(
        "Could not infer IO read/write columns. Provide explicit --read-cols/--write-cols."
    )


def _validate_columns_exist(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"Missing {label} columns: {missing}")


def _compute_runtime(frame: pd.DataFrame, start_col: str, end_col: str) -> pd.Series:
    start = pd.to_numeric(frame[start_col], errors="coerce")
    end = pd.to_numeric(frame[end_col], errors="coerce")
    return end - start


def _fit_runtime_scale_factor(
    train_frame: pd.DataFrame, total_io_col: str, runtime_col: str, min_runtime: float
) -> float:
    train_valid = train_frame.loc[
        train_frame[runtime_col] > min_runtime, [total_io_col, runtime_col]
    ].copy()
    if train_valid.empty:
        return 1.0
    runtime_median = float(train_valid[runtime_col].median())
    total_io_median = float(train_valid[total_io_col].median())
    if runtime_median <= 0 or not np.isfinite(runtime_median):
        return 1.0
    if not np.isfinite(total_io_median):
        return 1.0
    return total_io_median / runtime_median


def _append_features(
    frame: pd.DataFrame,
    read_cols: Sequence[str],
    write_cols: Sequence[str],
    start_col: str,
    end_col: str,
    runtime_scale_factor: float,
    min_runtime: float,
) -> pd.DataFrame:
    out = frame.copy()

    read_values = out[list(read_cols)].apply(pd.to_numeric, errors="coerce").fillna(0)
    write_values = out[list(write_cols)].apply(pd.to_numeric, errors="coerce").fillna(0)
    out["total_io"] = read_values.sum(axis=1) + write_values.sum(axis=1)
    out["runtime"] = _compute_runtime(out, start_col=start_col, end_col=end_col)
    out["runtime_scaled"] = out["runtime"] * runtime_scale_factor

    valid_runtime = out["runtime"] > min_runtime
    out["io_intensity"] = np.where(valid_runtime, out["total_io"] / out["runtime"], np.nan)
    return out


def _write_split_outputs(
    frames: Dict[str, pd.DataFrame], output_data_dir: Path
) -> Dict[str, str]:
    output_data_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    for split in SPLITS:
        out_path = output_data_dir / f"{split}.parquet"
        frames[split].to_parquet(out_path, index=False)
        outputs[split] = str(out_path.resolve())
    return outputs


def _process_dataset(dataset_name: str, data_dir: Path, config: FeatureConfig) -> Dict[str, object]:
    split_frames = _load_split_frames(data_dir)
    train_frame = split_frames["train"]

    _validate_columns_exist(train_frame, [config.start_time_col], "start/end time")
    _validate_columns_exist(train_frame, [config.end_time_col], "start/end time")

    if config.read_cols_override and config.write_cols_override:
        read_cols = list(config.read_cols_override)
        write_cols = list(config.write_cols_override)
        io_column_mode = "override"
    else:
        read_cols, write_cols, io_column_mode = _infer_io_columns(train_frame)

    for split in SPLITS:
        _validate_columns_exist(split_frames[split], read_cols, "read")
        _validate_columns_exist(split_frames[split], write_cols, "write")
        _validate_columns_exist(
            split_frames[split], [config.start_time_col, config.end_time_col], "time"
        )

    train_aug_pre = _append_features(
        split_frames["train"],
        read_cols=read_cols,
        write_cols=write_cols,
        start_col=config.start_time_col,
        end_col=config.end_time_col,
        runtime_scale_factor=1.0,
        min_runtime=config.min_runtime,
    )
    runtime_scale_factor = _fit_runtime_scale_factor(
        train_aug_pre,
        total_io_col="total_io",
        runtime_col="runtime",
        min_runtime=config.min_runtime,
    )

    split_augmented: Dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        split_augmented[split] = _append_features(
            split_frames[split],
            read_cols=read_cols,
            write_cols=write_cols,
            start_col=config.start_time_col,
            end_col=config.end_time_col,
            runtime_scale_factor=runtime_scale_factor,
            min_runtime=config.min_runtime,
        )

    out_dir = config.output_root / dataset_name
    outputs = _write_split_outputs(split_augmented, out_dir / "data")

    metadata = {
        "dataset_name": dataset_name,
        "input_data_dir": str(data_dir.resolve()),
        "io_column_mode": io_column_mode,
        "read_columns_used": read_cols,
        "write_columns_used": write_cols,
        "start_time_col": config.start_time_col,
        "end_time_col": config.end_time_col,
        "min_runtime": config.min_runtime,
        "runtime_scale_factor": runtime_scale_factor,
        "rows": {split: int(len(split_augmented[split])) for split in SPLITS},
        "invalid_runtime_rows": {
            split: int((split_augmented[split]["runtime"] <= config.min_runtime).sum())
            for split in SPLITS
        },
        "invalid_io_intensity_rows": {
            split: int(split_augmented[split]["io_intensity"].isna().sum()) for split in SPLITS
        },
        "outputs": outputs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def run(config: FeatureConfig) -> None:
    datasets = _discover_dataset_dirs(config.input_root, config.dataset_glob)
    summaries: List[Dict[str, object]] = []
    for dataset_name, data_dir in datasets:
        summaries.append(_process_dataset(dataset_name, data_dir, config))

    print("Feature engineering complete.")
    for summary in summaries:
        rows = summary["rows"]
        print(
            f"dataset={summary['dataset_name']} "
            f"rows(train/val/test)={rows['train']}/{rows['val']}/{rows['test']} "
            f"io_mode={summary['io_column_mode']} "
            f"runtime_scale_factor={summary['runtime_scale_factor']:.6g} "
            f"out={config.output_root / summary['dataset_name'] / 'data'}"
        )


if __name__ == "__main__":
    run(_parse_args())
