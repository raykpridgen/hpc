#!/usr/bin/env python3
"""Clean split totals parquet data for downstream feature/target phases."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class CleanConfig:
    input_dir: Path
    output_dir: Path
    variance_thresholds: Tuple[float, ...]
    drop_columns: Tuple[str, ...]
    critical_time_columns: Tuple[str, ...]


def _parse_csv_list(raw: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _parse_args() -> CleanConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Apply cleaning to split totals parquet files and emit cleaned artifacts "
            "for one or more variance thresholds."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("out/splits/totals/data"),
        help="Directory containing split parquet files (train/val/test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/preprocess/totals"),
        help="Directory to write cleaned datasets and metadata.",
    )
    parser.add_argument(
        "--variance-threshold",
        action="append",
        dest="variance_thresholds",
        type=float,
        help=(
            "Variance threshold to apply (repeatable). "
            "Default runs both 0.0 and 0.01."
        ),
    )
    parser.add_argument(
        "--drop-columns",
        type=str,
        default="uid,jobid",
        help="Comma-separated columns to drop as identifiers/useless fields.",
    )
    parser.add_argument(
        "--critical-time-columns",
        type=str,
        default="start_time,end_time",
        help=(
            "Comma-separated critical time columns that are never zero-imputed and "
            "must be non-null."
        ),
    )
    args = parser.parse_args()

    thresholds = args.variance_thresholds or [0.0, 0.01]
    if any(t < 0.0 for t in thresholds):
        raise ValueError("All --variance-threshold values must be >= 0.")

    return CleanConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        variance_thresholds=tuple(thresholds),
        drop_columns=_parse_csv_list(args.drop_columns),
        critical_time_columns=_parse_csv_list(args.critical_time_columns),
    )


def _threshold_tag(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    text = text if text else "0"
    return text.replace("-", "m").replace(".", "p")


def _load_split_frames(input_dir: Path) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = input_dir / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing split parquet: {path}")
        frames[split] = pd.read_parquet(path)
    return frames


def _drop_identifier_columns(
    frames: Dict[str, pd.DataFrame], drop_columns: Iterable[str]
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    present = []
    for col in drop_columns:
        if any(col in frame.columns for frame in frames.values()):
            present.append(col)

    cleaned = {
        split: frame.drop(columns=present, errors="ignore") for split, frame in frames.items()
    }
    return cleaned, present


def _drop_missing_critical_rows(
    frames: Dict[str, pd.DataFrame], critical_columns: Tuple[str, ...]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int], List[str]]:
    present_critical = [
        col for col in critical_columns if any(col in frame.columns for frame in frames.values())
    ]
    cleaned: Dict[str, pd.DataFrame] = {}
    dropped_per_split: Dict[str, int] = {}
    for split, frame in frames.items():
        if not present_critical:
            cleaned[split] = frame.copy()
            dropped_per_split[split] = 0
            continue

        missing_cols = [col for col in present_critical if col not in frame.columns]
        if missing_cols:
            raise KeyError(
                f"Split '{split}' missing required critical columns: {missing_cols}"
            )
        before = len(frame)
        valid_mask = frame[present_critical].notna().all(axis=1)
        cleaned[split] = frame.loc[valid_mask].reset_index(drop=True)
        dropped_per_split[split] = before - len(cleaned[split])
    return cleaned, dropped_per_split, present_critical


def _impute_numeric_noncritical_zero(
    frames: Dict[str, pd.DataFrame], protected_columns: Iterable[str]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, int], List[str]]:
    protected = set(protected_columns)
    cleaned: Dict[str, pd.DataFrame] = {}
    imputed_cells: Dict[str, int] = {}
    numeric_columns_used: set[str] = set()

    for split, frame in frames.items():
        out = frame.copy()
        numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
        candidate_cols = [col for col in numeric_cols if col not in protected]
        numeric_columns_used.update(candidate_cols)
        if candidate_cols:
            before_na = int(out[candidate_cols].isna().sum().sum())
            out[candidate_cols] = out[candidate_cols].fillna(0)
            imputed_cells[split] = before_na
        else:
            imputed_cells[split] = 0
        cleaned[split] = out

    return cleaned, imputed_cells, sorted(numeric_columns_used)


def _fit_low_variance_drop_list(
    train_frame: pd.DataFrame, candidate_columns: List[str], threshold: float
) -> List[str]:
    if not candidate_columns:
        return []
    variances = train_frame[candidate_columns].var(axis=0, ddof=0)
    to_drop = variances[variances <= threshold].index.tolist()
    return sorted(to_drop)


def _apply_column_drop(
    frames: Dict[str, pd.DataFrame], columns: List[str]
) -> Dict[str, pd.DataFrame]:
    return {
        split: frame.drop(columns=columns, errors="ignore").copy()
        for split, frame in frames.items()
    }


def _write_clean_outputs(
    frames: Dict[str, pd.DataFrame], output_data_dir: Path
) -> Dict[str, str]:
    output_data_dir.mkdir(parents=True, exist_ok=True)
    out_paths: Dict[str, str] = {}
    for split in SPLITS:
        path = output_data_dir / f"{split}.parquet"
        frames[split].to_parquet(path, index=False)
        out_paths[split] = str(path.resolve())
    return out_paths


def _run_threshold(
    frames_base: Dict[str, pd.DataFrame],
    threshold: float,
    output_dir: Path,
    config: CleanConfig,
    dropped_identifier_columns: List[str],
    dropped_critical_rows: Dict[str, int],
    present_critical_columns: List[str],
    imputed_cells: Dict[str, int],
    numeric_candidate_columns: List[str],
) -> Dict[str, object]:
    low_variance_drop_cols = _fit_low_variance_drop_list(
        frames_base["train"], numeric_candidate_columns, threshold
    )
    final_frames = _apply_column_drop(frames_base, low_variance_drop_cols)

    threshold_dir = output_dir / f"variance_{_threshold_tag(threshold)}"
    data_paths = _write_clean_outputs(final_frames, threshold_dir / "data")

    metadata = {
        "variance_threshold": threshold,
        "input_dir": str(config.input_dir.resolve()),
        "drop_columns_requested": list(config.drop_columns),
        "drop_columns_applied": dropped_identifier_columns,
        "critical_time_columns_requested": list(config.critical_time_columns),
        "critical_time_columns_applied": present_critical_columns,
        "rows_dropped_missing_critical": dropped_critical_rows,
        "numeric_cells_zero_imputed": imputed_cells,
        "low_variance_drop_count": len(low_variance_drop_cols),
        "low_variance_drop_columns": low_variance_drop_cols,
        "row_counts": {split: int(len(final_frames[split])) for split in SPLITS},
        "column_counts": {split: int(final_frames[split].shape[1]) for split in SPLITS},
        "outputs": data_paths,
    }
    metadata_path = threshold_dir / "metadata.json"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "threshold": threshold,
        "dir": str(threshold_dir.resolve()),
        "rows": metadata["row_counts"],
        "low_variance_drop_count": len(low_variance_drop_cols),
    }


def run(config: CleanConfig) -> None:
    raw_frames = _load_split_frames(config.input_dir)
    no_ids_frames, dropped_identifier_columns = _drop_identifier_columns(
        raw_frames, config.drop_columns
    )
    no_missing_critical_frames, dropped_critical_rows, present_critical = (
        _drop_missing_critical_rows(no_ids_frames, config.critical_time_columns)
    )
    imputed_frames, imputed_cells, numeric_candidate_columns = (
        _impute_numeric_noncritical_zero(
            no_missing_critical_frames, protected_columns=present_critical
        )
    )

    run_summaries: List[Dict[str, object]] = []
    for threshold in config.variance_thresholds:
        summary = _run_threshold(
            frames_base=imputed_frames,
            threshold=threshold,
            output_dir=config.output_dir,
            config=config,
            dropped_identifier_columns=dropped_identifier_columns,
            dropped_critical_rows=dropped_critical_rows,
            present_critical_columns=present_critical,
            imputed_cells=imputed_cells,
            numeric_candidate_columns=numeric_candidate_columns,
        )
        run_summaries.append(summary)

    print("Cleaning complete.")
    print(f"Input split dir: {config.input_dir.resolve()}")
    for summary in run_summaries:
        rows = summary["rows"]
        print(
            f"threshold={summary['threshold']} "
            f"rows(train/val/test)={rows['train']}/{rows['val']}/{rows['test']} "
            f"low_variance_dropped={summary['low_variance_drop_count']} "
            f"out={summary['dir']}"
        )


if __name__ == "__main__":
    run(_parse_args())
