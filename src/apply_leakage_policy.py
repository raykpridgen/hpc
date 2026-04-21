#!/usr/bin/env python3
"""Apply leakage policies and emit leakage-safe split datasets."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd


SPLITS = ("train", "val", "test")
VARIANTS = ("balanced", "ablation_no_bytes", "ablation_no_time")


@dataclass(frozen=True)
class LeakageConfig:
    input_root: Path
    output_root: Path
    dataset_glob: str
    target_col: str
    metadata_name: str


def _parse_args() -> LeakageConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Apply leakage policy to engineered totals datasets and write "
            "policy-specific train/val/test artifacts."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("out/features/totals"),
        help="Root containing engineered datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("out/leakage/totals"),
        help="Root to write leakage-policy outputs.",
    )
    parser.add_argument(
        "--dataset-glob",
        type=str,
        default="*",
        help="Dataset glob under input-root (e.g., variance_*).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="io_intensity",
        help="Target column to preserve in all variants.",
    )
    parser.add_argument(
        "--metadata-name",
        type=str,
        default="metadata.json",
        help="Metadata filename emitted by feature-engineering module.",
    )
    args = parser.parse_args()
    return LeakageConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        dataset_glob=args.dataset_glob,
        target_col=args.target_col,
        metadata_name=args.metadata_name,
    )


def _discover_datasets(root: Path, dataset_glob: str) -> List[Tuple[str, Path, Path]]:
    out: List[Tuple[str, Path, Path]] = []
    for dataset_dir in sorted(p for p in root.glob(dataset_glob) if p.is_dir()):
        data_dir = dataset_dir / "data"
        meta_path = dataset_dir / "metadata.json"
        if all((data_dir / f"{split}.parquet").exists() for split in SPLITS):
            out.append((dataset_dir.name, data_dir, meta_path))
    if not out:
        raise FileNotFoundError(
            f"No engineered datasets found under '{root}' for glob '{dataset_glob}'."
        )
    return out


def _load_splits(data_dir: Path) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        p = data_dir / f"{split}.parquet"
        frames[split] = pd.read_parquet(p)
    return frames


def _load_feature_metadata(meta_path: Path) -> Dict[str, object]:
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _time_like_columns(columns: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for col in columns:
        if re.search(r"(^|_)TIME($|_)", col):
            out.add(col)
        if col.endswith("_TIMESTAMP") or "_TIMESTAMP_" in col:
            out.add(col)
    for fixed in ("start_time", "end_time", "runtime", "runtime_scaled"):
        if fixed in columns:
            out.add(fixed)
    return out


def _byte_amount_columns(columns: Sequence[str]) -> Set[str]:
    """Columns that directly encode byte/size magnitudes.

    Keep operation-count style fields (READS/WRITES/etc) available.
    """
    out: Set[str] = set()
    for col in columns:
        if "_BYTES_" in col or col.endswith("_BYTES"):
            out.add(col)
            continue
        if "_SIZE_" in col:
            out.add(col)
            continue
        if "_MAX_BYTE_" in col:
            out.add(col)
            continue
    return out


def _baseline_forbidden(columns: Sequence[str], target_col: str) -> Set[str]:
    forbidden = set()
    for col in ("total_io", "runtime", "runtime_scaled"):
        if col in columns:
            forbidden.add(col)
    # start/end timestamps trivially reconstruct runtime.
    for col in ("start_time", "end_time"):
        if col in columns:
            forbidden.add(col)
    # Protect against accidental target leakage columns.
    for col in columns:
        if col.lower() in {"target", "label"}:
            forbidden.add(col)
    if target_col in forbidden:
        forbidden.remove(target_col)
    return forbidden


def _build_drop_sets(
    columns: Sequence[str], target_col: str, feature_meta: Dict[str, object]
) -> Dict[str, Dict[str, List[str]]]:
    base_forbidden = _baseline_forbidden(columns, target_col)
    time_cols = _time_like_columns(columns)
    byte_amount_cols = _byte_amount_columns(columns)
    read_cols_used = set(feature_meta.get("read_columns_used", []) or [])
    write_cols_used = set(feature_meta.get("write_columns_used", []) or [])
    direct_io_components = {
        c for c in (read_cols_used | write_cols_used) if c in set(columns)
    }

    # Balanced: drop direct components, explicit runtime/target builders,
    # and byte/size magnitude columns (but keep op-count fields like READS/WRITES).
    balanced_drop = set(base_forbidden) | direct_io_components | byte_amount_cols

    # Ablation variants extend balanced policy.
    no_bytes_drop = set(balanced_drop) | byte_amount_cols
    no_time_drop = set(balanced_drop) | time_cols

    def sorted_reasons(drop_set: Set[str]) -> Dict[str, List[str]]:
        reasons: Dict[str, List[str]] = {}
        for col in sorted(drop_set):
            rs: List[str] = []
            if col in base_forbidden:
                rs.append("direct_target_component")
            if col in direct_io_components:
                rs.append("direct_io_constructor")
            if col in byte_amount_cols:
                rs.append("byte_or_size_family")
            if col in time_cols:
                rs.append("time_family")
            reasons[col] = rs or ["policy_drop"]
        return reasons

    return {
        "balanced": sorted_reasons(balanced_drop),
        "ablation_no_bytes": sorted_reasons(no_bytes_drop),
        "ablation_no_time": sorted_reasons(no_time_drop),
    }


def _drop_columns(
    frames: Dict[str, pd.DataFrame], columns_to_drop: Iterable[str]
) -> Dict[str, pd.DataFrame]:
    cols = list(columns_to_drop)
    return {split: df.drop(columns=cols, errors="ignore") for split, df in frames.items()}


def _validate_target(frames: Dict[str, pd.DataFrame], target_col: str) -> None:
    for split, df in frames.items():
        if target_col not in df.columns:
            raise KeyError(f"Split '{split}' missing target column '{target_col}'.")


def _write_variant_outputs(
    frames: Dict[str, pd.DataFrame], out_dir: Path
) -> Dict[str, str]:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    for split in SPLITS:
        path = data_dir / f"{split}.parquet"
        frames[split].to_parquet(path, index=False)
        outputs[split] = str(path.resolve())
    return outputs


def _candidate_feature_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    return [c for c in df.columns if c != target_col]


def _apply_dataset(config: LeakageConfig, dataset: Tuple[str, Path, Path]) -> Dict[str, object]:
    name, data_dir, meta_path = dataset
    frames = _load_splits(data_dir)
    _validate_target(frames, config.target_col)

    feature_meta = _load_feature_metadata(meta_path)
    train_cols = list(frames["train"].columns)
    drop_sets = _build_drop_sets(
        columns=train_cols,
        target_col=config.target_col,
        feature_meta=feature_meta,
    )

    dataset_summary: Dict[str, object] = {"dataset": name, "variants": {}}
    for variant in VARIANTS:
        reason_map = drop_sets[variant]
        dropped_cols = list(reason_map.keys())
        variant_frames = _drop_columns(frames, dropped_cols)
        _validate_target(variant_frames, config.target_col)

        out_dir = config.output_root / name / variant
        outputs = _write_variant_outputs(variant_frames, out_dir)

        train_features = _candidate_feature_columns(variant_frames["train"], config.target_col)
        metadata = {
            "dataset": name,
            "variant": variant,
            "input_data_dir": str(data_dir.resolve()),
            "target_col": config.target_col,
            "drop_count": len(dropped_cols),
            "drop_columns": reason_map,
            "feature_columns_count": len(train_features),
            "feature_columns": train_features,
            "rows": {split: int(len(variant_frames[split])) for split in SPLITS},
            "outputs": outputs,
        }
        with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

        dataset_summary["variants"][variant] = {
            "drop_count": len(dropped_cols),
            "feature_columns_count": len(train_features),
            "out_dir": str(out_dir.resolve()),
        }
    return dataset_summary


def run(config: LeakageConfig) -> None:
    datasets = _discover_datasets(config.input_root, config.dataset_glob)
    summaries = [_apply_dataset(config, ds) for ds in datasets]
    print("Leakage policy application complete.")
    for s in summaries:
        name = s["dataset"]
        parts = []
        for variant in VARIANTS:
            v = s["variants"][variant]
            parts.append(
                f"{variant}:drop={v['drop_count']},features={v['feature_columns_count']}"
            )
        print(f"dataset={name} | " + " | ".join(parts))


if __name__ == "__main__":
    run(_parse_args())
