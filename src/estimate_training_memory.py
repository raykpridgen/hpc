#!/usr/bin/env python3
"""Estimate memory budget for parquet-surrogate training runs."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _human_gb(n_bytes: float) -> float:
    return float(n_bytes) / (1024.0**3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-glob", default="data/darshan_total/App*.parquet")
    p.add_argument("--sample-files", type=int, default=4)
    p.add_argument("--with-detail", action="store_true")
    args = p.parse_args()

    files = sorted(glob.glob(args.total_glob))
    if not files:
        raise SystemExit(f"No files matched {args.total_glob!r}")

    total_disk = sum(Path(f).stat().st_size for f in files)
    sample = files[: max(1, min(args.sample_files, len(files)))]

    ratios: list[float] = []
    n_rows = 0
    n_cols = 0
    for f in sample:
        df = pd.read_parquet(f)
        n_rows += int(len(df))
        n_cols = max(n_cols, int(df.shape[1]))
        mem = float(df.memory_usage(index=True, deep=True).sum())
        sz = float(Path(f).stat().st_size)
        if sz > 0:
            ratios.append(mem / sz)

    # Conservative ratio from samples; fallback if unavailable.
    ratio = float(np.median(ratios)) if ratios else 3.5
    ratio = max(ratio, 2.0)

    est_totals_df_mem = total_disk * ratio
    # During training, multiple materialized copies exist (calendar-expanded DF, X matrix, slices).
    pipeline_multiplier = 2.4
    est_peak = est_totals_df_mem * pipeline_multiplier
    if args.with_detail:
        # Detail is streamed in batches; add bounded headroom for batch buffers.
        est_peak += 1.5 * (1024.0**3)

    recommended = est_peak * 1.3  # 30% operational headroom

    print("[memory-estimate]")
    print(f"  total_files={len(files)}")
    print(f"  total_disk_gb={_human_gb(total_disk):.2f}")
    print(f"  inferred_in_memory_ratio={ratio:.2f}x")
    print(f"  estimated_totals_df_gb={_human_gb(est_totals_df_mem):.2f}")
    print(f"  estimated_peak_gb={_human_gb(est_peak):.2f}")
    print(f"  recommended_allocation_gb={_human_gb(recommended):.2f}")
    if n_rows and n_cols:
        print(f"  sample_rows={n_rows}, sample_max_cols={n_cols}")


if __name__ == "__main__":
    main()
