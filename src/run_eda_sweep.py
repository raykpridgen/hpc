#!/usr/bin/env python3
"""
Run a targeted EDA sweep over feature-group and leakage settings, then summarize.

The script executes a fixed matrix of runs (not brute-force combinatorics) and
automatically triggers post-sweep summarization.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run_cmd(cmd: list[str]) -> int:
    print("[run]", " ".join(cmd))
    p = subprocess.run(cmd, check=False)
    return int(p.returncode)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--python", default=str(Path(".venv/bin/python")))
    p.add_argument("--total-glob", default="data/darshan_total/App*.parquet")
    p.add_argument("--detail-root", default="data/2021")
    p.add_argument("--with-detail", action="store_true")
    p.add_argument(
        "--profile",
        choices=["totals", "detail_interactions"],
        default="totals",
        help="Sweep matrix profile.",
    )
    p.add_argument("--out-root", default="out/sweeps")
    p.add_argument("--optuna-trials", type=int, default=0)
    p.add_argument("--max-runs", type=int, default=0, help="Optional cap on number of runs.")
    args = p.parse_args()

    py = args.python
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    base = [py, "src/train_parquet_surrogate.py", "--total-glob", args.total_glob]
    with_detail = args.with_detail or (args.profile == "detail_interactions")
    if with_detail:
        base += ["--detail-root", args.detail_root]
    else:
        base += ["--no-detail"]
    if args.optuna_trials > 0:
        base += ["--optuna-trials", str(args.optuna_trials)]

    # Targeted matrix: keep total runs bounded and interpretable.
    if args.profile == "totals":
        matrix: list[tuple[str, list[str]]] = [
            ("baseline_totals", []),
            ("corr_090", ["--max-abs-corr-with-y", "0.90"]),
            ("corr_080", ["--max-abs-corr-with-y", "0.80"]),
            ("drop_posix", ["--exclude-groups", "posix"]),
            ("drop_mpiio", ["--exclude-groups", "mpiio"]),
            ("drop_h5", ["--exclude-groups", "h5"]),
            ("drop_stdio", ["--exclude-groups", "stdio"]),
            ("drop_pnetcdf", ["--exclude-groups", "pnetcdf"]),
            ("core_posix_app_cal", ["--include-groups", "posix", "calendar", "app"]),
            (
                "core_posix_mpi_h5",
                ["--include-groups", "posix", "mpiio", "h5", "calendar", "app"],
            ),
        ]
        baseline_run = "baseline_totals"
    else:
        matrix = [
            ("baseline_detail", []),
            ("drop_detail", ["--exclude-groups", "detail"]),
            ("detail_core", ["--include-groups", "detail", "posix", "calendar", "app"]),
            ("detail_plus_io", ["--include-groups", "detail", "posix", "mpiio", "h5", "calendar", "app"]),
            ("detail_corr_090", ["--max-abs-corr-with-y", "0.90"]),
            ("detail_drop_pnetcdf", ["--exclude-groups", "pnetcdf"]),
        ]
        baseline_run = "baseline_detail"

    if args.max_runs and args.max_runs > 0:
        matrix = matrix[: args.max_runs]

    failures: list[str] = []
    for run_id, extra in matrix:
        run_out = str(out_root / run_id)
        cmd = base + ["--out", run_out] + extra
        rc = _run_cmd(cmd)
        if rc != 0:
            failures.append(run_id)

    # Always run summarizer so partial sweeps still produce a report.
    sum_cmd = [
        py,
        "src/summarize_eda_sweep.py",
        "--sweep-root",
        str(out_root),
        "--baseline-run",
        baseline_run,
    ]
    _run_cmd(sum_cmd)

    # Memory estimate helper after sweep.
    est_cmd = [
        py,
        "src/estimate_training_memory.py",
        "--total-glob",
        args.total_glob,
    ]
    if with_detail:
        est_cmd += ["--with-detail"]
    _run_cmd(est_cmd)

    if failures:
        print("[warn] failed runs:", ", ".join(failures))
        sys.exit(1)
    print("[ok] sweep complete")


if __name__ == "__main__":
    main()
