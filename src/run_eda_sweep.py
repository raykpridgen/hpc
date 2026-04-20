#!/usr/bin/env python3
"""
Run a targeted EDA sweep over feature-group and leakage settings, then summarize.

The script executes a fixed matrix of runs (not brute-force combinatorics) and
automatically triggers post-sweep summarization.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone


def _run_cmd(cmd: list[str]) -> int:
    print("[run]", " ".join(cmd))
    p = subprocess.run(cmd, check=False)
    return int(p.returncode)


def _project_root() -> Path:
    """Project root containing `src/` (e.g. repo/hpc)."""
    return Path(__file__).resolve().parents[1]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_path(out_root: Path) -> Path:
    return out_root / "checkpoint.json"


def _load_checkpoint(out_root: Path) -> dict:
    p = _checkpoint_path(out_root)
    if not p.exists():
        return {
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "runs": {},
            "summary": {},
        }
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("checkpoint root must be object")
        data.setdefault("runs", {})
        data.setdefault("summary", {})
        data["updated_at"] = _utc_now()
        return data
    except Exception:
        # If corrupted, keep old copy and start fresh.
        bak = p.with_suffix(".json.bak")
        p.replace(bak)
        return {
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "runs": {},
            "summary": {"warning": f"checkpoint was corrupted; backup at {bak.name}"},
        }


def _save_checkpoint(out_root: Path, ckpt: dict) -> None:
    p = _checkpoint_path(out_root)
    ckpt["updated_at"] = _utc_now()
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)
    tmp.replace(p)


def _resolve_python_interpreter(python_arg: str | None) -> str:
    """
    Resolve Python executable for child runs.

    Default to current interpreter for consistency with active environment.
    """
    if python_arg and python_arg.strip():
        exe = python_arg.strip()
        if not Path(exe).is_absolute() and "/" in exe:
            exe = str((Path.cwd() / exe).resolve())
        elif "/" not in exe:
            # Let PATH resolution handle simple names like "python3".
            pass
    else:
        exe = sys.executable
    return exe


def _assert_python_version(py_exe: str, min_major: int = 3, min_minor: int = 10) -> None:
    cmd = [
        py_exe,
        "-c",
        (
            "import sys; "
            "print(f'{sys.version_info.major}.{sys.version_info.minor}'); "
            "print(sys.executable)"
        ),
    ]
    out = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(
            f"Unable to run interpreter {py_exe!r}. "
            f"stderr: {out.stderr.strip() or '<empty>'}"
        )
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    ver = lines[0] if lines else "0.0"
    try:
        major, minor = (int(x) for x in ver.split(".")[:2])
    except Exception:
        raise SystemExit(f"Could not parse Python version from {py_exe!r}: {ver!r}")

    if (major, minor) < (min_major, min_minor):
        resolved = lines[1] if len(lines) > 1 else py_exe
        raise SystemExit(
            "Python interpreter is too old for this project.\n"
            f"  requested: {py_exe}\n"
            f"  resolved:  {resolved}\n"
            f"  version:   {major}.{minor}\n"
            f"  required:  >= {min_major}.{min_minor}\n"
            "Tip: activate your project venv and omit --python, or pass a newer interpreter."
        )


def _expand_total_glob(pattern: str) -> list[str]:
    p = Path(pattern)
    candidates: list[str] = []
    if p.is_absolute():
        candidates = [str(p)]
    else:
        proj = _project_root()
        # Support both:
        # - running from repo/hpc with data under ../darshan_share
        # - running from repo root with local relative paths
        candidates = [
            str((Path.cwd() / p).resolve()),
            str((Path.cwd().parent / p).resolve()),
            str((proj / p).resolve()),
            str((proj.parent / p).resolve()),
        ]
    seen: set[str] = set()
    matches: list[str] = []
    for pat in candidates:
        for m in sorted(glob.glob(pat)):
            if m not in seen:
                seen.add(m)
                matches.append(m)
    return matches


def _resolve_detail_root(path_str: str) -> Path | None:
    p = Path(path_str)
    cands: list[Path]
    if p.is_absolute():
        cands = [p]
    else:
        proj = _project_root()
        cands = [
            (Path.cwd() / p).resolve(),
            (Path.cwd().parent / p).resolve(),
            (proj / p).resolve(),
            (proj.parent / p).resolve(),
        ]
    for c in cands:
        if c.exists() and c.is_dir():
            return c
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--python",
        default="",
        help=(
            "Python executable for child scripts. Default: current interpreter "
            "(sys.executable)."
        ),
    )
    p.add_argument("--total-glob", default="../darshan_share/darshan_total/App*.parquet")
    p.add_argument("--detail-root", default="../darshan_share/darshan_detail")
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
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint.json under out-root (default: on).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and rerun all matrix entries.",
    )
    p.add_argument(
        "--rerun-failed",
        action="store_true",
        help="When resuming, rerun failed entries (default skips only successful ones).",
    )
    args = p.parse_args()

    py = _resolve_python_interpreter(args.python)
    _assert_python_version(py, 3, 10)
    proj = _project_root()

    total_matches = _expand_total_glob(args.total_glob)
    if not total_matches:
        raise SystemExit(
            f"No files matched --total-glob {args.total_glob!r} "
            f"(searched from cwd={Path.cwd()} and project_root={proj})"
        )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    use_resume = args.resume and not args.no_resume
    ckpt = _load_checkpoint(out_root) if use_resume else {
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "runs": {},
        "summary": {},
    }
    if use_resume:
        _save_checkpoint(out_root, ckpt)

    base = [py, str(proj / "src" / "train_parquet_surrogate.py"), "--total-glob", *total_matches]
    with_detail = args.with_detail or (args.profile == "detail_interactions")
    if with_detail:
        droot = _resolve_detail_root(args.detail_root)
        if droot is None:
            raise SystemExit(
                f"--detail-root does not exist: {args.detail_root!r} "
                f"(searched from cwd/project roots)"
            )
        base += ["--detail-root", str(droot)]
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
        prior = ckpt.get("runs", {}).get(run_id, {})
        if use_resume and prior.get("status") == "ok":
            print(f"[skip] {run_id}: already completed in checkpoint")
            continue
        if use_resume and (not args.rerun_failed) and prior.get("status") == "failed":
            print(f"[skip] {run_id}: previous failure (use --rerun-failed to retry)")
            failures.append(run_id)
            continue

        run_out = str(out_root / run_id)
        cmd = base + ["--out", run_out] + extra
        ckpt["runs"][run_id] = {
            "status": "running",
            "started_at": _utc_now(),
            "cmd": cmd,
            "run_out": run_out,
        }
        _save_checkpoint(out_root, ckpt)
        rc = _run_cmd(cmd)
        if rc != 0:
            failures.append(run_id)
            ckpt["runs"][run_id] = {
                "status": "failed",
                "finished_at": _utc_now(),
                "return_code": rc,
                "cmd": cmd,
                "run_out": run_out,
            }
            _save_checkpoint(out_root, ckpt)
        else:
            ckpt["runs"][run_id] = {
                "status": "ok",
                "finished_at": _utc_now(),
                "return_code": rc,
                "cmd": cmd,
                "run_out": run_out,
            }
            _save_checkpoint(out_root, ckpt)

    # Always run summarizer so partial sweeps still produce a report.
    sum_cmd = [
        py,
        str(proj / "src" / "summarize_eda_sweep.py"),
        "--sweep-root",
        str(out_root),
        "--baseline-run",
        baseline_run,
    ]
    sum_rc = _run_cmd(sum_cmd)
    ckpt["summary"]["summarize_rc"] = sum_rc
    _save_checkpoint(out_root, ckpt)

    # Memory estimate helper after sweep.
    est_cmd = [
        py,
        str(proj / "src" / "estimate_training_memory.py"),
        "--total-glob",
        args.total_glob,
    ]
    if with_detail:
        est_cmd += ["--with-detail"]
    est_rc = _run_cmd(est_cmd)
    ckpt["summary"]["estimate_rc"] = est_rc
    _save_checkpoint(out_root, ckpt)

    if failures:
        print("[warn] failed runs:", ", ".join(failures))
        sys.exit(1)
    print("[ok] sweep complete")


if __name__ == "__main__":
    main()
