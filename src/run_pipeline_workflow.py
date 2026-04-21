#!/usr/bin/env python3
"""Run the full HPC surrogate pipeline with checkpoint/replay support."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


STAGES = [
    "split",
    "clean",
    "features",
    "leakage",
    "eda",
    "train",
    "viz_metrics",
    "viz_stages",
]


@dataclass(frozen=True)
class StageSpec:
    name: str
    script: str
    default_args: List[str]


STAGE_SPECS: Dict[str, StageSpec] = {
    "split": StageSpec("split", "load_split_totals.py", []),
    "clean": StageSpec("clean", "clean_split_totals.py", []),
    "features": StageSpec("features", "engineer_io_features.py", []),
    "leakage": StageSpec("leakage", "apply_leakage_policy.py", []),
    "eda": StageSpec(
        "eda",
        "run_eda_totals.py",
        ["--variant-glob", "balanced", "--log-target", "--drop-invalid-target"],
    ),
    "train": StageSpec(
        "train",
        "train_xgb_optuna.py",
        ["--variants", "balanced", "--target-modes", "log1p"],
    ),
    "viz_metrics": StageSpec("viz_metrics", "visualize_sweep_metrics.py", []),
    "viz_stages": StageSpec(
        "viz_stages",
        "visualize_pipeline_stages.py",
        ["--input-root", "out"],
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pipeline stages with resumable checkpointing. "
            "Use --replay-stage to rerun a stage and all downstream stages."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("out/workflow/checkpoint.json"),
        help="Checkpoint JSON path.",
    )
    parser.add_argument(
        "--from-stage",
        choices=STAGES,
        default=STAGES[0],
        help="Start stage for this invocation.",
    )
    parser.add_argument(
        "--to-stage",
        choices=STAGES,
        default=STAGES[-1],
        help="Last stage for this invocation.",
    )
    parser.add_argument(
        "--replay-stage",
        choices=STAGES,
        default=None,
        help=(
            "Reset checkpoint status from this stage onward before running. "
            "Useful for replaying train or later stages."
        ),
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable for subprocess stage calls.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stage commands without executing.",
    )

    for stage in STAGES:
        parser.add_argument(
            f"--{stage.replace('_', '-')}-args",
            type=str,
            default="",
            help=f'Extra args passed to stage "{stage}" (quoted string).',
        )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> Dict[str, object]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created_at": _utc_now(), "stages": {}}


def _save_checkpoint(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = _utc_now()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _stage_bounds(from_stage: str, to_stage: str) -> List[str]:
    i0 = STAGES.index(from_stage)
    i1 = STAGES.index(to_stage)
    if i0 > i1:
        raise ValueError("--from-stage must come before or equal to --to-stage.")
    return STAGES[i0 : i1 + 1]


def _reset_from_stage(checkpoint: Dict[str, object], stage: str) -> None:
    start_idx = STAGES.index(stage)
    stages = checkpoint.setdefault("stages", {})
    for s in STAGES[start_idx:]:
        stages.pop(s, None)


def _split_extra_args(raw: str) -> List[str]:
    return shlex.split(raw) if raw.strip() else []


def _command_for_stage(
    repo_root: Path, python_exec: str, stage: str, extra: List[str]
) -> List[str]:
    spec = STAGE_SPECS[stage]
    script_path = repo_root / "src" / spec.script
    return [python_exec, str(script_path), *spec.default_args, *extra]


def run() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    checkpoint = _load_checkpoint(args.checkpoint)
    checkpoint.setdefault("stages", {})

    if args.replay_stage:
        _reset_from_stage(checkpoint, args.replay_stage)
        _save_checkpoint(args.checkpoint, checkpoint)

    stages_to_consider = _stage_bounds(args.from_stage, args.to_stage)
    for stage in stages_to_consider:
        stage_state = checkpoint["stages"].get(stage, {})
        if stage_state.get("status") == "completed":
            print(f"[skip] {stage}: already completed in checkpoint")
            continue

        extra_raw = getattr(args, f"{stage.replace('_', '_')}_args", "")
        # argparse stores dashes as underscores already, stage names contain underscores.
        extra = _split_extra_args(extra_raw)
        cmd = _command_for_stage(repo_root, args.python, stage, extra)
        print(f"[run] {stage}: {' '.join(shlex.quote(c) for c in cmd)}")

        if args.dry_run:
            continue

        t0 = time.time()
        subprocess.run(cmd, cwd=repo_root, check=True)
        elapsed = time.time() - t0

        checkpoint["stages"][stage] = {
            "status": "completed",
            "completed_at": _utc_now(),
            "elapsed_sec": elapsed,
            "command": cmd,
        }
        _save_checkpoint(args.checkpoint, checkpoint)
        print(f"[done] {stage} ({elapsed:.1f}s)")

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Workflow complete. Checkpoint: {args.checkpoint.resolve()}")


if __name__ == "__main__":
    run()
