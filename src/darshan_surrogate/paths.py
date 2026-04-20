"""Default output layout under `out/` (design.md §11.1)."""

from pathlib import Path


def project_root() -> Path:
    """Repo root (parent of `src/`)."""
    return Path(__file__).resolve().parents[2]


def out_dirs(repo_root: Path | None = None, out_name: str = "out") -> dict[str, Path]:
    base = (repo_root or project_root()) / out_name
    return {
        "root": base,
        "eda": base / "eda",
        "train": base / "train",
        "models": base / "models",
    }


def ensure_out_dirs(repo_root: Path | None = None, out_name: str = "out") -> dict[str, Path]:
    d = out_dirs(repo_root, out_name)
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d
