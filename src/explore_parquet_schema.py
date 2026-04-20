"""
Print column inventory for darshan_total (App*.parquet) and darshan_detail parquets.

Usage:
  conda activate ml_base
  python src/explore_parquet_schema.py
  python src/explore_parquet_schema.py --total data/App10.parquet --detail data/2021/1/5/126712-0.parquet
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def summarize_total(path: Path) -> None:
    df = pd.read_parquet(path)
    cols = list(df.columns)
    print("=" * 70)
    print(f"DARSHAN TOTAL: {path}")
    print("=" * 70)
    print(f"shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    mod_counts: Counter[str] = Counter()
    for c in cols:
        if c.startswith("total_"):
            rest = c[len("total_") :]
            mod = rest.split("_")[0] if "_" in rest else rest
            mod_counts[mod] += 1
    print("total_* columns by module prefix:")
    for k, v in mod_counts.most_common():
        print(f"  {k}: {v}")

    print("\nMeta / non-total columns:")
    for c in cols:
        if not c.startswith("total_"):
            print(f"  {c}: {df[c].dtype}")

    byte_cols = [c for c in cols if "BYTE" in c.upper()]
    print(f"\nColumns with BYTE in name ({len(byte_cols)}):")
    for c in sorted(byte_cols):
        print(f"  {c}")

    for name in ("runtime", "total_POSIX_BYTES_READ", "total_POSIX_BYTES_WRITTEN"):
        if name in df.columns:
            s = df[name]
            print(f"\n{name}: dtype={s.dtype}, non-null={s.notna().sum()}")


def summarize_detail(path: Path) -> None:
    df = pd.read_parquet(path)
    print("\n" + "=" * 70)
    print(f"DARSHAN DETAIL: {path}")
    print("=" * 70)
    print(f"shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
    print(df.dtypes.value_counts())
    byte_cols = [c for c in df.columns if "BYTE" in c.upper()]
    print(f"\nColumns with BYTE in name ({len(byte_cols)}):")
    for c in sorted(byte_cols):
        print(f"  {c}")
    print("\nFirst 20 column names:")
    for c in df.columns[:20]:
        print(f"  {c}: {df[c].dtype}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--total", type=Path, default=root / "data" / "App10.parquet")
    p.add_argument("--detail", type=Path, default=None, help="Optional detail parquet path")
    args = p.parse_args()

    if not args.total.exists():
        raise SystemExit(f"Missing total file: {args.total}")

    summarize_total(args.total)

    detail = args.detail
    if detail is None:
        candidates = sorted((root / "data").glob("2021/**/*.parquet"))
        candidates = [c for c in candidates if c.name != "App10.parquet"]
        if candidates:
            detail = candidates[0]
    if detail is not None and detail.exists():
        summarize_detail(detail)
    else:
        print("\n(No detail parquet found; pass --detail PATH)")


if __name__ == "__main__":
    main()
