"""Smoke tests for Parquet surrogate pipeline (design.md §6, §14)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darshan_surrogate.leakage import (
    assert_no_forbidden_columns,
    apply_correlation_leakage_filter,
    apply_timestamp_prefilter,
)
from darshan_surrogate.detail_join import join_detail_features
from darshan_surrogate.features import apply_group_filters
from darshan_surrogate.training import train_surrogate
from run_eda_sweep import _expand_total_glob, _resolve_detail_root


def test_assert_no_forbidden_columns_ok() -> None:
    assert_no_forbidden_columns(pd.Index(["total_H5D_READS", "month_1", "app_id_code"]))


def test_assert_no_forbidden_columns_raises() -> None:
    with pytest.raises(ValueError, match="runtime"):
        assert_no_forbidden_columns(pd.Index(["total_H5D_READS", "runtime"]))


def test_correlation_filter_drops_perfect_proxy_on_train() -> None:
    rng = np.random.default_rng(42)
    n = 120
    y = rng.random(n)
    X = pd.DataFrame(
        {
            "total_proxy": y.copy(),
            "month_1": 0,
        }
    )
    train_pos = np.arange(96)
    X2, recs = apply_correlation_leakage_filter(X, train_pos, y, max_abs_corr=0.95)
    assert "total_proxy" not in X2.columns
    assert len(recs) >= 1
    assert recs[0]["column"] == "total_proxy"


def test_correlation_filter_keeps_month_columns() -> None:
    """month_* are not candidates for correlation filter (design.md §6)."""
    rng = np.random.default_rng(0)
    n = 50
    y = rng.random(n)
    X = pd.DataFrame(
        {
            "month_1": y.copy(),
            "total_safe": rng.random(n),
        }
    )
    train_pos = np.arange(40)
    X2, recs = apply_correlation_leakage_filter(X, train_pos, y, max_abs_corr=0.95)
    assert "month_1" in X2.columns
    assert "total_safe" in X2.columns
    assert not recs


def test_import_train_entrypoint() -> None:
    import train_parquet_surrogate  # noqa: F401 — smoke import path


def test_timestamp_prefilter_drops_timestamp_columns_only() -> None:
    X = pd.DataFrame(
        {
            "total_POSIX_F_OPEN_START_TIMESTAMP": [1.0, 2.0, 3.0],
            "total_POSIX_F_READ_TIME": [0.1, 0.2, 0.3],
            "total_POSIX_READS": [5, 6, 7],
        }
    )
    X2, dropped = apply_timestamp_prefilter(X)
    assert "total_POSIX_F_OPEN_START_TIMESTAMP" not in X2.columns
    assert "total_POSIX_F_READ_TIME" in X2.columns
    assert "total_POSIX_READS" in X2.columns
    assert dropped == ["total_POSIX_F_OPEN_START_TIMESTAMP"]


def test_train_surrogate_optuna_path_runs() -> None:
    rng = np.random.default_rng(7)
    n = 80
    X = pd.DataFrame(
        {
            "total_POSIX_READS": rng.normal(size=n),
            "total_MPIIO_HINTS": rng.normal(size=n),
            "month_1": rng.integers(0, 2, size=n),
            "app_id_code": rng.integers(0, 3, size=n),
        }
    )
    z = rng.normal(size=n)
    train_idx = np.arange(60)
    val_idx = np.arange(60, n)

    model, metrics = train_surrogate(
        X.iloc[train_idx],
        z[train_idx],
        X.iloc[val_idx],
        z[val_idx],
        n_optuna_trials=1,
        early_stopping_rounds=5,
    )
    assert model is not None
    assert "optuna_best_rmse" in metrics
    assert metrics["n_trials"] == 1


def test_join_detail_features_aggregates_across_shards(tmp_path) -> None:
    # Build detail hierarchy: YYYY/M/D/jobid-{0,1}.parquet
    detail_root = tmp_path / "detail"
    day_dir = detail_root / "2021" / "1" / "1"
    day_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "filename": ["f1", "f2"],
            "POSIX_BYTES_READ": [10.0, 20.0],
            "POSIX_BYTES_WRITTEN": [0.0, 5.0],
            "POSIX_F_VARIANCE_RANK_BYTES": [1.0, 3.0],
        }
    ).to_parquet(day_dir / "123-0.parquet", index=False)
    pd.DataFrame(
        {
            "filename": ["f2", "f3"],
            "POSIX_BYTES_READ": [30.0, 0.0],
            "POSIX_BYTES_WRITTEN": [5.0, 10.0],
            "POSIX_F_VARIANCE_RANK_BYTES": [5.0, 7.0],
        }
    ).to_parquet(day_dir / "123-1.parquet", index=False)

    # start_time for 2021-01-01 UTC.
    total_df = pd.DataFrame({"jobid": [123], "start_time": [1609459200]})
    out, stats = join_detail_features(total_df, detail_root, missing_threshold=0.2)

    assert stats["n_missing_detail"] == 0
    assert float(out["detail_n_files"].iloc[0]) == 3.0
    assert int(out["missing_detail"].iloc[0]) == 0
    assert float(out["detail_posix_var_mean"].iloc[0]) == 4.0


def test_apply_group_filters_include_and_exclude() -> None:
    X = pd.DataFrame(
        {
            "total_POSIX_READS": [1, 2],
            "total_MPIIO_HINTS": [0, 1],
            "month_1": [1, 0],
            "app_id_code": [0, 1],
        }
    )
    X2, dropped = apply_group_filters(X, include_groups=["posix", "calendar", "app"])
    assert list(X2.columns) == ["total_POSIX_READS", "month_1", "app_id_code"]
    assert "total_MPIIO_HINTS" in dropped


def test_run_sweep_path_helpers_support_parent_layout(tmp_path, monkeypatch) -> None:
    # Simulate repo/hpc (cwd) and sibling repo/darshan_share.
    repo = tmp_path / "repo"
    hpc = repo / "hpc"
    totals = repo / "darshan_share" / "darshan_total"
    details = repo / "darshan_share" / "darshan_detail"
    hpc.mkdir(parents=True, exist_ok=True)
    totals.mkdir(parents=True, exist_ok=True)
    details.mkdir(parents=True, exist_ok=True)
    (totals / "App1.parquet").write_text("x", encoding="utf-8")

    monkeypatch.chdir(hpc)
    m = _expand_total_glob("../darshan_share/darshan_total/App*.parquet")
    assert len(m) == 1
    d = _resolve_detail_root("../darshan_share/darshan_detail")
    assert d is not None and d.resolve() == details.resolve()
