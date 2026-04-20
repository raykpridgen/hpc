"""Smoke tests for Parquet surrogate pipeline (design.md §6, §14)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from darshan_surrogate.leakage import (
    assert_no_forbidden_columns,
    apply_correlation_leakage_filter,
)


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
