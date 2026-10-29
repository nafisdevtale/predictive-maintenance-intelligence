"""
Unit tests for the Statistical Analysis Engine (SciPy hypothesis testing, FDR correction, Chi-Square).
Designed to run strictly offline with small, fast test fixtures.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import TARGET_COLUMN
from src.stats.hypothesis_testing import (
    compute_mann_whitney_tests,
    compute_chi2_type_test,
    run_full_statistical_analysis,
)


@pytest.fixture
def mock_stats_df() -> pd.DataFrame:
    """Creates a balanced fixture with known properties for testing."""
    np.random.seed(42)
    n = 20
    # Group 0 (Non-failure) has lower torque, higher rotational speed
    # Group 1 (Failure) has higher torque, lower rotational speed
    is_failure = np.array([0] * 10 + [1] * 10)
    torque = np.where(is_failure == 1, np.random.normal(55, 3, n), np.random.normal(38, 3, n))
    speed = np.where(is_failure == 1, np.random.normal(1350, 40, n), np.random.normal(1550, 40, n))
    types = ["L"] * 12 + ["M"] * 6 + ["H"] * 2

    return pd.DataFrame(
        {
            "Type": types,
            "Air temperature [K]": np.random.uniform(298, 302, n),
            "Process temperature [K]": np.random.uniform(308, 312, n),
            "Rotational speed [rpm]": speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": np.random.uniform(0, 200, n),
            TARGET_COLUMN: is_failure,
        }
    )


def test_mann_whitney_basic_properties(mock_stats_df):
    """Verifies that Mann-Whitney U test computes valid bounded stats and p-values."""
    res_df = compute_mann_whitney_tests(mock_stats_df)

    assert not res_df.empty
    assert len(res_df) == 7  # 5 raw numeric + 2 engineered

    for _, row in res_df.iterrows():
        # p-values must be within [0, 1]
        assert 0.0 <= row["raw_p_value"] <= 1.0
        assert 0.0 <= row["fdr_adjusted_p_value"] <= 1.0

        # FDR adjusted p-value must be >= raw p-value
        assert row["fdr_adjusted_p_value"] >= row["raw_p_value"]

        # Rank-biserial effect size must be in [-1, +1]
        assert -1.0 <= row["rank_biserial_r"] <= 1.0

        # Sample sizes must match
        assert row["n_non_failure"] == 10
        assert row["n_failure"] == 10

    # Torque in fixture is significantly higher in failure group
    torque_row = res_df[res_df["feature"] == "Torque [Nm]"].iloc[0]
    assert torque_row["rank_biserial_r"] > 0
    assert torque_row["median_difference"] > 0


def test_mann_whitney_insufficient_samples():
    """Verifies that fewer than 3 samples in a group generates a warning rather than crashing."""
    tiny_df = pd.DataFrame(
        {
            "Type": ["L", "L", "L"],
            "Air temperature [K]": [300.0, 301.0, 302.0],
            "Process temperature [K]": [310.0, 311.0, 312.0],
            "Rotational speed [rpm]": [1500, 1520, 1510],
            "Torque [Nm]": [40.0, 42.0, 41.0],
            "Tool wear [min]": [10, 20, 30],
            TARGET_COLUMN: [0, 0, 1],  # Only 1 failure
        }
    )

    res_df = compute_mann_whitney_tests(tiny_df)
    assert not res_df.empty
    for _, row in res_df.iterrows():
        assert row["warning"] is not None
        assert np.isnan(row["u_statistic"])


def test_mann_whitney_missing_target_raises(mock_stats_df):
    """Verifies that missing TARGET_COLUMN raises a ValueError."""
    bad_df = mock_stats_df.drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match="Target column"):
        compute_mann_whitney_tests(bad_df)


def test_chi2_type_test_properties(mock_stats_df):
    """Verifies chi-square calculation, degrees of freedom, and Cramer's V."""
    res = compute_chi2_type_test(mock_stats_df)

    assert res["status"] == "VALID"
    assert res["chi2_statistic"] >= 0.0
    assert 0.0 <= res["p_value"] <= 1.0
    assert 0.0 <= res["cramers_v"] <= 1.0
    assert res["degrees_of_freedom"] == 2  # 3 types (df = 3-1 = 2)
    assert "contingency_table" in res
    assert "breakdown_by_type" in res


def test_chi2_type_single_category_graceful_handling():
    """Verifies that degenerate contingency tables are handled gracefully."""
    single_type_df = pd.DataFrame(
        {
            "Type": ["L", "L", "L", "L"],
            TARGET_COLUMN: [0, 0, 1, 1],
        }
    )
    res = compute_chi2_type_test(single_type_df)
    assert res["status"] == "ERROR"
    assert "warning" in res


def test_run_full_statistical_analysis_export(mock_stats_df, tmp_path):
    """Verifies full execution pipeline and file serialization in outputs/."""
    report = run_full_statistical_analysis(mock_stats_df, output_dir=tmp_path)

    assert report["sample_size"] == 20
    assert "numeric_tests" in report
    assert "categorical_test" in report
    assert "interpretations" in report

    # Verify serialized files
    num_csv = tmp_path / "statistical_tests_numeric.csv"
    cat_csv = tmp_path / "contingency_table_type.csv"
    json_path = tmp_path / "statistical_analysis_report.json"

    assert num_csv.exists()
    assert cat_csv.exists()
    assert json_path.exists()

    # Verify JSON is valid and non-empty
    with open(json_path) as f:
        data = json.load(f)
    assert data["sample_size"] == 20
    assert len(data["numeric_tests"]) == 7
