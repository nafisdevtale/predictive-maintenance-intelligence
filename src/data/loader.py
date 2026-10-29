"""
Data loading and validation module for Predictive Maintenance Intelligence System.
Provides reproducible dataset loading, strict schema checks, and integrity verification.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from src.config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    ID_COLUMNS,
    LEAKAGE_FAILURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    FEATURE_RANGES,
)

EXPECTED_COLUMNS = (
    ID_COLUMNS
    + CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
    + [TARGET_COLUMN]
    + LEAKAGE_FAILURE_COLUMNS
)


def validate_dataset(df: pd.DataFrame, strict_row_count: bool = False) -> Dict[str, Any]:
    """
    Performs comprehensive schema and domain integrity validation on the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded predictive maintenance DataFrame.
    strict_row_count : bool
        If True, enforces exactly 10,000 rows (for benchmark verification).

    Returns
    -------
    dict
        Validation report details.

    Raises
    ------
    ValueError
        If schema, data types, nulls, or value constraints are violated.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None.")

    # 1. Expected Row Count Check
    if strict_row_count and len(df) != 10000:
        raise ValueError(f"Expected exactly 10,000 rows, found {len(df)}.")

    # 2. Required Columns Check
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # 3. Missing Value (Null / NaN) Check
    null_counts = df[EXPECTED_COLUMNS].isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Dataset contains {total_nulls} null values in columns: {bad_cols}")

    # 4. Target & Failure Mode Binary Check
    binary_columns = [TARGET_COLUMN] + LEAKAGE_FAILURE_COLUMNS
    for col in binary_columns:
        unique_vals = set(df[col].dropna().unique())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Column '{col}' must only contain binary values {0, 1}, found {unique_vals}"
            )

    # 5. Product Type Categorical Check
    valid_types = set(FEATURE_RANGES["Type"]["options"])
    type_vals = set(df["Type"].dropna().unique())
    if not type_vals.issubset(valid_types):
        raise ValueError(f"Column 'Type' contains invalid values: {type_vals - valid_types}")

    # 6. Physical Sensor Plausibility Checks
    if (df["Air temperature [K]"] <= 0).any():
        raise ValueError("Air temperature contains non-positive values.")
    if (df["Process temperature [K]"] <= 0).any():
        raise ValueError("Process temperature contains non-positive values.")
    if (df["Rotational speed [rpm]"] <= 0).any():
        raise ValueError("Rotational speed contains non-positive values.")
    if (df["Torque [Nm]"] <= 0).any():
        raise ValueError("Torque contains non-positive values.")
    if (df["Tool wear [min]"] < 0).any():
        raise ValueError("Tool wear contains negative values.")

    return {
        "status": "VALID",
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": 0,
        "valid_types": sorted(list(type_vals)),
        "binary_columns_checked": binary_columns,
    }


def load_dataset(
    filepath: Optional[Path] = None,
    validate: bool = True,
    strict_row_count: bool = False,
) -> pd.DataFrame:
    """
    Loads the CSV dataset from disk and optionally runs validation checks.

    Parameters
    ----------
    filepath : Path, optional
        Path to the CSV file. If None, uses RAW_DATA_PATH from config.
    validate : bool
        Whether to run validate_dataset() upon loading.
    strict_row_count : bool
        Whether to enforce exact 10,000 rows.

    Returns
    -------
    pd.DataFrame
        Loaded and verified pandas DataFrame.
    """
    csv_path = Path(filepath) if filepath else RAW_DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found at {csv_path}. "
            f"Run 'src/data/download.py' or check data/README.md for download instructions."
        )

    df = pd.read_csv(csv_path)

    if validate:
        validate_dataset(df, strict_row_count=strict_row_count)

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes real, non-fabricated dataset statistics and integrity metrics.

    Parameters
    ----------
    df : pd.DataFrame
        The validated DataFrame.

    Returns
    -------
    dict
        Structured summary statistics including class balance, failure types,
        and sensor telemetry distributions.
    """
    total_records = len(df)
    failure_counts = df[TARGET_COLUMN].value_counts().to_dict()
    failure_rate = float(df[TARGET_COLUMN].mean()) * 100.0

    # Failure mode occurrences
    failure_modes = {}
    for col in LEAKAGE_FAILURE_COLUMNS:
        failure_modes[col] = int(df[col].sum())

    # Check duplicates
    duplicate_rows_all = int(df.duplicated().sum())
    duplicate_features = int(df.duplicated(subset=CATEGORICAL_FEATURES + NUMERIC_FEATURES).sum())

    # Type distribution
    type_distribution = df["Type"].value_counts().to_dict()

    # Numeric sensor statistics
    sensor_stats = {}
    for col in NUMERIC_FEATURES:
        sensor_stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "median": float(df[col].median()),
        }

    return {
        "total_records": total_records,
        "total_columns": len(df.columns),
        "target_distribution": {
            "Normal (0)": int(failure_counts.get(0, 0)),
            "Failure (1)": int(failure_counts.get(1, 0)),
            "failure_rate_pct": round(failure_rate, 3),
        },
        "failure_mode_counts": failure_modes,
        "type_distribution": type_distribution,
        "duplicates": {
            "all_columns": duplicate_rows_all,
            "feature_subset": duplicate_features,
        },
        "sensor_statistics": sensor_stats,
    }
