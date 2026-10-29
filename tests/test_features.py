"""
Unit tests for feature engineering, preprocessor pipeline, and leakage-prevention splitting.
"""

import io
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    ALL_PREDICTOR_COLUMNS,
    ID_COLUMNS,
    LEAKAGE_FAILURE_COLUMNS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.data.loader import load_dataset
from src.features.preprocessor import (
    ALL_NUMERIC_FEATURES,
    FeatureEngineer,
    create_preprocessor,
    get_train_test_split,
)


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Creates a controlled sample DataFrame of legitimate predictors."""
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H", "L"],
            "Air temperature [K]": [300.0, 301.0, 298.0, 302.0],
            "Process temperature [K]": [310.0, 311.5, 309.0, 312.0],
            "Rotational speed [rpm]": [1500, 1400, 1600, 1350],
            "Torque [Nm]": [40.0, 45.0, 38.0, 50.0],
            "Tool wear [min]": [10, 20, 5, 100],
        }
    )


def test_feature_engineer_calculations(sample_feature_df):
    """Verifies the mathematical formulas for Temp difference and Mechanical Power."""
    fe = FeatureEngineer()
    fe.fit(sample_feature_df)
    transformed = fe.transform(sample_feature_df)

    # Check Temp difference: 310.0 - 300.0 = 10.0
    assert np.isclose(transformed.loc[0, "Temp difference [K]"], 10.0)
    assert np.isclose(transformed.loc[1, "Temp difference [K]"], 10.5)

    # Check Power: Torque * rpm * 2 * pi / 60
    # For row 0: 40.0 * 1500 * 2 * pi / 60 = 40.0 * 1500 * pi / 30 = 6283.1853 W
    expected_power_0 = 40.0 * 1500.0 * (2.0 * np.pi / 60.0)
    assert np.isclose(transformed.loc[0, "Power [W]"], expected_power_0)
    assert "Temp difference [K]" in transformed.columns
    assert "Power [W]" in transformed.columns


def test_feature_engineer_does_not_mutate_input(sample_feature_df):
    """Ensures input DataFrame is unmodified during transform."""
    original_cols = list(sample_feature_df.columns)
    fe = FeatureEngineer()
    fe.transform(sample_feature_df)
    assert list(sample_feature_df.columns) == original_cols


def test_feature_engineer_missing_required_column(sample_feature_df):
    """Ensures ValueError is raised if a required sensor column is missing."""
    bad_df = sample_feature_df.drop(columns=["Torque [Nm]"])
    fe = FeatureEngineer()
    with pytest.raises(ValueError, match="missing required columns"):
        fe.transform(bad_df)


def test_preprocessor_pipeline_transformation(sample_feature_df):
    """Verifies that create_preprocessor pipeline outputs expected 10 columns with finite numbers."""
    preprocessor = create_preprocessor()
    X_transformed = preprocessor.fit_transform(sample_feature_df)

    # 7 numeric features + 3 one-hot Type categories (H, L, M) = 10 columns
    assert X_transformed.shape == (4, 10)
    assert np.isfinite(X_transformed).all()


def test_preprocessor_handles_unknown_category(sample_feature_df):
    """Verifies that unseen categorical values do not crash and are encoded as all zeros."""
    preprocessor = create_preprocessor()
    preprocessor.fit(sample_feature_df)

    test_unseen = sample_feature_df.copy()
    test_unseen.loc[0, "Type"] = "UnknownVariant"

    transformed_unseen = preprocessor.transform(test_unseen)
    assert transformed_unseen.shape == (4, 10)
    assert np.isfinite(transformed_unseen).all()
    # The last 3 columns correspond to Type_H, Type_L, Type_M. All should be 0 for unknown
    assert np.all(transformed_unseen[0, -3:] == 0.0)


def test_joblib_serialization_compatibility(sample_feature_df, tmp_path):
    """Verifies that the preprocessor pipeline serializes and reloads cleanly via Joblib."""
    preprocessor = create_preprocessor()
    X_orig = preprocessor.fit_transform(sample_feature_df)

    artifact_path = tmp_path / "test_preprocessor.joblib"
    joblib.dump(preprocessor, artifact_path)

    loaded_preprocessor = joblib.load(artifact_path)
    X_reloaded = loaded_preprocessor.transform(sample_feature_df)

    assert np.allclose(X_orig, X_reloaded)


def test_leakage_exclusion_guarantee():
    """Verifies that none of the 7 leakage/identifier columns are in ALL_PREDICTOR_COLUMNS."""
    forbidden = ID_COLUMNS + LEAKAGE_FAILURE_COLUMNS + [TARGET_COLUMN]
    for col in forbidden:
        assert col not in ALL_PREDICTOR_COLUMNS


def test_get_train_test_split_on_real_data():
    """Validates reproducible stratified split on actual local dataset."""
    if not RAW_DATA_PATH.exists():
        pytest.skip("Dataset file not available locally.")

    df = load_dataset(RAW_DATA_PATH, validate=True, strict_row_count=True)
    orig_shape = df.shape

    X_train, X_test, y_train, y_test = get_train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Verify original DataFrame was not mutated
    assert df.shape == orig_shape

    # Check split dimensions
    assert len(X_train) == 8000
    assert len(X_test) == 2000
    assert len(y_train) == 8000
    assert len(y_test) == 2000

    # Check predictor columns strictly match ALL_PREDICTOR_COLUMNS
    assert list(X_train.columns) == ALL_PREDICTOR_COLUMNS
    assert list(X_test.columns) == ALL_PREDICTOR_COLUMNS

    # Ensure no leakage columns are in X
    for col in ID_COLUMNS + LEAKAGE_FAILURE_COLUMNS + [TARGET_COLUMN]:
        assert col not in X_train.columns
        assert col not in X_test.columns

    # Verify stratified balance: target proportion must be virtually identical
    train_rate = y_train.mean()
    test_rate = y_test.mean()
    pop_rate = df[TARGET_COLUMN].mean()

    assert np.isclose(train_rate, pop_rate, atol=0.002)
    assert np.isclose(test_rate, pop_rate, atol=0.002)

    # Verify reproducibility (same random_state yields identical split)
    X_tr2, X_te2, y_tr2, y_te2 = get_train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    pd.testing.assert_frame_equal(X_train, X_tr2)
    pd.testing.assert_series_equal(y_train, y_tr2)
