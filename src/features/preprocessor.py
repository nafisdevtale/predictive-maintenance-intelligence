"""
Preprocessing and feature engineering pipeline for Predictive Maintenance Intelligence System.
Guarantees zero-leakage transformations, scikit-learn Pipeline compatibility, and Joblib serialization.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_PREDICTOR_COLUMNS,
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)

# Complete list of numeric features post-engineering
ALL_NUMERIC_FEATURES: List[str] = NUMERIC_FEATURES + ENGINEERED_FEATURES


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer that computes physics-based domain features:
    1. 'Temp difference [K]' = 'Process temperature [K]' - 'Air temperature [K]'
    2. 'Power [W]' = 'Torque [Nm]' * 'Rotational speed [rpm]' * (2 * pi / 60)
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """
        Fit method (stateless; validates input columns).
        """
        self._validate_input_columns(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates and appends engineered features without mutating input X.
        """
        self._validate_input_columns(X)
        X_df = X.copy()

        # 1. Temperature difference (Heat dissipation proxy)
        X_df["Temp difference [K]"] = (
            X_df["Process temperature [K]"] - X_df["Air temperature [K]"]
        )

        # 2. Mechanical Power = Torque (Nm) * Angular velocity (rad/s)
        # Angular velocity = Rotational speed (rpm) * 2 * pi / 60
        angular_velocity = X_df["Rotational speed [rpm]"] * (2.0 * np.pi / 60.0)
        X_df["Power [W]"] = X_df["Torque [Nm]"] * angular_velocity

        return X_df

    def _validate_input_columns(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame as input, got {type(X).__name__}")

        required = [
            "Process temperature [K]",
            "Air temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
        ]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"FeatureEngineer missing required columns: {missing}")

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if input_features is None:
            return ALL_PREDICTOR_COLUMNS + ENGINEERED_FEATURES
        return list(input_features) + ENGINEERED_FEATURES


def create_preprocessor() -> Pipeline:
    """
    Creates an unfitted scikit-learn Pipeline encapsulating:
    1. FeatureEngineer: generates domain features ('Temp difference [K]', 'Power [W]').
    2. ColumnTransformer:
       - StandardScaler for all numeric features (raw sensors + engineered).
       - OneHotEncoder for 'Type' with handle_unknown='ignore'.

    IMPORTANT: This pipeline is returned unfitted. It must only be fitted on training
    data (e.g. during model training), never on the full dataset or test split.

    Returns
    -------
    Pipeline
        Unfitted scikit-learn preprocessing pipeline.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                ALL_NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    preprocessor = Pipeline(
        steps=[
            ("feature_engineer", FeatureEngineer()),
            ("column_transformer", column_transformer),
        ]
    )
    return preprocessor


def get_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Extracts legitimate predictors and target from the raw dataset and performs
    a strictly reproducible stratified train/test split.

    Guarantees:
    - Never mutates the source DataFrame.
    - Excludes all identifier columns ('UDI', 'Product ID') and all failure mode
      indicators ('TWF', 'HDF', 'PWF', 'OSF', 'RNF') from X.
    - Preserves class balance across both splits using stratified sampling.
    - Verifies both splits contain both positive and negative examples.

    Parameters
    ----------
    df : pd.DataFrame
        Full validated dataset.
    test_size : float
        Proportion of dataset reserved for testing (default 0.20).
    random_state : int
        Seed for deterministic split (default 42).

    Returns
    -------
    X_train : pd.DataFrame
        Training predictors (80%).
    X_test : pd.DataFrame
        Held-out testing predictors (20%).
    y_train : pd.Series
        Training target labels.
    y_test : pd.Series
        Held-out testing target labels.
    """
    if df is None or df.empty:
        raise ValueError("Cannot split empty DataFrame.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    # Validate target values
    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    if not unique_targets.issubset({0, 1}) or len(unique_targets) < 2:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' must be binary and contain both classes {0, 1}. Found: {unique_targets}"
        )

    # Validate predictor columns
    missing_predictors = [c for c in ALL_PREDICTOR_COLUMNS if c not in df.columns]
    if missing_predictors:
        raise ValueError(f"Missing predictor columns in DataFrame: {missing_predictors}")

    # Explicit selection of predictors (Strict leakage exclusion)
    X = df[ALL_PREDICTOR_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy().astype(int)

    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True,
    )

    # Verify both splits contain both classes
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training set does not contain both target classes after split.")
    if len(np.unique(y_test)) < 2:
        raise ValueError("Test set does not contain both target classes after split.")

    return X_train, X_test, y_train, y_test
