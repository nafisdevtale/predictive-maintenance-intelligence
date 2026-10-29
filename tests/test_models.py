"""
Unit tests for classification pipelines, model inference, metric calculations, and serialization.
Runs strictly offline with small, fast test fixtures without relying on hardcoded performance claims.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    ALL_PREDICTOR_COLUMNS,
    ID_COLUMNS,
    LEAKAGE_FAILURE_COLUMNS,
    TARGET_COLUMN,
)
from src.models.evaluate import (
    calculate_classification_metrics,
    find_optimal_threshold,
)
from src.models.train import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    get_feature_names_from_fitted_pipeline,
)


@pytest.fixture
def small_train_fixture():
    """Creates a small 30-sample dataset of valid predictors and binary labels."""
    np.random.seed(42)
    n = 30
    y = np.array([0] * 25 + [1] * 5)
    types = ["L"] * 18 + ["M"] * 9 + ["H"] * 3
    df = pd.DataFrame(
        {
            "Type": types,
            "Air temperature [K]": np.random.uniform(297, 303, n),
            "Process temperature [K]": np.random.uniform(307, 313, n),
            "Rotational speed [rpm]": np.random.randint(1300, 1700, n),
            "Torque [Nm]": np.random.uniform(30, 60, n),
            "Tool wear [min]": np.random.randint(0, 180, n),
        }
    )
    return df, pd.Series(y)


def test_pipeline_construction():
    """Ensures pipelines contain both preprocessor and classifier stages."""
    lr_pipe = build_logistic_regression_pipeline()
    rf_pipe = build_random_forest_pipeline()

    assert "preprocessor" in lr_pipe.named_steps
    assert "classifier" in lr_pipe.named_steps
    assert "preprocessor" in rf_pipe.named_steps
    assert "classifier" in rf_pipe.named_steps


def test_logistic_regression_fit_and_predict(small_train_fixture):
    """Verifies that Logistic Regression fits, predicts binary classes, and outputs valid probabilities."""
    X, y = small_train_fixture
    pipe = build_logistic_regression_pipeline()
    pipe.fit(X, y)

    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_random_forest_fit_and_predict(small_train_fixture):
    """Verifies that Random Forest fits, predicts binary classes, and outputs valid probabilities."""
    X, y = small_train_fixture
    pipe = build_random_forest_pipeline()
    pipe.fit(X, y)

    preds = pipe.predict(X)
    probs = pipe.predict_proba(X)

    assert len(preds) == len(X)
    assert set(np.unique(preds)).issubset({0, 1})
    assert probs.shape == (len(X), 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_feature_names_extraction_from_fitted_pipeline(small_train_fixture):
    """Verifies that 10 output features (7 numeric + 3 one-hot) are extracted accurately."""
    X, y = small_train_fixture
    pipe = build_random_forest_pipeline()
    pipe.fit(X, y)

    names = get_feature_names_from_fitted_pipeline(pipe)
    assert len(names) == 10
    assert "Temp difference [K]" in names
    assert "Power [W]" in names
    assert "Type_L" in names or "Type_M" in names or "Type_H" in names


def test_joblib_pipeline_serialization(small_train_fixture, tmp_path):
    """Verifies complete serialization and identical predictions upon reloading."""
    X, y = small_train_fixture
    pipe = build_random_forest_pipeline()
    pipe.fit(X, y)

    orig_probs = pipe.predict_proba(X)

    save_path = tmp_path / "rf_test.joblib"
    joblib.dump(pipe, save_path)

    reloaded_pipe = joblib.load(save_path)
    reloaded_probs = reloaded_pipe.predict_proba(X)

    assert np.allclose(orig_probs, reloaded_probs)


def test_forbidden_columns_guarantee():
    """Verifies that forbidden leakage columns are never included in ALL_PREDICTOR_COLUMNS."""
    forbidden = ID_COLUMNS + LEAKAGE_FAILURE_COLUMNS + [TARGET_COLUMN]
    for col in forbidden:
        assert col not in ALL_PREDICTOR_COLUMNS


def test_calculate_classification_metrics_properties():
    """Verifies that calculated metrics are mathematically bounded and consistent."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.7, 0.9])

    metrics = calculate_classification_metrics(y_true, y_prob, threshold=0.5)

    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["f2_score"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert 0.0 <= metrics["average_precision_pr_auc"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    cm = metrics["confusion_matrix"]
    total = cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
    assert total == len(y_true)


def test_find_optimal_threshold_f2_optimization():
    """Verifies threshold search on training probabilities."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.15, 0.35, 0.45, 0.48, 0.85])

    opt = find_optimal_threshold(y_true, y_prob, metric="f2")

    assert "optimal_threshold" in opt
    assert 0.05 <= opt["optimal_threshold"] <= 0.95
    assert opt["best_score"] >= opt["default_0_5_metrics"]["f2_score"]
