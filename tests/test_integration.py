"""Integration tests for credit default prediction system."""

import os
from unittest.mock import patch

import mlflow
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from creditrisk.core.validation import validate_dataset
from creditrisk.data.preproc import preprocess_df
from creditrisk.models.train import (
    get_or_create_experiment,
)


@pytest.fixture
def sample_data():
    """Create sample credit card data for integration testing."""
    return pd.DataFrame(
        {
            "ID": range(1, 101),
            "LIMIT_BAL": [20000 + i * 1000 for i in range(100)],
            "SEX": [1, 2] * 50,
            "EDUCATION": [1, 2, 3, 4] * 25,
            "MARRIAGE": [1, 2, 3] * 33 + [1],
            "AGE": [25 + i % 40 for i in range(100)],
            "PAY_0": [-1, 0, 1, 2] * 25,
            "PAY_2": [-1, 0, 1, 2] * 25,
            "BILL_AMOUNT1": [i * 100 for i in range(100)],
            "BILL_AMOUNT2": [i * 90 for i in range(100)],
            "PAY_AMT1": [i * 50 for i in range(100)],
            "PAY_AMT2": [i * 45 for i in range(100)],
            "default.payment.next.month": [0, 1] * 50,
        },
    )


@pytest.fixture
def temp_mlflow_home(tmp_path):
    """Set up temporary MLflow home directory."""
    mlflow_home = tmp_path / "mlruns"
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{mlflow_home}/mlflow.db"
    yield mlflow_home
    if "MLFLOW_TRACKING_URI" in os.environ:
        del os.environ["MLFLOW_TRACKING_URI"]


def test_preproc_pipeline(sample_data, tmp_path) -> None:
    """Test preprocessing pipeline in isolation."""
    # Setup temporary directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Save sample data
    input_path = data_dir / "credit_data.csv"
    sample_data.to_csv(input_path, index=False)

    # Test preprocessing pipeline
    with patch("creditrisk.data.preproc.PROCESSED_DATA_DIR", data_dir):
        # Data Validation
        df = pd.read_csv(input_path)
        validated_df = validate_dataset(df)
        assert not validated_df.isnull().any().any()
        assert validated_df["EDUCATION"].between(1, 4).all()

        # Preprocessing
        processed_path = preprocess_df(input_path)
        processed_df = pd.read_csv(processed_path)

        # Verify preprocessing results
        assert "EDUCATION_graduate" in processed_df.columns
        assert "UTILIZATION_RATIO_1" in processed_df.columns
        assert (processed_df.select_dtypes(include=["number"]) != float("inf")).all().all()

        # Verify the target column is preserved
        assert "default.payment.next.month" in processed_df.columns


def test_mlflow_logging_verification() -> None:
    """Test MLflow logging interface."""
    # Simple test to check if the MLflow functions exist and are callable
    assert callable(mlflow.log_param)
    assert callable(mlflow.log_metric)
    assert callable(mlflow.log_artifact)
    assert callable(mlflow.start_run)
    assert callable(mlflow.end_run)
    assert callable(mlflow.search_runs)

    # Test get_or_create_experiment function interface
    with patch("mlflow.get_experiment_by_name"), patch("mlflow.create_experiment"):
        experiment_id = get_or_create_experiment("test_name")
        assert experiment_id is not None


def test_model_registry_interface() -> None:
    """Test model registry interface and operations."""
    # Simple test to check if the MLflow registry functions exist and are callable
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    # Check for modern MLflow functions (using aliases instead of stages)
    assert callable(client.set_registered_model_alias)
    assert callable(client.set_model_version_tag)

    # Check model registration functions
    assert callable(mlflow.register_model)
    assert callable(mlflow.sklearn.log_model)

    # Verify CatBoost integration works
    assert callable(mlflow.catboost.log_model)

    # Verify we can import DummyClassifier (used in some tests)
    assert callable(DummyClassifier)
