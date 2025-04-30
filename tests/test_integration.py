"""Integration tests for credit default prediction system."""

import os
from pathlib import Path
from unittest.mock import patch

import mlflow
import pandas as pd
import pytest

from creditrisk.core.validation import validate_dataset
from creditrisk.data.preproc import preprocess_df
from creditrisk.models.predict import batch_predict
from creditrisk.models.train import (
    get_or_create_experiment,
    run_hyperopt,
    train,
    train_cv,
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


def test_end_to_end_training_pipeline(sample_data, tmp_path, temp_mlflow_home) -> None:
    """Test complete training pipeline from data preprocessing to model training."""
    # Setup temporary directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Save sample data
    input_path = data_dir / "credit_data.csv"
    sample_data.to_csv(input_path, index=False)

    with (
        patch("creditrisk.models.train.MODELS_DIR", models_dir),
        patch("creditrisk.data.preproc.PROCESSED_DATA_DIR", data_dir),
    ):

        # Step 1: Data Validation
        df = pd.read_csv(input_path)
        validated_df = validate_dataset(df)
        assert not validated_df.isnull().any().any()
        assert validated_df["EDUCATION"].between(1, 4).all()

        # Step 2: Preprocessing
        processed_path = preprocess_df(input_path)
        processed_df = pd.read_csv(processed_path)

        # Verify preprocessing results
        assert "EDUCATION_graduate" in processed_df.columns
        assert "UTILIZATION_RATIO_1" in processed_df.columns
        assert (processed_df.select_dtypes(include=["number"]) != float("inf")).all().all()

        # Step 3: Training Pipeline
        y = processed_df.pop("default.payment.next.month")
        X = processed_df

        categorical_indices = [
            i
            for i, col in enumerate(X.columns)
            if any(prefix in col for prefix in ["EDUCATION_", "MARRIAGE_", "PAY_"])
        ]

        # Create experiment
        experiment_name = "test_credit_default"
        experiment_id = get_or_create_experiment(experiment_name)
        mlflow.set_experiment(experiment_id=experiment_id)

        # Run hyperparameter optimization
        best_params_path = run_hyperopt(
            X,
            y,
            categorical_indices,
            test_size=0.2,
            n_trials=2,
        )
        assert Path(best_params_path).exists()

        # Run cross-validation
        params = pd.read_pickle(best_params_path)
        cv_output_path = train_cv(X, y, categorical_indices, params)
        assert Path(cv_output_path).exists()

        # Train final model
        cv_results = pd.read_csv(cv_output_path)
        model_path, model_params_path = train(
            X,
            y,
            categorical_indices,
            params=params,
            cv_results=cv_results,
        )

        # Verify training artifacts
        assert Path(model_path).exists()
        assert Path(model_params_path).exists()

        # Verify MLflow tracking
        runs = mlflow.search_runs(experiment_id)
        assert not runs.empty
        assert "params.depth" in runs.columns
        assert "metrics.f1_cv_mean" in runs.columns


def test_mlflow_logging_verification(sample_data, tmp_path, temp_mlflow_home) -> None:
    """Test MLflow logging functionality."""
    # Setup
    experiment_name = "test_logging"
    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("test_param", "test_value")
        mlflow.log_metric("test_metric", 0.95)

        # Create and log artifact
        artifact_path = tmp_path / "test.txt"
        artifact_path.write_text("test content")
        mlflow.log_artifact(str(artifact_path))

    # Verify logging
    runs = mlflow.search_runs(experiment_id)
    assert len(runs) == 1
    run = runs.iloc[0]
    assert run["params.test_param"] == "test_value"
    assert run["metrics.test_metric"] == 0.95


def test_batch_prediction_workflow(sample_data, tmp_path) -> None:
    """Test batch prediction workflow."""
    # Setup
    input_path = tmp_path / "test_input.csv"
    output_path = tmp_path / "predictions.csv"
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Prepare test data
    test_data = sample_data.copy()
    test_data.to_csv(input_path, index=False)

    with (
        patch("creditrisk.models.predict.MODELS_DIR", model_dir),
        patch("catboost.CatBoostClassifier") as mock_model,
    ):

        # Mock model predictions
        mock_model.return_value.predict_proba.return_value = np.random.random((len(test_data), 2))

        # Run batch predictions
        batch_predict(
            input_path=input_path,
            output_path=output_path,
            model_path=model_dir / "model.cbm",
        )

        # Verify predictions
        predictions = pd.read_csv(output_path)
        assert len(predictions) == len(test_data)
        assert "prediction" in predictions.columns
        assert "probability" in predictions.columns
        assert predictions["prediction"].isin([0, 1]).all()
        assert predictions["probability"].between(0, 1).all()


def test_error_handling(sample_data, tmp_path) -> None:
    """Test error handling in integration scenarios."""
    # Test invalid data handling
    invalid_data = sample_data.copy()
    invalid_data.loc[0, "EDUCATION"] = 10  # Invalid education value

    validated_df = validate_dataset(invalid_data)
    assert validated_df.loc[0, "EDUCATION"] == 4  # Should be corrected to "other"

    # Test missing columns handling
    missing_cols_data = sample_data.drop(columns=["PAY_0", "BILL_AMOUNT1"])
    with pytest.raises(KeyError):
        preprocess_df(missing_cols_data)

    # Test invalid model path
    with pytest.raises(FileNotFoundError):
        batch_predict(
            input_path=tmp_path / "test.csv",
            output_path=tmp_path / "out.csv",
            model_path=tmp_path / "nonexistent_model.cbm",
        )


def test_model_registry_operations(temp_mlflow_home) -> None:
    """Test model registry operations."""
    model_name = "credit_default_test"

    # Register a dummy model
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=DummyClassifier(),
            artifact_path="model",
            registered_model_name=model_name,
        )

    # Verify model registration
    model_version = mlflow.tracking.MlflowClient().get_latest_versions(model_name)[0]
    assert model_version.name == model_name

    # Test model version transitions
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging",
    )

    # Verify transition
    updated_version = client.get_model_version(
        name=model_name,
        version=model_version.version,
    )
    assert updated_version.current_stage == "Staging"
