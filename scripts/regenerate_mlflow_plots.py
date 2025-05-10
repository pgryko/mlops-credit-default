"""Script to regenerate plots and update them in MLflow."""

import os
from pathlib import Path

import joblib
import mlflow
from mlflow.client import MlflowClient
import pandas as pd

from creditrisk.core.config import FIGURES_DIR, MODEL_NAME, MODELS_DIR
from creditrisk.models.train import plot_error_scatter


def main():
    """Regenerate plots and update them in MLflow."""
    # Setup MLflow environment
    MLFLOW_DIR = Path(".mlflow")
    MLFLOW_DB_DIR = MLFLOW_DIR / "db"
    MLFLOW_ARTIFACTS_DIR = MLFLOW_DIR / "artifacts"

    # Set MLflow environment variables
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{MLFLOW_DB_DIR}/mlflow.db"
    os.environ["MLFLOW_ARTIFACT_ROOT"] = str(MLFLOW_ARTIFACTS_DIR.absolute())

    # Set tracking URI
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient(mlflow.get_tracking_uri())

    # Ensure the figures directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load CV results
    cv_results_path = MODELS_DIR / "cv_results.csv"
    cv_results = pd.read_csv(cv_results_path)

    # Load model parameters
    model_params_path = MODELS_DIR / "model_params.pkl"
    params = joblib.load(model_params_path)

    # Get the latest model
    model_info = None
    try:
        for model_version in client.search_model_versions(f"name='{MODEL_NAME}'"):
            if (
                model_version.current_stage in ["Production", "Staging"]
                or model_version.current_stage == "None"
            ):
                model_info = model_version
                break
    except Exception as e:
        print(f"Error finding model: {e}")
        return

    if model_info is None:
        print("No model found. Please train a model first.")
        return

    # Start a new MLflow run with the existing run ID
    with mlflow.start_run(run_id=model_info.run_id):
        print(f"Updating plots for run ID: {model_info.run_id}")

        # Regenerate F1 score plot
        fig1 = plot_error_scatter(
            df_plot=cv_results,
            name="Mean F1 Score",
            title="Cross-Validation (N=5) Mean F1 score with Error Bands",
            xtitle="Training Steps",
            ytitle="Performance Score",
            yaxis_range=[0.4, 1.0],  # Adjusted to show the actual data range
        )

        # Regenerate Logloss plot
        fig2 = plot_error_scatter(
            cv_results,
            x="iterations",
            y="test-Logloss-mean",
            err="test-Logloss-std",
            name="Mean Logloss",
            title="Cross-Validation (N=5) Mean Logloss with Error Bands",
            xtitle="Training Steps",
            ytitle="Logloss",
        )

        # Log the regenerated plots to MLflow
        mlflow.log_figure(fig1, "test-F1-mean_vs_iterations.png")
        mlflow.log_figure(fig2, "test-logloss-mean_vs_iterations.png")

        print(f"Plots regenerated and updated in MLflow run {model_info.run_id}")


if __name__ == "__main__":
    main()
