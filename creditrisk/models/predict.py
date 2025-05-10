"""Functions for generating and explaining credit default predictions.

This module provides functions for making predictions using trained models,
generating SHAP-based explanations, and calculating feature importance.
It supports batch processing and detailed per-prediction explanations.

Functions:
    plot_shap: Generate SHAP plots and calculate feature group importance
    explain_prediction: Generate explanation for a single prediction
    predict: Make batch predictions with explanations


"""

import json
import os
from pathlib import Path
import sys

from catboost import CatBoostClassifier
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowException
from mlflow.client import MlflowClient
import numpy as np
import pandas as pd
import shap

from creditrisk.core.config import (
    FIGURES_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    categorical,
    target,
)
from creditrisk.core.metrics import (
    calculate_business_metrics,
    calculate_pr_auc,
    optimize_threshold,
)
from creditrisk.models.resolve import get_model_by_alias


def plot_shap(model: CatBoostClassifier, df_plot: pd.DataFrame) -> dict[str, float]:
    """Generate SHAP plots and calculate feature group importance.

    Creates SHAP summary plots for overall feature importance and for specific
    feature groups (demographics, payment history, bill amounts, payment amounts).
    Also calculates the average absolute SHAP value for each feature group.

    Parameters
    ----------
    model : CatBoostClassifier
        Trained CatBoost model
    df_plot : pd.DataFrame
        Input features for SHAP analysis

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature groups to their importance scores:
        - demographics: Age, sex, education, marriage
        - payment_history: Payment status features
        - bill_amounts: Bill amount features
        - payment_amounts: Payment amount features

    Example
    -------
    >>> group_importance = plot_shap(model, X_test)
    >>> print("Feature group importance:")
    >>> for group, score in group_importance.items():
    ...     print(f"{group}: {score:.4f}")

    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_plot)

    # Plot overall SHAP summary
    shap.summary_plot(shap_values, df_plot, show=False)
    plt.savefig(FIGURES_DIR / "test_shap_overall.png")
    plt.close()

    # Calculate feature group importance
    feature_groups = {
        "demographics": ["SEX", "EDUCATION", "MARRIAGE", "AGE"],
        "payment_history": [col for col in df_plot.columns if col.startswith("PAY_")],
        "bill_amounts": [col for col in df_plot.columns if col.startswith("BILL_AMT")],
        "payment_amounts": [col for col in df_plot.columns if col.startswith("PAY_AMT")],
    }

    group_importance = {}
    for group_name, features in feature_groups.items():
        feature_indices = [i for i, col in enumerate(df_plot.columns) if col in features]
        if feature_indices:  # Only calculate if group has features
            group_shap = np.abs(shap_values[feature_indices]).mean()
            group_importance[group_name] = float(group_shap)

            # Create group-specific SHAP plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values[:, feature_indices],
                df_plot.iloc[:, feature_indices],
                show=False,
            )
            plt.title(f"SHAP Values - {group_name.replace('_', ' ').title()}")
            plt.savefig(FIGURES_DIR / f"test_shap_{group_name}.png")
            plt.close()

    return group_importance


def explain_prediction(
    model: CatBoostClassifier,
    sample: pd.DataFrame,
    shap_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, any]:
    """Generate detailed explanation for a single prediction.

    Creates an explanation dictionary containing the predicted class,
    probability, and top contributing features with their values and
    SHAP contributions.

    Parameters
    ----------
    model : CatBoostClassifier
        Trained CatBoost model
    sample : pd.DataFrame
        Single sample to explain (one row)
    shap_values : np.ndarray
        Pre-calculated SHAP values for the sample
    feature_names : List[str]
        Names of features in the same order as shap_values

    Returns
    -------
    Dict[str, any]
        Prediction explanation containing:
        - prediction: Binary prediction (0/1)
        - probability: Predicted probability of default
        - top_features: List of top 5 contributing features, each with:
          * feature: Feature name
          * value: Feature value for this sample
          * contribution: SHAP value (contribution to prediction)

    Example
    -------
    >>> explanation = explain_prediction(
    ...     model, customer_data,
    ...     shap_values, feature_names
    ... )
    >>> print(f"Prediction: {explanation['prediction']}")
    >>> print(f"Probability: {explanation['probability']:.3f}")
    >>> for feature in explanation['top_features']:
    ...     print(f"{feature['feature']}: {feature['contribution']:.3f}")

    """
    prob = model.predict_proba(sample)[0, 1]
    pred = int(prob >= 0.5)

    # Get top contributing features
    feature_importance = list(zip(feature_names, shap_values, strict=False))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
    top_features = feature_importance[:5]

    return {
        "prediction": pred,
        "probability": float(prob),
        "top_features": [
            {
                "feature": feat,
                "value": float(sample[feat].iloc[0]),
                "contribution": float(importance),
            }
            for feat, importance in top_features
        ],
    }


def predict(
    model: CatBoostClassifier,
    df_pred: pd.DataFrame,
    params: dict,
    batch_size: int = 1000,
    optimize_thresh: bool = True,
) -> tuple[str, dict[str, any]]:
    """Generate predictions and explanations for credit default data.

    Makes predictions in batches, generates SHAP explanations for each prediction,
    optimizes the probability threshold if requested, and calculates various
    metrics. Results are saved to files for later analysis.

    Parameters
    ----------
    model : CatBoostClassifier
        Trained CatBoost model
    df_pred : pd.DataFrame
        Data to make predictions on
    params : dict
        Model parameters including feature_columns
    batch_size : int, optional
        Number of samples to process at once, by default 1000
    optimize_thresh : bool, optional
        Whether to optimize the probability threshold, by default True

    Returns
    -------
    Tuple[str, Dict[str, any]]
        - Path to saved predictions CSV file
        - Results dictionary containing:
          * predictions: Binary predictions array
          * probabilities: Probability scores array
          * explanations: List of prediction explanations
          * feature_importance: Feature group importance scores
          * threshold: Optimal probability threshold
          * metrics: Performance metrics if true labels available

    Example
    -------
    >>> preds_path, results = predict(
    ...     model=model,
    ...     df_pred=test_data,
    ...     params=model_params,
    ...     optimize_thresh=True
    ... )
    >>> print(f"Predictions saved to: {preds_path}")
    >>> print(f"Default rate: {results['metrics']['default_rate']:.2%}")

    """
    feature_columns = params.pop("feature_columns")
    X_pred = df_pred[feature_columns].copy()

    # Get categorical features from config or use a heuristic approach
    cat_features = [col for col in X_pred.columns if col in categorical]

    # Log categorical features before conversion
    logger.debug(f"Categorical features: {cat_features}")

    # Convert categorical features and any float columns in categorical to string to avoid CatBoost errors
    for col in X_pred.columns:
        # If column is in the categorical list or is a float column, convert to string
        if col in categorical:
            logger.debug(f"Converting column {col} of type {X_pred[col].dtype} to string")
            X_pred[col] = X_pred[col].astype(str)

    # Log column types after conversion
    logger.debug(f"Column dtypes after conversion: {X_pred.dtypes}")

    # Initialize results
    all_probs = []
    all_explanations = []

    # Batch processing
    for start_idx in range(0, len(X_pred), batch_size):
        end_idx = min(start_idx + batch_size, len(X_pred))
        batch = X_pred.iloc[start_idx:end_idx]

        # Get probabilities and SHAP values for batch
        try:
            probs = model.predict_proba(batch)[:, 1]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(batch)

            all_probs.extend(probs)

            # Generate explanations for batch
            for i in range(len(batch)):
                explanation = explain_prediction(
                    model,
                    batch.iloc[[i]],
                    shap_values[i],
                    feature_columns,
                )
                all_explanations.append(explanation)

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(f"Batch data types: {batch.dtypes}")
            logger.error(f"First row of batch: {batch.iloc[0].to_dict()}")
            raise

    all_probs = np.array(all_probs)

    # Optimize threshold if requested and validation data available
    threshold = 0.5
    if optimize_thresh and target in df_pred.columns:
        threshold, _ = optimize_threshold(df_pred[target], all_probs)

    # Make final predictions
    preds = (all_probs >= threshold).astype(int)

    # Calculate and plot feature importance
    group_importance = plot_shap(model, X_pred)

    # Save predictions and explanations
    results = {
        "predictions": preds,
        "probabilities": all_probs,
        "explanations": all_explanations,
        "feature_importance": group_importance,
        "threshold": threshold,
    }

    # Calculate metrics if true values available
    if target in df_pred.columns:
        business_metrics = calculate_business_metrics(df_pred[target], preds)
        pr_auc, _, _ = calculate_pr_auc(df_pred[target], all_probs)
        results["metrics"] = {
            **business_metrics,
            "pr_auc": pr_auc,
        }

    # Save predictions
    df_pred[target] = preds
    df_pred["probability"] = all_probs
    preds_path = MODELS_DIR / "preds.csv"
    df_pred.to_csv(preds_path, index=False)

    # Save explanations
    explanations_path = MODELS_DIR / "explanations.json"
    with open(explanations_path, "w") as f:
        json.dump(results, f, indent=2)

    return preds_path, results


if __name__ == "__main__":
    # Setup MLflow environment
    MLFLOW_DIR = Path(".mlflow")
    MLFLOW_DB_DIR = MLFLOW_DIR / "db"
    MLFLOW_ARTIFACTS_DIR = MLFLOW_DIR / "artifacts"

    # Set MLflow environment variables
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{MLFLOW_DB_DIR}/mlflow.db"
    os.environ["MLFLOW_ARTIFACT_ROOT"] = str(MLFLOW_ARTIFACTS_DIR.absolute())

    # Set tracking URI
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Load test data
    df_test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    logger.info(f"Loaded test data with {len(df_test)} samples")

    # Get model from registry
    logger.debug(
        f"[predict.py] MLFLOW_TRACKING_URI from env: {os.environ.get('MLFLOW_TRACKING_URI')}"
    )
    logger.debug(
        f"[predict.py] mlflow.get_tracking_uri() before client: {mlflow.get_tracking_uri()}"
    )
    client = MlflowClient(mlflow.get_tracking_uri())
    model_info = get_model_by_alias(client, alias="champion")

    if model_info is None:
        logger.info("No champion model found, checking for any models...")
        try:
            registered_models = client.search_registered_models()
            if not registered_models:
                logger.info("No models found in registry. Please train a model first.")
                sys.exit(0)
            model_info = registered_models[0].latest_versions[0]
            logger.info(f"Using latest model version: {model_info.version}")
        except MlflowException as e:
            logger.error(f"Error searching for models: {e}")
            sys.exit(0)

    # Load model and parameters
    run_data_dict = client.get_run(model_info.run_id).data.to_dictionary()
    run = client.get_run(model_info.run_id)
    log_model_meta = json.loads(run.data.tags["mlflow.log-model.history"])

    _, artifact_folder = os.path.split(model_info.source)
    model_uri = f"runs:/{model_info.run_id}/{artifact_folder}"
    logger.info(f"Loading model from {model_uri}")
    loaded_model = mlflow.catboost.load_model(model_uri)

    params = run_data_dict["params"]

    # Check if signature exists in the model metadata
    if "signature" in log_model_meta[0]:
        # Get feature names from signature if available
        params["feature_columns"] = [
            inp["name"] for inp in json.loads(log_model_meta[0]["signature"]["inputs"])
        ]
    else:
        # Fallback to using all columns in the test data except the target
        logger.warning(
            "Model signature not found in metadata. Using all test data columns as features."
        )
        params["feature_columns"] = [col for col in df_test.columns if col != target]

    # Make predictions with explanations
    logger.info("Making predictions with explanations...")
    preds_path, results = predict(
        model=loaded_model,
        df_pred=df_test,
        params=params,
        batch_size=1000,
        optimize_thresh=True,
    )

    # Log results
    logger.info(f"\nPredictions saved to: {preds_path}")
    logger.info("Explanations saved to: " + str(MODELS_DIR / "explanations.json"))

    logger.info("\nFeature Group Importance:")
    for group, importance in results["feature_importance"].items():
        logger.info(f"{group}: {importance:.4f}")

    if "metrics" in results:
        logger.info("\nPrediction Metrics:")
        for metric, value in results["metrics"].items():
            if isinstance(value, int | float):
                logger.info(f"{metric}: {value:.4f}")

    logger.info(f"\nOptimal threshold: {results['threshold']:.4f}")
    logger.info(f"Total predictions: {len(results['predictions'])}")
    default_rate = sum(results["predictions"]) / len(results["predictions"]) * 100
    logger.info(f"Predicted defaults: {sum(results['predictions'])} ({default_rate:.1f}%)")
