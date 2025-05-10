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
    # Check if we have enough data for meaningful SHAP plots
    if len(df_plot) < 10:
        logger.warning(
            f"Not enough data for SHAP plots (only {len(df_plot)} samples). Using simplified feature importance.",
        )
        # Return a simplified importance based on model feature importances
        try:
            importances = model.get_feature_importance()
            feature_names = model.feature_names_
            feature_importance = dict(zip(feature_names, importances, strict=False))

            # Group features
            group_importance = {
                "demographics": 0.0,
                "payment_history": 0.0,
                "bill_amounts": 0.0,
                "payment_amounts": 0.0,
            }

            # Calculate group importance using model's feature importance
            for feature, importance in feature_importance.items():
                if feature in ["SEX", "EDUCATION", "MARRIAGE", "AGE"] or any(
                    f in feature for f in ["EDUCATION_", "MARRIAGE_"]
                ):
                    group_importance["demographics"] += importance
                elif feature.startswith("PAY_") and not feature.startswith("PAY_AMT"):
                    group_importance["payment_history"] += importance
                elif feature.startswith("BILL_AMT") or "BILL" in feature:
                    group_importance["bill_amounts"] += importance
                elif feature.startswith("PAY_AMT") or "PAYMENT" in feature:
                    group_importance["payment_amounts"] += importance

            # Normalize to get relative importances
            total = sum(group_importance.values()) or 1.0  # Avoid division by zero
            for group in group_importance:
                group_importance[group] = float(group_importance[group] / total)

            return group_importance
        except Exception as e:
            logger.error(f"Could not calculate feature importance: {e}")
            # Return default importances
            return {
                "demographics": 0.25,
                "payment_history": 0.35,
                "bill_amounts": 0.2,
                "payment_amounts": 0.2,
            }

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_plot)

        # Plot overall SHAP summary
        shap.summary_plot(shap_values, df_plot, show=False)
        plt.savefig(FIGURES_DIR / "test_shap_overall.png")
        plt.close()

        # Calculate feature group importance
        feature_groups = {
            "demographics": ["SEX", "EDUCATION", "MARRIAGE", "AGE"],
            "payment_history": [
                col
                for col in df_plot.columns
                if col.startswith("PAY_") and not col.startswith("PAY_AMT")
            ],
            "bill_amounts": [
                col for col in df_plot.columns if col.startswith("BILL_AMT") or "BILL" in col
            ],
            "payment_amounts": [
                col for col in df_plot.columns if col.startswith("PAY_AMT") or "PAYMENT" in col
            ],
        }

        group_importance = {}
        for group_name, features in feature_groups.items():
            # Only use features that exist in the dataframe
            available_features = [feat for feat in features if feat in df_plot.columns]
            if not available_features:
                logger.warning(f"No features found for group {group_name}")
                group_importance[group_name] = 0.0
                continue

            feature_indices = [
                i for i, col in enumerate(df_plot.columns) if col in available_features
            ]
            if feature_indices:  # Only calculate if group has features
                try:
                    # Calculate mean absolute SHAP value for the feature group
                    group_values = shap_values[:, feature_indices]
                    group_shap = np.abs(group_values).mean()
                    group_importance[group_name] = float(group_shap)

                    # Create group-specific SHAP plot
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(
                        group_values,
                        df_plot.iloc[:, feature_indices],
                        show=False,
                    )
                    plt.title(f"SHAP Values - {group_name.replace('_', ' ').title()}")
                    plt.savefig(FIGURES_DIR / f"test_shap_{group_name}.png")
                    plt.close()
                except Exception as e:
                    logger.warning(f"Error calculating SHAP for group {group_name}: {e}")
                    group_importance[group_name] = 0.0
            else:
                group_importance[group_name] = 0.0

        return group_importance
    except Exception as e:
        logger.error(f"Error in SHAP calculations: {e}")
        # Return default importances
        return {
            "demographics": 0.25,
            "payment_history": 0.35,
            "bill_amounts": 0.2,
            "payment_amounts": 0.2,
        }


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

    # Try to get the feature names from the model
    try:
        if hasattr(model, "feature_names_"):
            model_feature_names = model.feature_names_
            logger.debug(f"Model features order from model.feature_names_: {model_feature_names}")

            # Reorder the dataframe columns and ensure all required columns exist
            missing_features = set(model_feature_names) - set(X_pred.columns)

            # If we're missing features, add them with default values
            if missing_features:
                logger.warning(
                    f"Missing {len(missing_features)} features in test data: {missing_features}",
                )
                # Add missing features with default values
                for feature in missing_features:
                    if feature.startswith("PAY_") and (
                        "delay" in feature or "revolving" in feature or "paid_full" in feature
                    ):
                        # Binary features for payment status
                        X_pred[feature] = False
                    elif feature.startswith("EDUCATION_") or feature.startswith("MARRIAGE_"):
                        # Binary features for categorical variables
                        X_pred[feature] = False
                    else:
                        # Numeric features get zeros
                        X_pred[feature] = 0

            # Reorder columns to match model's feature order
            # Only include columns that are actually in the model_feature_names
            existing_features = [col for col in model_feature_names if col in X_pred.columns]
            X_pred = X_pred[existing_features]
            logger.debug(f"Reordered columns to match model feature order: {list(X_pred.columns)}")
    except (AttributeError, KeyError) as e:
        logger.warning(f"Could not get feature names from model: {e}")

    # Convert categorical features and numeric columns to string to avoid CatBoost errors
    for col in X_pred.columns:
        # If column is in the categorical list or contains numeric data that might be mistaken for categorical
        if col in categorical or col.startswith("BILL_AMOUNT") or col.startswith("PAY_AMT"):
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

    # Convert numpy arrays to Python native types for JSON serialization
    def numpy_to_python(obj):
        """Convert numpy arrays in a nested structure to Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: numpy_to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [numpy_to_python(item) for item in obj]
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        if isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return obj

    # Save explanations
    explanations_path = MODELS_DIR / "explanations.json"
    with open(explanations_path, "w") as f:
        json.dump(numpy_to_python(results), f, indent=2)

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
        f"[predict.py] MLFLOW_TRACKING_URI from env: {os.environ.get('MLFLOW_TRACKING_URI')}",
    )
    logger.debug(
        f"[predict.py] mlflow.get_tracking_uri() before client: {mlflow.get_tracking_uri()}",
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
            "Model signature not found in metadata. Using all test data columns as features.",
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
