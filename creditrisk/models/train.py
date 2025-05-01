"""Functions for training credit default prediction models.

This module implements the training pipeline for credit card default prediction,
including hyperparameter optimization, cross-validation, and model training with
MLflow tracking. It uses CatBoost for gradient boosting and optimizes for
business-specific metrics.

Functions:
    run_hyperopt: Run Optuna hyperparameter optimization
    train_cv: Perform cross-validated training
    train: Train final model on full dataset
    plot_error_scatter: Plot performance metrics with error bands
    get_or_create_experiment: Get or create MLflow experiment


"""

from pathlib import Path

from catboost import CatBoostClassifier, Pool, cv
import joblib
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow.client import MlflowClient
import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import shap
from sklearn.model_selection import train_test_split

from creditrisk.core.config import (
    FIGURES_DIR,
    MODEL_NAME,
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
from creditrisk.utils.helpers import get_git_commit_hash

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///.mlflow/db/mlflow.db")

# comment to trigger workflow ver4


def run_hyperopt(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    categorical_indices: list[int],
    test_size: float = 0.25,
    n_trials: int = 20,
    overwrite: bool = False,
) -> str | Path:
    """Run Optuna hyperparameter optimization for CatBoost model.

    Performs hyperparameter optimization using Optuna, optimizing for minimum
    business cost. Uses MLflow to track trials and metrics. Parameters are
    saved for later use in training.

    Args:
        X_train: Training features
        y_train: Training labels
        categorical_indices: List of indices for categorical features
        test_size: Fraction of data to use for validation (default: 0.25)
        n_trials: Number of optimization trials (default: 20)
        overwrite: Whether to overwrite existing best parameters (default: False)

    Returns:
        Path to saved best parameters

    Example:
        >>> best_params_path = run_hyperopt(
        ...     X_train, y_train,
        ...     categorical_indices=[0, 1, 2],
        ...     n_trials=50
        ... )
        >>> print(f"Best parameters saved to: {best_params_path}")

    """
    best_params_path = MODELS_DIR / "best_params.pkl"
    if not best_params_path.is_file() or overwrite:
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            X_train,
            y_train,
            test_size=test_size,
            random_state=42,
        )

        def objective(trial: optuna.trial.Trial) -> float:
            with mlflow.start_run(nested=True):
                # Calculate class weights based on data distribution
                class_counts = y_train_opt.value_counts()
                total = len(y_train_opt)
                class_weights = {
                    0: total / (2 * class_counts[0]),
                    1: total / (2 * class_counts[1]),
                }

                params = {
                    "depth": trial.suggest_int("depth", 4, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1),
                    "iterations": trial.suggest_int("iterations", 100, 1000),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
                    "bagging_temperature": trial.suggest_float("bagging_temperature", 0.1, 2),
                    "random_strength": trial.suggest_float(
                        "random_strength",
                        1e-3,
                        10.0,
                        log=True,
                    ),
                    "class_weights": class_weights,  # Using custom class weights based on data distribution
                    "ignored_features": [0],
                }
                model = CatBoostClassifier(**params, verbose=0)
                model.fit(
                    X_train_opt,
                    y_train_opt,
                    eval_set=(X_val_opt, y_val_opt),
                    cat_features=categorical_indices,
                    early_stopping_rounds=50,
                )
                mlflow.log_params(params)
                preds = model.predict(X_val_opt)
                probs = model.predict_proba(X_val_opt)[:, 1]

                # Calculate various metrics
                business_metrics = calculate_business_metrics(y_val_opt, preds)
                pr_auc, _, _ = calculate_pr_auc(y_val_opt, probs)

                # Log all metrics
                mlflow.log_metrics(
                    {
                        "f1": business_metrics["f1_score"],
                        "precision": business_metrics["precision"],
                        "recall": business_metrics["recall"],
                        "pr_auc": pr_auc,
                        "approval_rate": business_metrics["approval_rate"],
                        "avg_cost_per_decision": business_metrics["avg_cost_per_decision"],
                    },
                )

                # Optimize threshold based on business costs
                best_threshold, best_metrics = optimize_threshold(y_val_opt, probs)
                mlflow.log_metric("optimal_threshold", best_threshold)
                mlflow.log_metric("optimal_cost", best_metrics["total_cost"])

            return best_metrics["total_cost"]  # Optimize for minimum business cost

        study = optuna.create_study(direction="minimize", study_name="credit_default_optimization")
        study.optimize(objective, n_trials=n_trials)

        joblib.dump(study.best_params, best_params_path)

        params = study.best_params
    else:
        params = joblib.load(best_params_path)
    logger.info("Best Parameters: " + str(params))
    return best_params_path


def train_cv(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    categorical_indices: list[int],
    params: dict,
    eval_metric: str = "F1",
    n: int = 5,
    stratify: bool = True,
) -> str | Path:
    """Perform cross-validated training of CatBoost model.

    Runs n-fold cross-validation to estimate model performance and stability.
    Generates learning curves and saves results for analysis.

    Args:
        X_train: Training features
        y_train: Training labels
        categorical_indices: List of indices for categorical features
        params: Model parameters (from hyperparameter optimization)
        eval_metric: Metric to evaluate during CV (default: "F1")
        n: Number of CV folds (default: 5)
        stratify: Whether to use stratified folds (default: True)

    Returns:
        Path to saved cross-validation results

    Example:
        >>> cv_path = train_cv(
        ...     X_train, y_train,
        ...     categorical_indices=[0, 1, 2],
        ...     params=best_params,
        ...     n=10
        ... )
        >>> cv_results = pd.read_csv(cv_path)

    """
    params["eval_metric"] = eval_metric
    params["loss_function"] = "Logloss"
    params["ignored_features"] = [0]  # ignore passengerid

    data = Pool(X_train, y_train, cat_features=categorical_indices)

    cv_results = cv(
        params=params,
        pool=data,
        fold_count=n,
        partition_random_seed=42,
        shuffle=True,
        plot=True,
        stratified=stratify,
    )

    cv_output_path = MODELS_DIR / "cv_results.csv"
    cv_results.to_csv(cv_output_path, index=False)

    return cv_output_path


def train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    categorical_indices: list[int],
    params: dict | None,
    artifact_name: str = "catboost_model_credit_default",
    cv_results=None,
) -> tuple[str | Path]:
    """Train final CatBoost model on full dataset.

    Trains the model using optimized parameters, logs metrics and artifacts to MLflow,
    and saves the model to the model registry. Also generates and logs feature
    importance plots using SHAP values.

    Args:
        X_train: Training features
        y_train: Training labels
        categorical_indices: List of indices for categorical features
        params: Model parameters (from hyperparameter optimization)
        artifact_name: Name for saved model file (default: "catboost_model_credit_default")
        cv_results: Cross-validation results for logging (optional)

    Returns:
        Tuple of (model_path, model_params_path)

    Example:
        >>> model_path, params_path = train(
        ...     X_train, y_train,
        ...     categorical_indices=[0, 1, 2],
        ...     params=best_params,
        ...     cv_results=cv_results
        ... )

    """
    if params is None:
        logger.info("Training model without tuned hyperparameters")
        params = {}
    with mlflow.start_run():
        params["ignored_features"] = [0]

        model = CatBoostClassifier(
            **params,
            verbose=True,
        )

        model.fit(
            X_train,
            y_train,
            verbose_eval=50,
            early_stopping_rounds=50,
            cat_features=categorical_indices,
            use_best_model=False,
            plot=True,
        )
        params["feature_columns"] = X_train.columns
        mlflow.log_params(params)

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODELS_DIR / f"{artifact_name}.cbm"
        model.save_model(model_path)
        mlflow.log_artifact(model_path)

        # Log cross-validation metrics
        cv_metric_mean = cv_results["test-F1-mean"].mean()
        mlflow.log_metric("f1_cv_mean", cv_metric_mean)

        # Calculate and log SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # Group features by domain and calculate importance
        feature_groups = {
            "demographics": ["SEX", "EDUCATION", "MARRIAGE", "AGE"],
            "payment_history": [col for col in X_train.columns if col.startswith("PAY_")],
            "bill_amounts": [col for col in X_train.columns if col.startswith("BILL_AMT")],
            "payment_amounts": [col for col in X_train.columns if col.startswith("PAY_AMT")],
        }

        group_importance = {}
        for group_name, features in feature_groups.items():
            feature_indices = [i for i, col in enumerate(X_train.columns) if col in features]
            group_shap = np.abs(shap_values[feature_indices]).mean()
            group_importance[group_name] = float(group_shap)
            mlflow.log_metric(f"shap_importance_{group_name}", group_shap)

        # Generate and log SHAP summary plot
        shap.summary_plot(
            shap_values,
            X_train,
            feature_names=X_train.columns,
            show=False,
            plot_size=(10, 6),
        )
        mlflow.log_figure(plt.gcf(), "shap_summary.png")
        plt.close()

        # Log the model
        model_info = mlflow.catboost.log_model(
            cb_model=model,
            artifact_path="model",
            input_example=X_train,
            registered_model_name=MODEL_NAME,
        )
        client = MlflowClient(mlflow.get_tracking_uri())
        model_info = client.get_latest_versions(MODEL_NAME)[0]
        client.set_registered_model_alias(MODEL_NAME, "challenger", model_info.version)
        client.set_model_version_tag(
            name=model_info.name,
            version=model_info.version,
            key="git_sha",
            value=get_git_commit_hash(),
        )
        model_params_path = MODELS_DIR / "model_params.pkl"
        joblib.dump(params, model_params_path)
        fig1 = plot_error_scatter(
            df_plot=cv_results,
            name="Mean F1 Score",
            title="Cross-Validation (N=5) Mean F1 score with Error Bands",
            xtitle="Training Steps",
            ytitle="Performance Score",
            yaxis_range=[0.5, 1.0],
        )
        mlflow.log_figure(fig1, "test-F1-mean_vs_iterations.png")
        fig2 = plot_error_scatter(
            cv_results,
            x="iterations",
            y="test-Logloss-mean",
            err="test-Logloss-std",
            name="Mean logloss",
            title="Cross-Validation (N=5) Mean Logloss with Error Bands",
            xtitle="Training Steps",
            ytitle="Logloss",
        )
        mlflow.log_figure(fig2, "test-logloss-mean_vs_iterations.png")

    return (model_path, model_params_path)


def plot_error_scatter(  # noqa: PLR0913
    df_plot: pd.DataFrame,
    x: str = "iterations",
    y: str = "test-F1-mean",
    err: str = "test-F1-std",
    name: str = "",
    title: str = "",
    xtitle: str = "",
    ytitle: str = "",
    yaxis_range: list[float] | None = None,
) -> None:
    """Plot performance metrics with error bands using plotly.

    Creates an interactive scatter plot with shaded error regions, useful for
    visualizing cross-validation results and learning curves.

    Args:
        df_plot: DataFrame containing metrics
        x: Column name for x-axis (default: "iterations")
        y: Column name for y-axis (default: "test-F1-mean")
        err: Column name for error values (default: "test-F1-std")
        name: Legend name for the metric (default: "")
        title: Plot title (default: "")
        xtitle: X-axis label (default: "")
        ytitle: Y-axis label (default: "")
        yaxis_range: Y-axis range limits [min, max] (optional)

    Returns:
        None. Displays plot and saves to file.

    Example:
        >>> plot_error_scatter(
        ...     cv_results,
        ...     y="test-F1-mean",
        ...     err="test-F1-std",
        ...     title="Cross-Validation F1 Score",
        ...     yaxis_range=[0.5, 1.0]
        ... )

    """
    # Create figure
    fig = go.Figure()

    if not len(name):
        name = y

    # Add mean performance line
    fig.add_trace(
        go.Scatter(
            x=df_plot[x],
            y=df_plot[y],
            mode="lines",
            name=name,
            line={"color": "blue"},
        ),
    )

    # Add shaded error region
    fig.add_trace(
        go.Scatter(
            x=pd.concat([df_plot[y], df_plot[x][::-1]]),
            y=pd.concat([df_plot[y] + df_plot[err], df_plot[y] - df_plot[err]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line={"color": "rgba(255, 255, 255, 0)"},
            showlegend=False,
        ),
    )

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=xtitle,
        yaxis_title=ytitle,
        template="plotly_white",
    )

    if yaxis_range is not None:
        fig.update_layout(
            yaxis={"range": yaxis_range},
        )

    fig.show()
    fig.write_image(FIGURES_DIR / f"{y}_vs_{x}.png")
    return fig


def get_or_create_experiment(experiment_name: str) -> str:
    """Get or create an MLflow experiment.

    Retrieves an existing MLflow experiment by name or creates a new one if it
    doesn't exist. This ensures consistent experiment tracking across multiple
    training runs.

    Args:
        experiment_name: Name of the MLflow experiment

    Returns:
        Experiment ID (string)

    Example:
        >>> experiment_id = get_or_create_experiment("credit_default_training")
        >>> mlflow.set_experiment(experiment_id=experiment_id)

    """
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id

    return mlflow.create_experiment(experiment_name)


# def champion_callback(study, frozen_trial):
#     """
#     Logging callback that will report when a new trial iteration improves upon existing
#     best trial values.

#     Note: This callback is not intended for use in distributed computing systems such as Spark
#     or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
#     workers or agents.
#     The race conditions with file system state management for distributed trials will render
#     inconsistent values with this callback.
#     """

#     winner = study.user_attrs.get("winner", None)

#     if study.best_value and winner != study.best_value:
#         study.set_user_attr("winner", study.best_value)
#         if winner:
#             improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
#             print(
#                 f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
#                 f"{improvement_percent: .4f}% improvement"
#             )
#         else:
#             print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")


if __name__ == "__main__":
    df_train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")

    y_train = df_train.pop(target)
    X_train = df_train

    categorical_indices = [
        X_train.columns.get_loc(col) for col in categorical if col in X_train.columns
    ]
    experiment_id = get_or_create_experiment("credit_default_hyperparam_tuning")
    mlflow.set_experiment(experiment_id=experiment_id)
    best_params_path = run_hyperopt(X_train, y_train, categorical_indices)
    params = joblib.load(best_params_path)
    cv_output_path = train_cv(X_train, y_train, categorical_indices, params)
    cv_results = pd.read_csv(cv_output_path)

    experiment_id = get_or_create_experiment("credit_default_full_training")
    mlflow.set_experiment(experiment_id=experiment_id)
    model_path, model_params_path = train(
        X_train,
        y_train,
        categorical_indices,
        params,
        cv_results=cv_results,
    )
