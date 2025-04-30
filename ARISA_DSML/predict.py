"""Run prediction on test data."""
import json
import os
from pathlib import Path

from catboost import CatBoostClassifier
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow.client import MlflowClient
import pandas as pd
import shap

from ARISA_DSML.config import (
    FIGURES_DIR,
    MODEL_NAME,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    target,
)
from ARISA_DSML.resolve import get_model_by_alias


def plot_shap(model:CatBoostClassifier, df_plot:pd.DataFrame)->None:
    """Plot model shapley overview plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_plot)

    shap.summary_plot(shap_values, df_plot, show=False)
    plt.savefig(FIGURES_DIR / "test_shap_overall.png")


def predict(model:CatBoostClassifier, df_pred:pd.DataFrame, params:dict)->str|Path:
    """Do predictions on test data."""
    feature_columns = params.pop("feature_columns")

    preds = model.predict(df_pred[feature_columns])
    plot_shap(model, df_pred[feature_columns])
    df_pred[target] = preds
    preds_path = MODELS_DIR / "preds.csv"
    df_pred[["PassengerId", target]].to_csv(preds_path, index=False)

    return preds_path


if __name__=="__main__":
    df_test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    client = MlflowClient(mlflow.get_tracking_uri())
    model_info = get_model_by_alias(client, alias="champion")
    
    if model_info is None:
        logger.info("No champion model found, checking for any models...")
        try:
            registered_models = client.search_registered_models()
            if not registered_models:
                logger.info("No models found in registry. Please train a model first.")
                exit(0)
            model_info = registered_models[0].latest_versions[0]
            logger.info(f"Using latest model version: {model_info.version}")
        except MlflowException as e:
            logger.error(f"Error searching for models: {e}")
            exit(0)

    # extract params/metrics data for run `test_run_id` in a single dict
    run_data_dict = client.get_run(model_info.run_id).data.to_dictionary()
    run = client.get_run(model_info.run_id)
    log_model_meta = json.loads(run.data.tags["mlflow.log-model.history"])
    log_model_meta[0]["signature"]

    _, artifact_folder = os.path.split(model_info.source)
    logger.info(artifact_folder)
    model_uri = f"runs:/{model_info.run_id}/{artifact_folder}"
    logger.info(model_uri)
    loaded_model = mlflow.catboost.load_model(model_uri)

    params = run_data_dict["params"]
    params["feature_columns"] = [inp["name"] for inp in json.loads(log_model_meta[0]["signature"]["inputs"])]
    preds_path = predict(loaded_model, df_test, params)


