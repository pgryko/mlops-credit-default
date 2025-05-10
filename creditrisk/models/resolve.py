import os
from pathlib import Path
import sys

from loguru import logger
import mlflow
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException

from creditrisk.core.config import MODEL_NAME


def get_model_by_alias(client, model_name: str = MODEL_NAME, alias: str = "champion"):
    """Get model version by alias, handling the case where model doesn't exist."""
    logger.debug(
        f"[resolve.py/get_model_by_alias] MLFLOW_TRACKING_URI from env: {os.environ.get('MLFLOW_TRACKING_URI')}"
    )
    logger.debug(
        f"[resolve.py/get_model_by_alias] mlflow.get_tracking_uri() from client: {client.tracking_uri}"
    )
    logger.debug(
        f"[resolve.py/get_model_by_alias] Effective client tracking_uri: {client._tracking_client.tracking_uri}"
    )
    try:
        # First check if the registered model exists
        try:
            client.get_registered_model(model_name)
        except MlflowException as e:
            # Check for the specific error code for resource not found
            if hasattr(e, "error_code") and e.error_code == "RESOURCE_DOES_NOT_EXIST":
                logger.info(
                    f"Model {model_name} not found in registry (error_code: RESOURCE_DOES_NOT_EXIST)"
                )
                return None
            # Fallback for older MLflow versions or different exception structures, check string content
            if (
                "RESOURCE_DOES_NOT_EXIST" in str(e)
                or "Could not find a registered model with name" in str(e)
                or f"Registered Model with name={model_name} not found" in str(e)
            ):
                logger.info(
                    f"Model {model_name} not found in registry (string match fallback: {e!s})"
                )
                return None
            logger.error(
                f"An MlflowException occurred while trying to get registered model '{model_name}': {e}"
            )
            raise  # Re-raise if it's not a recognized "not found" exception

        # If model exists, try to get the aliased version
        try:
            return client.get_model_version_by_alias(model_name, alias)
        except MlflowException as e:
            if (
                hasattr(e, "error_code") and e.error_code == "RESOURCE_DOES_NOT_EXIST"
            ):  # Also check for alias not found
                logger.info(
                    f"Alias '{alias}' not found for model '{model_name}' (error_code: RESOURCE_DOES_NOT_EXIST)"
                )
                return None
            # Fallback for older MLflow versions or different exception structures for alias not found
            if (
                f"No version of model '{model_name}' with alias '{alias}' found" in str(e)
                or f"alias '{alias}' not found" in str(e)
                or f"Tag for name={model_name}, alias={alias} not found" in str(e)
                or "Registered model alias" in str(e)
            ):
                logger.info(
                    f"Alias '{alias}' not found for model '{model_name}' (string match fallback: {e!s})"
                )
                return None
            logger.error(
                f"An MlflowException occurred while trying to get model version by alias '{alias}' for model '{model_name}': {e}"
            )
            raise  # Re-raise if it's not a recognized "not found" exception
    except Exception as e:
        logger.error(
            f"Unexpected error in get_model_by_alias for {model_name} alias {alias}: {e!s}"
        )
        return None


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

    logger.debug(
        f"[resolve.py/__main__] MLFLOW_TRACKING_URI from env: {os.environ.get('MLFLOW_TRACKING_URI')}"
    )
    logger.debug(
        f"[resolve.py/__main__] mlflow.get_tracking_uri() before client: {mlflow.get_tracking_uri()}"
    )
    client = MlflowClient(mlflow.get_tracking_uri())
    champ_mv = get_model_by_alias(client)

    # Exit gracefully if no models exist
    if champ_mv is None:
        logger.info("No champion model found. Exiting gracefully.")
        sys.exit(0)

    chall_mv = get_model_by_alias(client, alias="challenger")

    if chall_mv:
        champ_run = client.get_run(champ_mv.run_id)
        f1_champ = champ_run.data.metrics["f1_cv_mean"]

        chall_run = client.get_run(chall_mv.run_id)
        f1_chall = chall_run.data.metrics["f1_cv_mean"]

        if f1_chall >= f1_champ:
            logger.info(
                "Challenger model surpassed metric of current champion, promoting challenger to champion.",
            )
            client.delete_registered_model_alias(MODEL_NAME, "challenger")
            client.set_registered_model_alias(MODEL_NAME, "champion", chall_mv.version)
        else:
            logger.info("Challenger model does not surpass champion. Keeping current champion.")
    else:
        logger.info("No challenger model found. Keeping current champion.")
