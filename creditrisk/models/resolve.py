import sys

from loguru import logger
import mlflow
from mlflow.client import MlflowClient
from mlflow.exceptions import MlflowException

from creditrisk.core.config import MODEL_NAME


def get_model_by_alias(client, model_name: str = MODEL_NAME, alias: str = "champion"):
    """Get model version by alias, handling the case where model doesn't exist."""
    try:
        # First check if the registered model exists
        try:
            client.get_registered_model(model_name)
        except MlflowException as e:
            if "RESOURCE_DOES_NOT_EXIST" in str(e):
                logger.info(f"Model {model_name} not found in registry")
                return None
            raise

        # If model exists, try to get the aliased version
        try:
            return client.get_model_version_by_alias(model_name, alias)
        except MlflowException as e:
            if f"alias {alias} not found" in str(e):
                logger.info(f"No model found with alias {alias}")
                return None
            raise
    except Exception as e:
        logger.error(f"Unexpected error: {e!s}")
        return None


if __name__ == "__main__":
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
