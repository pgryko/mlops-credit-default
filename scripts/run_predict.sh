#!/bin/bash

# Check if virtual environment exists
if [ -d ".venv" ]; then
    # Activate virtual environment and run the prediction
    source .venv/bin/activate

    # Create a small test dataset for predictions with correct columns
    python - << EOF
import os
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import json
import sys
from loguru import logger

# Setup directory paths
PROCESSED_DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

# Set MLflow tracking URI
mlflow_db_dir = Path(".mlflow/db")
mlflow_tracking_uri = f"sqlite:///{mlflow_db_dir}/mlflow.db"
os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
mlflow.set_tracking_uri(mlflow_tracking_uri)

try:
    # Find the latest model's run ID
    client = MlflowClient()
    try:
        models = client.search_registered_models()
        if not models:
            print("No registered models found. Please train a model first.")
            sys.exit(1)

        # Get the latest model version
        latest_model = None
        for model in models:
            for version in model.latest_versions:
                if version.current_stage in ['Production', 'Staging'] or version.current_stage == 'None':
                    latest_model = version
                    break
            if latest_model:
                break

        if not latest_model:
            print("No suitable model version found.")
            sys.exit(1)

        run_id = latest_model.run_id
        print(f"Found model with run_id: {run_id}")

        # Get the run data to extract feature names
        run = client.get_run(run_id)
        log_model_meta = json.loads(run.data.tags.get("mlflow.log-model.history", "[]"))

        if log_model_meta and "signature" in log_model_meta[0]:
            # Get feature names from model signature
            feature_names = [inp["name"] for inp in json.loads(log_model_meta[0]["signature"]["inputs"])]
            print(f"Found {len(feature_names)} features in model signature")

            # Check if test.csv exists and has the right format
            test_path = PROCESSED_DATA_DIR / "test.csv"
            if not test_path.exists() or os.path.getsize(test_path) < 100:
                print("Creating a minimal valid test file...")

                # Create a minimal dataframe with the right columns
                data = {name: [0] for name in feature_names}
                # Add categorical columns as strings
                for col in ["SEX", "EDUCATION", "MARRIAGE"] + [f"PAY_{i}" for i in range(7)]:
                    if col in data:
                        data[col] = ["0"]

                # Add the target column if it's not in features
                if "default.payment.next.month" not in data:
                    data["default.payment.next.month"] = [0]

                df = pd.DataFrame(data)
                df.to_csv(test_path, index=False)
                print(f"Created test file at {test_path} with columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error accessing MLflow: {e}")
        sys.exit(1)
except Exception as e:
    print(f"Setup error: {e}")
    sys.exit(1)
EOF

    # Now run the prediction script
    python -m creditrisk.models.predict
else
    echo "Virtual environment not found. Please run 'make requirements' first."
    exit 1
fi