"""Functions for preprocessing credit card default prediction data.

This module handles the preprocessing pipeline for credit card default prediction,
including data download, feature engineering, and transformations. It implements
domain-specific preprocessing steps tailored for credit risk assessment.

Functions:
    get_raw_data: Download dataset from Kaggle
    encode_categorical_features: Encode categorical features
    scale_amount_features: Scale amount features using StandardScaler
    engineer_features: Create new features from existing data
    preprocess_df: Run complete preprocessing pipeline


"""

import os
from pathlib import Path
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger
import pandas as pd
from sklearn.preprocessing import StandardScaler

from creditrisk.core.config import DATASET, PROCESSED_DATA_DIR, RAW_DATA_DIR
from creditrisk.core.validation import validate_dataset


def get_raw_data(dataset: str = DATASET) -> None:
    """Download credit card default dataset from Kaggle.

    Uses the Kaggle API to download the UCI ML Credit Card Default Dataset.
    Requires Kaggle API credentials to be properly configured.

    Args:
        dataset: Kaggle dataset identifier (default: from config.py)

    Returns:
        None. Files are downloaded to RAW_DATA_DIR

    Raises:
        KaggleApiError: If authentication fails or dataset not found

    Example:
        >>> get_raw_data("uciml/default-of-credit-card-clients-dataset")

    """
    api = KaggleApi()
    api.authenticate()

    download_folder = Path(RAW_DATA_DIR)
    zip_path = download_folder / "credit_default.zip"

    logger.info(f"RAW_DATA_DIR is: {RAW_DATA_DIR}")
    api.dataset_download_files(dataset, path=str(download_folder))

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(str(download_folder))

    Path.unlink(zip_path)


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for credit default prediction.

    This function performs the following encodings:
    1. Education status (1-4) -> ['graduate', 'university', 'high_school', 'other']
    2. Marriage status (1-3) -> ['married', 'single', 'other']
    3. Payment status (-2 to 8) -> ['no_consumption', 'paid_full', 'revolving', 'delay_Nm']
    4. One-hot encoding of all categorical features

    Args:
        df: Input DataFrame containing categorical features

    Returns:
        DataFrame with encoded categorical features

    Example:
        >>> df = pd.DataFrame({
        ...     "EDUCATION": [1, 2, 3],
        ...     "MARRIAGE": [1, 2, 1],
        ...     "PAY_0": [-1, 0, 2]
        ... })
        >>> encoded_df = encode_categorical_features(df)

    """
    # If DataFrame is empty, return it as is
    if df.empty:
        return df

    # Education status encoding
    education_map = {
        1: "graduate",
        2: "university",
        3: "high_school",
        4: "other",
    }
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].map(education_map)

    # Marriage status encoding
    marriage_map = {
        1: "married",
        2: "single",
        3: "other",
    }
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].map(marriage_map)

    # Payment status encoding (-2: no consumption, -1: paid in full, 0: revolving credit, 1-8: months delay)
    # Define all possible payment columns
    all_payment_cols = [f"PAY_{i}" for i in range(7)]

    # Filter to only include columns that exist in the dataframe
    existing_payment_cols = [col for col in all_payment_cols if col in df.columns]

    for col in existing_payment_cols:
        # Make sure there are no null values
        if df[col].isna().any():
            df[col] = df[col].fillna(0)  # Fill NaN with revolving credit as default

        df[col] = df[col].map(
            {
                -2: "no_consumption",
                -1: "paid_full",
                0: "revolving",
                **{i: f"delay_{i}m" for i in range(1, 9)},
            },
        )

    # One-hot encode all categorical columns that exist
    columns_to_encode = []
    if "EDUCATION" in df.columns:
        columns_to_encode.append("EDUCATION")
    if "MARRIAGE" in df.columns:
        columns_to_encode.append("MARRIAGE")
    columns_to_encode.extend(existing_payment_cols)

    # Only apply get_dummies if there are columns to encode
    if columns_to_encode:
        return pd.get_dummies(df, columns=columns_to_encode)
    return df


def scale_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale amount features using StandardScaler.

    Applies standard scaling (zero mean, unit variance) to all amount-related
    features to ensure they are on the same scale for model training.
    Only affects columns containing 'AMOUNT' in their name.

    Args:
        df: Input DataFrame containing amount features

    Returns:
        DataFrame with scaled amount features

    Example:
        >>> df = pd.DataFrame({
        ...     "BILL_AMOUNT1": [1000, 2000, 3000],
        ...     "PAY_AMT1": [500, 1000, 1500]
        ... })
        >>> scaled_df = scale_amount_features(df)

    """
    # If DataFrame is empty, return it as is
    if df.empty:
        return df

    # Look for columns with either "AMOUNT" or "AMT" in their name
    amount_cols = [col for col in df.columns if ("AMOUNT" in col or "AMT" in col)]

    # Only proceed if we found some amount columns and have more than one row
    # (StandardScaler needs at least 2 samples to calculate variance)
    if amount_cols and len(df) > 1:
        logger.debug(f"Scaling amount columns: {amount_cols}")
        scaler = StandardScaler()
        df[amount_cols] = scaler.fit_transform(df[amount_cols])
    elif amount_cols and len(df) == 1:
        # For single row, just center the data
        logger.debug(f"Only one row found, centering amount columns: {amount_cols}")
        df[amount_cols] = 0.0  # Center data to mean 0
    elif not amount_cols:
        logger.warning("No amount columns found for scaling")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer new features from existing credit card data.

    Creates the following new features:
    1. Utilization ratios:
       - Individual ratios for each bill amount
       - Average utilization across all months
    2. Payment metrics:
       - Total payment amount
       - Average payment amount
       - Payment trend (weighted average favoring recent months)
       - Payment consistency (proportion of months with payments)
    3. Bill metrics:
       - Total bill amount
       - Average bill amount
       - Bill trend (weighted average favoring recent months)

    Args:
        df: Input DataFrame containing base features

    Returns:
        DataFrame with additional engineered features

    Example:
        >>> df = pd.DataFrame({
        ...     "LIMIT_BAL": [20000, 30000],
        ...     "BILL_AMOUNT1": [5000, 10000],
        ...     "PAY_AMT1": [2000, 3000]
        ... })
        >>> enhanced_df = engineer_features(df)

    """
    # If DataFrame is empty or missing required columns, return it as is
    if df.empty:
        return df

    if "LIMIT_BAL" not in df.columns:
        logger.warning("LIMIT_BAL column not found, skipping feature engineering")
        return df

    # Payment history patterns - adapt to actual column names in the dataset
    # Check if we have BILL_AMOUNT or BILL_AMT columns
    if "BILL_AMOUNT1" in df.columns:
        bill_cols = [f"BILL_AMOUNT{i}" for i in range(1, 7)]
    else:
        bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]

    pay_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

    # Log available columns for debugging
    logger.debug(f"Available columns: {df.columns.tolist()}")
    logger.debug(f"Using bill columns: {bill_cols}")
    logger.debug(f"Using payment columns: {pay_cols}")

    # Calculate utilization ratios (bill amount / limit balance)
    for i, col in enumerate(bill_cols, 1):
        if col in df.columns:
            df[f"UTILIZATION_RATIO_{i}"] = df[col] / df["LIMIT_BAL"]
        else:
            logger.warning(f"Column {col} not found, skipping utilization ratio calculation")

    # Calculate average utilization for available columns
    utilization_cols = [
        f"UTILIZATION_RATIO_{i}" for i in range(1, 7) if f"UTILIZATION_RATIO_{i}" in df.columns
    ]
    if utilization_cols:
        df["AVG_UTILIZATION"] = df[utilization_cols].mean(axis=1)
    else:
        logger.warning("No utilization ratio columns available, skipping average calculation")

    # Filter to only include columns that exist in the dataframe
    existing_pay_cols = [col for col in pay_cols if col in df.columns]
    existing_bill_cols = [col for col in bill_cols if col in df.columns]

    # Payment amount trends - only if columns exist
    if existing_pay_cols:
        df["TOTAL_PAYMENT"] = df[existing_pay_cols].sum(axis=1)
        df["AVG_PAYMENT"] = df[existing_pay_cols].mean(axis=1)

        # Check if all required payment columns exist
        if all(
            col in df.columns
            for col in ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
        ):
            df["PAYMENT_TREND"] = (
                df["PAY_AMT1"] * 6
                + df["PAY_AMT2"] * 5
                + df["PAY_AMT3"] * 4
                + df["PAY_AMT4"] * 3
                + df["PAY_AMT5"] * 2
                + df["PAY_AMT6"]
            ) / 21  # Weighted average giving more importance to recent payments
        else:
            logger.warning("Not all PAY_AMT columns found, skipping payment trend calculation")

        # Payment consistency
        df["PAYMENT_CONSISTENCY"] = (df[existing_pay_cols] > 0).sum(axis=1) / len(
            existing_pay_cols,
        )
    else:
        logger.warning("No payment columns found, skipping payment metrics calculations")

    # Bill amount trends - only if columns exist
    if existing_bill_cols:
        df["TOTAL_BILL"] = df[existing_bill_cols].sum(axis=1)
        df["AVG_BILL"] = df[existing_bill_cols].mean(axis=1)

        # Check for column presence before calculating trend
        if all(col in df.columns for col in bill_cols):
            df["BILL_TREND"] = (
                df[bill_cols[0]] * 6
                + df[bill_cols[1]] * 5
                + df[bill_cols[2]] * 4
                + df[bill_cols[3]] * 3
                + df[bill_cols[4]] * 2
                + df[bill_cols[5]]
            ) / 21
        else:
            logger.warning("Not all bill amount columns found, skipping bill trend calculation")
    else:
        logger.warning("No bill amount columns found, skipping bill metrics calculations")

    return df


def preprocess_df(file: str | Path) -> str | Path:
    """Preprocess credit card default dataset.

    Applies the complete preprocessing pipeline:
    1. Data validation and cleaning
    2. Categorical feature encoding
    3. Amount feature scaling
    4. Feature engineering

    The processed dataset is saved to PROCESSED_DATA_DIR with the same
    filename as the input file.

    Args:
        file: Path to the raw data file

    Returns:
        Path to the processed output file

    Example:
        >>> input_path = "data/raw/credit_data.csv"
        >>> output_path = preprocess_df(input_path)
        >>> print(f"Processed data saved to: {output_path}")

    """
    _, file_name = os.path.split(file)
    df_data = pd.read_csv(file)

    # Validate and clean data
    df_data = validate_dataset(df_data)

    # Encode categorical features
    df_data = encode_categorical_features(df_data)

    # Scale amount features
    df_data = scale_amount_features(df_data)

    # Engineer new features
    df_data = engineer_features(df_data)

    # Save processed data
    logger.debug(f"Saving processed data to directory: {PROCESSED_DATA_DIR}")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    outfile_path = PROCESSED_DATA_DIR / file_name
    logger.debug(f"Writing to file: {outfile_path} (absolute: {outfile_path.resolve()})")
    df_data.to_csv(outfile_path, index=False)

    # Verify the file was created
    if outfile_path.exists():
        logger.info(f"Successfully saved processed data to {outfile_path}")
        logger.debug(f"File size: {outfile_path.stat().st_size} bytes")
    else:
        logger.error(f"Failed to save processed data to {outfile_path}")

    return outfile_path


if __name__ == "__main__":
    try:
        # Log environment information
        import sys

        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Python paths: {sys.path}")
        logger.debug(f"Environment PROJ_ROOT: {os.environ.get('PROJ_ROOT')}")
        logger.debug(f"Environment PYTHONPATH: {os.environ.get('PYTHONPATH')}")
        logger.debug(f"Environment DATA_DIR: {os.environ.get('DATA_DIR')}")
        logger.debug(f"Environment PROCESSED_DATA_DIR: {os.environ.get('PROCESSED_DATA_DIR')}")

        # Skip Kaggle download and use existing file
        logger.info("Using existing dataset")
        # Try looking for either UCI_Credit_Card.csv or train.csv
        raw_file_path = RAW_DATA_DIR / "UCI_Credit_Card.csv"
        if not raw_file_path.exists():
            logger.info("UCI_Credit_Card.csv not found, checking for train.csv")
            raw_file_path = RAW_DATA_DIR / "train.csv"

        if not raw_file_path.exists():
            logger.error(f"No suitable data file found in {RAW_DATA_DIR}")
            available_files = list(RAW_DATA_DIR.glob("*.csv"))
            if available_files:
                logger.info(f"Available CSV files in {RAW_DATA_DIR}: {available_files}")
                # Use the first available CSV file as fallback
                raw_file_path = available_files[0]
                logger.info(f"Using {raw_file_path.name} as fallback")
            else:
                logger.error(f"No CSV files found in {RAW_DATA_DIR}")
                sys.exit(1)
        else:
            logger.info(f"Raw data file found at {raw_file_path}")

        # Preprocess the dataset
        logger.info("Preprocessing data")
        output_path = preprocess_df(raw_file_path)
        logger.info(f"Preprocessing completed. Output saved to {output_path}")

        # List processed files for verification
        processed_files = list(PROCESSED_DATA_DIR.glob("*"))
        logger.info(f"Files in processed directory: {processed_files}")

    except (
        OSError,
        FileNotFoundError,
        PermissionError,
        ValueError,
        KeyError,
        pd.errors.EmptyDataError,
    ) as e:
        logger.exception(f"Error during preprocessing: {e}")
        sys.exit(1)
