"""Data validation functions for credit card default prediction.

This module provides functions for validating and cleaning credit card data before model training
or prediction. It implements domain-specific validation rules and data quality checks.

Functions:
    validate_value_ranges: Validate and clean value ranges for credit card features
    handle_missing_values: Handle missing values using appropriate strategies
    detect_outliers: Detect and handle outliers using the IQR method
    validate_dataset: Run all validation checks on a dataset


"""

from loguru import logger
import numpy as np
import pandas as pd


def validate_value_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean value ranges for credit card data.

    This function checks and corrects values in credit card data to ensure they fall within
    valid ranges. Invalid values are replaced with appropriate defaults.

    Value Ranges:
        - Payment status (PAY_0 to PAY_6): -2 to 8
          * -2: No consumption
          * -1: Paid in full
          *  0: Revolving credit
          * 1-8: Months delay
        - Amount columns: Non-negative
        - Education: 1-4 (1: graduate, 2: university, 3: high school, 4: other)
        - Marriage: 1-3 (1: married, 2: single, 3: other)
        - Sex: 1-2 (1: male, 2: female)

    Args:
        df: Input DataFrame containing credit card data

    Returns:
        DataFrame with validated and cleaned values

    Example:
        >>> df = pd.DataFrame({
        ...     "PAY_0": [-2, 9, 1],  # 9 is invalid
        ...     "BILL_AMOUNT1": [1000, -500, 2000]  # -500 is invalid
        ... })
        >>> cleaned_df = validate_value_ranges(df)

    """
    # Payment status validation
    payment_cols = [f"PAY_{i}" for i in range(7)]
    for col in payment_cols:
        if col in df.columns:
            invalid_mask = ~df[col].between(-2, 8)
            if invalid_mask.any():
                logger.warning(f"Found {invalid_mask.sum()} invalid values in {col}")
                df.loc[invalid_mask, col] = df[col].median()

    # Amount columns validation (should be non-negative)
    amount_cols = [col for col in df.columns if "AMOUNT" in col]
    for col in amount_cols:
        invalid_mask = df[col] < 0
        if invalid_mask.any():
            logger.warning(f"Found {invalid_mask.sum()} negative values in {col}")
            df.loc[invalid_mask, col] = 0

    # Categorical variables validation
    if "EDUCATION" in df.columns:
        df.loc[~df["EDUCATION"].between(1, 4), "EDUCATION"] = 4  # Other education

    if "MARRIAGE" in df.columns:
        df.loc[~df["MARRIAGE"].between(1, 3), "MARRIAGE"] = 3  # Other status

    if "SEX" in df.columns:
        df.loc[~df["SEX"].between(1, 2), "SEX"] = df["SEX"].mode()[0]

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the credit card dataset.

    This function implements the following strategy for handling missing values:
    - Numeric columns: Fill with median values
    - Categorical columns: Fill with mode (most frequent value)

    Args:
        df: Input DataFrame that may contain missing values

    Returns:
        DataFrame with all missing values handled

    Example:
        >>> df = pd.DataFrame({
        ...     "LIMIT_BAL": [20000, np.nan, 30000],
        ...     "EDUCATION": [1, 2, np.nan]
        ... })
        >>> filled_df = handle_missing_values(df)

    """
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Found missing values:\n{missing[missing > 0]}")

    # Handle missing values based on column type
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    # Fill numeric columns with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical columns with mode
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    return df


def detect_outliers(df: pd.DataFrame, threshold: float = 3) -> pd.DataFrame:
    """Detect and handle outliers in amount columns using the IQR method.

    This function identifies outliers in amount-related columns using the Interquartile Range
    (IQR) method and caps them at the specified threshold. Only applies to columns containing
    'AMOUNT' in their name.

    Args:
        df: Input DataFrame containing amount columns
        threshold: Number of IQRs beyond which values are considered outliers (default: 3)

    Returns:
        DataFrame with outliers capped at threshold boundaries

    Example:
        >>> df = pd.DataFrame({
        ...     "BILL_AMOUNT1": [1000, 2000, 100000],  # 100000 might be outlier
        ...     "PAY_AMT1": [500, 1000, 50000]  # 50000 might be outlier
        ... })
        >>> cleaned_df = detect_outliers(df, threshold=3)

    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    amount_cols = [col for col in numeric_cols if "AMOUNT" in col]

    for col in amount_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers = ~df[col].between(lower_bound, upper_bound)
        if outliers.any():
            logger.warning(f"Found {outliers.sum()} outliers in {col}")
            # Cap outliers at bounds instead of removing
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound

    return df


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Run all validation checks on the credit card dataset.

    This function applies a complete validation pipeline including:
    1. Missing value handling
    2. Value range validation
    3. Outlier detection and handling

    Args:
        df: Input DataFrame to validate

    Returns:
        DataFrame with all validation checks applied

    Example:
        >>> df = pd.read_csv("raw_credit_data.csv")
        >>> validated_df = validate_dataset(df)

    """
    df = handle_missing_values(df)
    df = validate_value_ranges(df)
    return detect_outliers(df)
