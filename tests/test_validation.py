"""Unit tests for validation module."""

import numpy as np
import pandas as pd
import pytest

from creditrisk.core.validation import (
    detect_outliers,
    handle_missing_values,
    validate_dataset,
    validate_value_ranges,
)


@pytest.fixture
def sample_credit_data():
    """Create sample credit card data for testing."""
    return pd.DataFrame(
        {
            "PAY_0": [-2, 0, 2, 9],  # One invalid value (9)
            "PAY_2": [0, 2, -2, 3],  # All valid
            "BILL_AMOUNT1": [1000, -500, 2000, 3000],  # One negative value
            "EDUCATION": [1, 2, 5, 3],  # One invalid value (5)
            "MARRIAGE": [1, 2, 4, 2],  # One invalid value (4)
            "SEX": [1, 2, 3, 1],  # One invalid value (3)
        },
    )


@pytest.fixture
def sample_missing_data():
    """Create sample data with missing values."""
    return pd.DataFrame(
        {
            "PAY_0": [1, np.nan, 2, 0],
            "BILL_AMOUNT1": [1000, 2000, np.nan, 3000],
            "EDUCATION": [1, 2, np.nan, 3],
        },
    )


def test_validate_value_ranges(sample_credit_data) -> None:
    """Test value range validation and cleaning."""
    df = validate_value_ranges(sample_credit_data.copy())

    # Test payment status correction
    assert df["PAY_0"].max() <= 8

    # Test amount columns are non-negative
    assert (df["BILL_AMOUNT1"] >= 0).all()

    # Test categorical variables
    assert df["EDUCATION"].max() <= 4
    assert df["MARRIAGE"].max() <= 3
    assert df["SEX"].max() <= 2


def test_handle_missing_values(sample_missing_data) -> None:
    """Test missing value handling."""
    df = handle_missing_values(sample_missing_data.copy())

    # Verify no missing values remain
    assert not df.isnull().any().any()

    # Verify numeric columns filled with median
    assert df["BILL_AMOUNT1"].iloc[2] == sample_missing_data["BILL_AMOUNT1"].median()

    # Verify categorical columns filled with mode
    assert not pd.isna(df["EDUCATION"].iloc[2])


def test_detect_outliers() -> None:
    """Test outlier detection and handling."""
    df = pd.DataFrame(
        {
            "BILL_AMOUNT1": [1000, 2000, 3000, 100000, 500, 1500],  # 100000 is outlier
            "BILL_AMOUNT2": [1500, 2500, 3500, 2000, 1000, 150000],  # 150000 is outlier
        },
    )

    cleaned_df = detect_outliers(df.copy(), threshold=3)

    # Verify outliers were capped
    assert cleaned_df["BILL_AMOUNT1"].max() < 100000
    assert cleaned_df["BILL_AMOUNT2"].max() < 150000

    # Verify non-outlier values remained unchanged
    assert cleaned_df["BILL_AMOUNT1"].iloc[0] == 1000
    assert cleaned_df["BILL_AMOUNT2"].iloc[0] == 1500


def test_validate_dataset(sample_credit_data) -> None:
    """Test full dataset validation pipeline."""
    # Add some missing values
    df = sample_credit_data.copy()
    df.loc[0, "PAY_0"] = np.nan
    df.loc[1, "BILL_AMOUNT1"] = -1000

    cleaned_df = validate_dataset(df)

    # Verify no missing values
    assert not cleaned_df.isnull().any().any()

    # Verify value ranges
    assert cleaned_df["PAY_0"].between(-2, 8).all()
    assert (cleaned_df["BILL_AMOUNT1"] >= 0).all()
    assert cleaned_df["EDUCATION"].between(1, 4).all()
    assert cleaned_df["MARRIAGE"].between(1, 3).all()
    assert cleaned_df["SEX"].between(1, 2).all()


def test_edge_cases() -> None:
    """Test edge cases and empty dataframes."""
    # Empty DataFrame
    empty_df = pd.DataFrame()
    assert validate_dataset(empty_df).empty

    # Single column DataFrame
    single_col_df = pd.DataFrame({"PAY_0": [-2, 0, 2]})
    result = validate_dataset(single_col_df)
    assert len(result) == 3
    assert "PAY_0" in result.columns

    # All missing values in a column
    all_missing_df = pd.DataFrame({"BILL_AMOUNT1": [np.nan, np.nan, np.nan]})
    result = validate_dataset(all_missing_df)
    assert not result.isnull().any().any()
