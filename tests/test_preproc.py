"""Unit tests for preprocessing module."""

import pandas as pd
import pytest

from creditrisk.data.preproc import (
    encode_categorical_features,
    engineer_features,
    preprocess_df,
    scale_amount_features,
)


@pytest.fixture
def sample_credit_data():
    """Create sample credit card data for testing."""
    return pd.DataFrame(
        {
            "LIMIT_BAL": [20000, 30000, 40000],
            "SEX": [1, 2, 1],
            "EDUCATION": [1, 2, 3],
            "MARRIAGE": [1, 2, 2],
            "PAY_0": [-1, 0, 2],
            "PAY_2": [0, 0, -1],
            "BILL_AMOUNT1": [3000, 1000, 2000],
            "BILL_AMOUNT2": [2000, 2000, 1000],
            "PAY_AMT1": [1000, 500, 1500],
            "PAY_AMT2": [800, 600, 1000],
        },
    )


def test_encode_categorical_features(sample_credit_data) -> None:
    """Test categorical feature encoding."""
    df = encode_categorical_features(sample_credit_data.copy())

    # Check education encoding
    assert "EDUCATION_graduate" in df.columns
    assert "EDUCATION_university" in df.columns
    assert "EDUCATION_high_school" in df.columns

    # Check marriage encoding
    assert "MARRIAGE_married" in df.columns
    assert "MARRIAGE_single" in df.columns

    # Check payment status encoding
    assert "PAY_0_paid_full" in df.columns
    assert "PAY_0_revolving" in df.columns
    assert "PAY_0_delay_2m" in df.columns

    # Verify original categorical columns were dropped
    assert "EDUCATION" not in df.columns
    assert "MARRIAGE" not in df.columns
    assert "PAY_0" not in df.columns

    # Check one-hot encoding is binary
    one_hot_cols = [
        col
        for col in df.columns
        if any(x in col for x in ["EDUCATION_", "MARRIAGE_"])
        or (col.startswith("PAY_") and not col.startswith("PAY_AMT"))
    ]
    assert df[one_hot_cols].isin([0, 1]).all().all()


def test_scale_amount_features(sample_credit_data) -> None:
    """Test amount feature scaling."""
    df = scale_amount_features(sample_credit_data.copy())

    # Check both "AMOUNT" and "AMT" columns since the implementation scales both
    amount_cols = [col for col in df.columns if ("AMOUNT" in col or "AMT" in col)]

    # Check scaling results
    for col in amount_cols:
        scaled_values = df[col].values
        assert abs(scaled_values.mean()) < 1e-10  # Close to 0
        assert abs(scaled_values.std() - 1.0) < 1e-10  # Close to 1

    # Non-amount columns should remain unchanged
    non_amount_cols = [col for col in df.columns if ("AMOUNT" not in col and "AMT" not in col)]
    for col in non_amount_cols:
        assert df[col].equals(sample_credit_data[col])


def test_engineer_features(sample_credit_data) -> None:
    """Test feature engineering."""
    df = engineer_features(sample_credit_data.copy())

    # Check utilization ratios
    assert "UTILIZATION_RATIO_1" in df.columns
    assert df["UTILIZATION_RATIO_1"].equals(df["BILL_AMOUNT1"] / df["LIMIT_BAL"])

    # Check average utilization
    assert "AVG_UTILIZATION" in df.columns

    # Check payment metrics (always available)
    assert "TOTAL_PAYMENT" in df.columns
    assert "AVG_PAYMENT" in df.columns
    assert "PAYMENT_CONSISTENCY" in df.columns

    # Check bill metrics (always available)
    assert "TOTAL_BILL" in df.columns
    assert "AVG_BILL" in df.columns

    # Optional metrics - depend on full column availability
    # PAYMENT_TREND requires all PAY_AMT1 through PAY_AMT6 to be present
    # BILL_TREND requires all BILL_AMOUNT1 through BILL_AMOUNT6 to be present

    # We don't test for these in this test since not all columns are available
    # assert "PAYMENT_TREND" in df.columns
    # assert "BILL_TREND" in df.columns

    # Verify payment consistency is between 0 and 1
    assert (df["PAYMENT_CONSISTENCY"] >= 0).all()
    assert (df["PAYMENT_CONSISTENCY"] <= 1).all()


def test_preprocess_df(tmp_path) -> None:
    """Test full preprocessing pipeline."""
    # Create a temporary CSV file
    input_df = pd.DataFrame(
        {
            "LIMIT_BAL": [20000, 30000],
            "SEX": [1, 2],
            "EDUCATION": [1, 2],
            "MARRIAGE": [1, 2],
            "PAY_0": [-1, 0],
            "BILL_AMOUNT1": [3000, 1000],
            "PAY_AMT1": [1000, 500],
            "default.payment.next.month": [0, 1],
        },
    )

    input_path = tmp_path / "test_input.csv"
    input_df.to_csv(input_path, index=False)

    # Run preprocessing
    output_path = preprocess_df(input_path)

    # Load and check processed data
    processed_df = pd.read_csv(output_path)

    # Check if all expected transformations were applied
    assert "EDUCATION_graduate" in processed_df.columns  # Categorical encoding
    assert abs(processed_df["BILL_AMOUNT1"].mean()) < 1e-10  # Scaling
    assert "UTILIZATION_RATIO_1" in processed_df.columns  # Feature engineering

    # Check if target variable is preserved
    assert "default.payment.next.month" in processed_df.columns


def test_edge_cases() -> None:
    """Test edge cases for preprocessing functions."""
    # Empty DataFrame
    empty_df = pd.DataFrame()
    encoded_empty = encode_categorical_features(empty_df.copy())
    assert encoded_empty.empty

    # Missing columns
    minimal_df = pd.DataFrame(
        {
            "LIMIT_BAL": [10000],
            "BILL_AMOUNT1": [1000],
        },
    )

    # Should not raise errors
    scaled_df = scale_amount_features(minimal_df.copy())
    assert "BILL_AMOUNT1" in scaled_df.columns

    engineered_df = engineer_features(minimal_df.copy())
    assert "UTILIZATION_RATIO_1" in engineered_df.columns

    # DataFrame with all zeros
    zero_df = pd.DataFrame(
        {
            "BILL_AMOUNT1": [0, 0, 0],
            "BILL_AMOUNT2": [0, 0, 0],
        },
    )
    scaled_zero = scale_amount_features(zero_df.copy())
    assert (scaled_zero == 0).all().all()
