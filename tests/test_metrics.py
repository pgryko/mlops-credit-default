"""Unit tests for credit scoring metrics module."""

import numpy as np
import pytest

from creditrisk.core.metrics import (
    calculate_business_metrics,
    calculate_pr_auc,
    optimize_threshold,
)


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.6, 0.3, 0.8, 0.4, 0.9, 0.2, 0.7])
    return y_true, y_pred, y_prob


def test_calculate_business_metrics(sample_predictions) -> None:
    """Test business metrics calculation."""
    y_true, y_pred, _ = sample_predictions

    # Test with default cost matrix
    metrics = calculate_business_metrics(y_true, y_pred)

    # Check basic metrics
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1

    # Check business metrics
    assert 0 <= metrics["approval_rate"] <= 1
    assert 0 <= metrics["default_rate"] <= 1
    assert metrics["total_cost"] >= 0
    assert metrics["avg_cost_per_decision"] >= 0

    # Check confusion matrix components
    assert metrics["true_negatives"] >= 0
    assert metrics["false_positives"] >= 0
    assert metrics["false_negatives"] >= 0
    assert metrics["true_positives"] >= 0

    # Test with custom cost matrix
    custom_costs = {"fp_cost": 2.0, "fn_cost": 10.0}
    custom_metrics = calculate_business_metrics(y_true, y_pred, custom_costs)

    # Custom costs should result in higher total cost
    assert custom_metrics["total_cost"] > metrics["total_cost"]


def test_calculate_pr_auc(sample_predictions) -> None:
    """Test Precision-Recall AUC calculation."""
    y_true, _, y_prob = sample_predictions

    pr_auc, precision, recall = calculate_pr_auc(y_true, y_prob)

    # Check PR-AUC score
    assert 0 <= pr_auc <= 1

    # Check precision and recall arrays
    assert len(precision) == len(recall)
    assert np.all(precision >= 0)
    assert np.all(precision <= 1)
    assert np.all(recall >= 0)
    assert np.all(recall <= 1)

    # Check monotonicity of recall
    assert np.all(np.diff(recall) >= 0) or np.all(np.diff(recall) <= 0)


def test_optimize_threshold(sample_predictions) -> None:
    """Test threshold optimization."""
    y_true, _, y_prob = sample_predictions

    # Test with default parameters
    threshold, metrics = optimize_threshold(y_true, y_prob)

    # Check threshold is in valid range
    assert 0 < threshold < 1

    # Check returned metrics
    assert isinstance(metrics, dict)
    assert "total_cost" in metrics
    assert metrics["total_cost"] >= 0

    # Test with custom parameters
    custom_costs = {"fp_cost": 2.0, "fn_cost": 3.0}
    custom_thresholds = np.array([0.3, 0.5, 0.7])

    threshold_custom, metrics_custom = optimize_threshold(
        y_true,
        y_prob,
        cost_matrix=custom_costs,
        threshold_range=custom_thresholds,
    )

    # Check threshold is one of the custom values
    assert threshold_custom in custom_thresholds


def test_edge_cases() -> None:
    """Test edge cases for metrics functions."""
    # All negative predictions
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    np.array([0.1, 0.2, 0.3, 0.4])

    metrics_all_neg = calculate_business_metrics(y_true, y_pred)
    assert metrics_all_neg["precision"] == 0  # No positive predictions
    assert metrics_all_neg["recall"] == 0  # No true positives

    # All positive predictions
    y_pred_all_pos = np.ones_like(y_pred)
    metrics_all_pos = calculate_business_metrics(y_true, y_pred_all_pos)
    assert metrics_all_pos["approval_rate"] == 1.0  # All approved

    # Perfect predictions
    y_true_perfect = np.array([0, 1, 0, 1])
    y_pred_perfect = np.array([0, 1, 0, 1])
    y_prob_perfect = np.array([0.1, 0.9, 0.1, 0.9])

    metrics_perfect = calculate_business_metrics(y_true_perfect, y_pred_perfect)
    assert metrics_perfect["precision"] == 1.0
    assert metrics_perfect["recall"] == 1.0
    assert metrics_perfect["f1_score"] == 1.0

    pr_auc_perfect, _, _ = calculate_pr_auc(y_true_perfect, y_prob_perfect)
    assert pr_auc_perfect == 1.0


def test_threshold_optimization_consistency() -> None:
    """Test consistency of threshold optimization."""
    # Create synthetic data with clear optimal threshold
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8])

    # Test with different cost matrices
    cost_matrices = [
        {"fp_cost": 1.0, "fn_cost": 1.0},  # Balanced costs
        {"fp_cost": 1.0, "fn_cost": 5.0},  # High false negative cost
        {"fp_cost": 5.0, "fn_cost": 1.0},  # High false positive cost
    ]

    thresholds = []
    for cost_matrix in cost_matrices:
        threshold, _ = optimize_threshold(y_true, y_prob, cost_matrix)
        thresholds.append(threshold)

    # Higher FN cost should lead to lower threshold
    assert thresholds[1] <= thresholds[0]
    # Higher FP cost should lead to higher threshold
    assert thresholds[2] >= thresholds[0]
