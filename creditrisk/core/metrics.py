"""Credit scoring specific metrics and evaluation functions.

This module provides specialized metrics and evaluation functions for credit default prediction,
including business-focused metrics that consider the costs of false positives (wrongly denied credit)
and false negatives (unexpected defaults).

Functions:
    calculate_business_metrics: Calculate business-specific metrics for credit default prediction
    calculate_pr_auc: Calculate Precision-Recall AUC and curves
    optimize_threshold: Find optimal probability threshold based on business costs

Example:
    >>> from ARISA_DSML.metrics import calculate_business_metrics
    >>> metrics = calculate_business_metrics(y_true, y_pred, cost_matrix={
    ...     'fp_cost': 1.0,  # Cost of wrongly denying credit
    ...     'fn_cost': 5.0   # Cost of unexpected default
    ... })

"""

import numpy as np
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: dict[str, float] | None = None,
) -> dict[str, float]:
    """Calculate business-specific metrics for credit default prediction.

    This function calculates both standard classification metrics (precision, recall, F1)
    and business-specific metrics like approval rate and cost-based metrics. The cost matrix
    allows for different weights for false positives (wrongly denied credit) and false
    negatives (unexpected defaults).

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0: no default, 1: default)
    y_pred : np.ndarray
        Predicted labels (0: approve credit, 1: deny credit)
    cost_matrix : Dict[str, float], optional
        Cost matrix with keys:
        - 'fp_cost': Cost of wrongly denying credit (default: 1.0)
        - 'fn_cost': Cost of unexpected default (default: 5.0)

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - precision: Precision score
        - recall: Recall score
        - f1_score: F1 score
        - approval_rate: Proportion of approved applications
        - default_rate: Proportion of actual defaults
        - total_cost: Total cost based on false positives and negatives
        - avg_cost_per_decision: Average cost per decision
        - true_negatives: Count of correct approvals
        - false_positives: Count of wrong denials
        - false_negatives: Count of unexpected defaults
        - true_positives: Count of correct denials

    Example
    -------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 1, 1, 0])
    >>> metrics = calculate_business_metrics(
    ...     y_true,
    ...     y_pred,
    ...     cost_matrix={'fp_cost': 2.0, 'fn_cost': 10.0}
    ... )
    >>> print(f"Average cost per decision: ${metrics['avg_cost_per_decision']:.2f}")

    """
    if cost_matrix is None:
        cost_matrix = {"fp_cost": 1.0, "fn_cost": 5.0}

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate basic metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Business specific metrics
    approval_rate = (tp + fp) / len(y_true)
    default_rate = (tp + fn) / len(y_true)

    # Cost calculations
    total_cost = (fp * cost_matrix["fp_cost"]) + (fn * cost_matrix["fn_cost"])
    avg_cost_per_decision = total_cost / len(y_true)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "approval_rate": approval_rate,
        "default_rate": default_rate,
        "total_cost": total_cost,
        "avg_cost_per_decision": avg_cost_per_decision,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def calculate_pr_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculate Precision-Recall AUC and curves.

    For imbalanced classification problems like credit default prediction,
    Precision-Recall curves and AUC provide better insights than ROC curves.
    This function calculates both the PR-AUC score and the full precision-recall
    curves for detailed analysis.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0: no default, 1: default)
    y_prob : np.ndarray
        Predicted probabilities for the default class (between 0 and 1)

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray]
        - PR-AUC score: Area under the precision-recall curve
        - precision values: Array of precision values for curve plotting
        - recall values: Array of recall values for curve plotting

    Example
    -------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    >>> pr_auc, precision, recall = calculate_pr_auc(y_true, y_prob)
    >>> print(f"PR-AUC Score: {pr_auc:.3f}")

    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    return pr_auc, precision, recall


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_matrix: dict[str, float] | None = None,
    threshold_range: np.ndarray = None,
) -> tuple[float, dict[str, float]]:
    """Find optimal probability threshold based on business costs.

    This function finds the optimal probability threshold for converting predicted
    probabilities into binary decisions (approve/deny credit) by minimizing the
    total business cost. It evaluates multiple thresholds and returns the one
    that achieves the lowest total cost based on the provided cost matrix.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0: no default, 1: default)
    y_prob : np.ndarray
        Predicted probabilities for the default class (between 0 and 1)
    cost_matrix : Dict[str, float], optional
        Cost matrix with keys:
        - 'fp_cost': Cost of wrongly denying credit (default: 1.0)
        - 'fn_cost': Cost of unexpected default (default: 5.0)
    threshold_range : np.ndarray, optional
        Array of thresholds to evaluate
        Default: np.arange(0.1, 0.9, 0.01) for thorough search

    Returns
    -------
    Tuple[float, Dict[str, float]]
        - Optimal threshold that minimizes total cost
        - Dictionary of metrics at the optimal threshold

    Example
    -------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    >>> threshold, metrics = optimize_threshold(
    ...     y_true,
    ...     y_prob,
    ...     cost_matrix={'fp_cost': 1.0, 'fn_cost': 5.0}
    ... )
    >>> print(f"Optimal threshold: {threshold:.2f}")
    >>> print(f"Total cost: ${metrics['total_cost']:.2f}")

    """
    if threshold_range is None:
        threshold_range = np.arange(0.1, 0.9, 0.01)

    if cost_matrix is None:
        cost_matrix = {"fp_cost": 1.0, "fn_cost": 5.0}

    best_threshold = 0.5
    best_metrics = None
    min_cost = float("inf")

    for threshold in threshold_range:
        y_pred = (y_prob >= threshold).astype(int)
        metrics = calculate_business_metrics(y_true, y_pred, cost_matrix)

        if metrics["total_cost"] < min_cost:
            min_cost = metrics["total_cost"]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics
