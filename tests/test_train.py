"""Unit tests for model training module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import mlflow
import joblib
from pathlib import Path

from ARISA_DSML.train import (
    run_hyperopt,
    train_cv,
    train,
    plot_error_scatter,
    get_or_create_experiment
)


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    X = pd.DataFrame({
        'LIMIT_BAL': [20000, 30000, 40000, 50000],
        'AGE': [25, 35, 45, 55],
        'BILL_AMOUNT1': [1000, 2000, 3000, 4000],
        'PAY_0': [0, 1, 0, 2],
        'EDUCATION': [1, 2, 1, 3]
    })
    y = pd.Series([0, 1, 0, 1])
    categorical_indices = [3, 4]  # PAY_0 and EDUCATION columns
    return X, y, categorical_indices


@pytest.fixture
def sample_cv_results():
    """Create sample cross-validation results."""
    return pd.DataFrame({
        'iterations': range(1, 101),
        'test-F1-mean': np.random.uniform(0.7, 0.9, 100),
        'test-F1-std': np.random.uniform(0.01, 0.05, 100),
        'test-Logloss-mean': np.random.uniform(0.3, 0.5, 100),
        'test-Logloss-std': np.random.uniform(0.01, 0.05, 100)
    })


@patch('mlflow.start_run')
@patch('optuna.create_study')
def test_run_hyperopt(mock_create_study, mock_start_run, sample_training_data, tmp_path):
    """Test hyperparameter optimization."""
    X, y, categorical_indices = sample_training_data
    
    # Mock study object
    mock_study = MagicMock()
    mock_study.best_params = {
        'depth': 6,
        'learning_rate': 0.1,
        'iterations': 100
    }
    mock_create_study.return_value = mock_study
    
    # Run hyperopt with temporary directory
    with patch('ARISA_DSML.train.MODELS_DIR', tmp_path):
        best_params_path = run_hyperopt(
            X, y, categorical_indices,
            test_size=0.2, n_trials=2
        )
    
    # Verify study was created and optimized
    mock_create_study.assert_called_once()
    mock_study.optimize.assert_called_once()
    
    # Check if parameters were saved
    assert Path(best_params_path).exists()
    loaded_params = joblib.load(best_params_path)
    assert isinstance(loaded_params, dict)
    assert 'depth' in loaded_params


@patch('catboost.cv')
def test_train_cv(mock_cv, sample_training_data, sample_cv_results, tmp_path):
    """Test cross-validation training."""
    X, y, categorical_indices = sample_training_data
    
    # Mock CV results
    mock_cv.return_value = sample_cv_results
    
    params = {
        'depth': 6,
        'learning_rate': 0.1,
        'iterations': 100
    }
    
    # Run CV with temporary directory
    with patch('ARISA_DSML.train.MODELS_DIR', tmp_path):
        cv_output_path = train_cv(
            X, y, categorical_indices,
            params, eval_metric="F1"
        )
    
    # Verify CV was called with correct parameters
    mock_cv.assert_called_once()
    call_args = mock_cv.call_args[1]
    assert call_args['params'] == params
    assert call_args['fold_count'] == 5
    
    # Check if results were saved
    assert Path(cv_output_path).exists()
    loaded_results = pd.read_csv(cv_output_path)
    assert not loaded_results.empty


@patch('mlflow.start_run')
@patch('mlflow.catboost.log_model')
@patch('catboost.CatBoostClassifier')
def test_train(mock_catboost, mock_log_model, mock_start_run, 
               sample_training_data, sample_cv_results, tmp_path):
    """Test model training."""
    X, y, categorical_indices = sample_training_data
    
    # Mock CatBoost model
    mock_model = MagicMock()
    mock_catboost.return_value = mock_model
    
    params = {
        'depth': 6,
        'learning_rate': 0.1,
        'iterations': 100
    }
    
    # Run training with temporary directory
    with patch('ARISA_DSML.train.MODELS_DIR', tmp_path):
        model_path, params_path = train(
            X, y, categorical_indices,
            params=params,
            cv_results=sample_cv_results
        )
    
    # Verify model was trained with correct parameters
    mock_catboost.assert_called_once_with(**params, verbose=True)
    mock_model.fit.assert_called_once()
    
    # Check if model and parameters were saved
    assert Path(model_path).parent == tmp_path
    assert Path(params_path).exists()


def test_plot_error_scatter(sample_cv_results, tmp_path):
    """Test error scatter plot generation."""
    with patch('ARISA_DSML.train.FIGURES_DIR', tmp_path):
        fig = plot_error_scatter(
            df_plot=sample_cv_results,
            x="iterations",
            y="test-F1-mean",
            err="test-F1-std",
            name="Test Plot",
            title="Test Title",
            xtitle="Iterations",
            ytitle="F1 Score",
            yaxis_range=[0, 1]
        )
    
    # Verify figure properties
    assert fig.layout.title.text == "Test Title"
    assert fig.layout.xaxis.title.text == "Iterations"
    assert fig.layout.yaxis.title.text == "F1 Score"
    assert fig.layout.yaxis.range == [0, 1]
    
    # Check if plot was saved
    assert (tmp_path / "test-F1-mean_vs_iterations.png").exists()


@patch('mlflow.get_experiment_by_name')
@patch('mlflow.create_experiment')
def test_get_or_create_experiment(mock_create_experiment, mock_get_experiment):
    """Test experiment creation/retrieval."""
    experiment_name = "test_experiment"
    
    # Test existing experiment
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "123"
    mock_get_experiment.return_value = mock_experiment
    
    experiment_id = get_or_create_experiment(experiment_name)
    assert experiment_id == "123"
    mock_get_experiment.assert_called_with(experiment_name)
    mock_create_experiment.assert_not_called()
    
    # Test new experiment
    mock_get_experiment.return_value = None
    mock_create_experiment.return_value = "456"
    
    experiment_id = get_or_create_experiment(experiment_name)
    assert experiment_id == "456"
    mock_create_experiment.assert_called_with(experiment_name)


def test_edge_cases(sample_training_data, tmp_path):
    """Test edge cases for training functions."""
    X, y, categorical_indices = sample_training_data
    
    # Test training with empty parameters
    with patch('ARISA_DSML.train.MODELS_DIR', tmp_path), \
         patch('mlflow.start_run'), \
         patch('catboost.CatBoostClassifier'):
        model_path, params_path = train(
            X, y, categorical_indices,
            params=None,
            cv_results=pd.DataFrame()
        )
        assert Path(model_path).parent == tmp_path
        assert Path(params_path).exists()
    
    # Test CV with minimal parameters
    with patch('ARISA_DSML.train.MODELS_DIR', tmp_path), \
         patch('catboost.cv') as mock_cv:
        mock_cv.return_value = pd.DataFrame()
        cv_output_path = train_cv(
            X, y, categorical_indices,
            params={}, eval_metric="F1"
        )
        assert Path(cv_output_path).exists()


def test_integration(sample_training_data, sample_cv_results, tmp_path):
    """Test integration between training functions."""
    X, y, categorical_indices = sample_training_data
    
    # Mock all external dependencies
    with patch('ARISA_DSML.train.MODELS_DIR', tmp_path), \
         patch('mlflow.start_run'), \
         patch('optuna.create_study') as mock_create_study, \
         patch('catboost.cv') as mock_cv, \
         patch('catboost.CatBoostClassifier'):
        
        # Setup mocks
        mock_study = MagicMock()
        mock_study.best_params = {'depth': 6, 'learning_rate': 0.1}
        mock_create_study.return_value = mock_study
        mock_cv.return_value = sample_cv_results
        
        # Run full training pipeline
        best_params_path = run_hyperopt(X, y, categorical_indices, n_trials=2)
        params = joblib.load(best_params_path)
        cv_output_path = train_cv(X, y, categorical_indices, params)
        cv_results = pd.read_csv(cv_output_path)
        model_path, params_path = train(
            X, y, categorical_indices,
            params=params,
            cv_results=cv_results
        )
        
        # Verify artifacts
        assert Path(best_params_path).exists()
        assert Path(cv_output_path).exists()
        assert Path(model_path).exists()
        assert Path(params_path).exists()