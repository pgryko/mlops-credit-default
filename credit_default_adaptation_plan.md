# Credit Card Default Prediction - Project Adaptation Plan

## Overview
This document outlines the plan for adapting the existing MLOps project structure for credit card default prediction using the UCI ML Credit Card Default Dataset.

## Table of Contents
- [Project Structure Changes](#project-structure-changes)
- [Detailed Component Changes](#detailed-component-changes)
- [Development Environment](#development-environment)
- [Implementation Timeline](#implementation-timeline)
- [Testing Strategy](#testing-strategy)

## Project Structure Changes

### Configuration Changes (config.py)
- Update dataset configuration:
  ```python
  DATASET = "uciml/default-of-credit-card-clients-dataset"
  MODEL_NAME = "credit-default-classifier"
  ```
- Define new categorical features:
  ```python
  categorical = [
      "SEX", 
      "EDUCATION",
      "MARRIAGE",
      "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"
  ]
  target = "default.payment.next.month"
  ```

### Data Pipeline Changes (preproc.py)
- New preprocessing module features:
  - Remove Titanic-specific processing
  - Add credit card specific preprocessing:
    - Feature scaling for amount columns
    - Payment status encoding
    - Education/marriage status encoding
  - Add data validation checks:
    - Value range validation
    - Missing value handling
    - Outlier detection
  - Add feature engineering:
    - Payment history patterns
    - Utilization ratios
    - Payment amount trends

### Model Pipeline Changes (train.py)
- Hyperparameter optimization updates:
  - Adjust parameter ranges for credit default task
  - Add new parameters for imbalanced classification
  - Modify cross-validation strategy
- Evaluation metrics:
  - Add precision-recall AUC
  - Add confusion matrix logging
  - Add business metrics (cost matrix)
- Feature importance analysis:
  - Update SHAP analysis for credit features
  - Add domain-specific feature grouping

### Prediction Pipeline Changes (predict.py)
- Update prediction output format
- Add probability threshold optimization
- Add prediction explanations
- Add batch prediction capabilities

## Detailed Component Changes

### New Modules
1. `validation.py`:
   - Input data validation
   - Feature validation
   - Model input validation

2. `metrics.py`:
   - Custom credit scoring metrics
   - Business KPI calculations

3. `monitoring.py`:
   - Model monitoring functions
   - Drift detection
   - Alert generation

### MLOps Infrastructure Updates
- MLflow experiments:
  ```python
  "credit_default_hyperparam_tuning"
  "credit_default_full_training"
  ```
- Model registry enhancements:
  - Data version tracking
  - Feature set versioning
  - Performance metrics tracking
- Monitoring setup:
  - Feature drift detection
  - Performance monitoring
  - Data quality checks

## Development Environment

### Package Management with UV
The project uses UV for dependency management instead of pip/requirements.txt. Key files:

- `pyproject.toml`: Primary configuration file containing:
  - Project metadata
  - Python version requirement (>=3.11.9)
  - Core dependencies
  - Development dependencies
  - Tool configurations (black, ruff)

Key dependencies:
```toml
dependencies = [
    "catboost>=1.2.7",
    "mlflow",  # To be added
    "optuna>=4.2.1",
    "pandas>=2.2.3",
    "scikit-learn>=1.6.1",
    "shap>=0.46.0"
]
```

Development dependencies:
```toml
dev = [
    "black>=25.1.0",
    "mypy>=1.15.0",
    "pytest>=8.3.5",
    "ruff>=0.9.7"
]
```

### Environment Setup
```bash
# Create new virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Install UV
pip install uv

# Install dependencies
uv pip install -e .
```

## Implementation Timeline

1. Phase 1: Infrastructure Setup (Week 1)
   - Update configuration
   - Set up new MLflow experiments
   - Configure monitoring

2. Phase 2: Data Pipeline (Week 2)
   - Implement new preprocessing
   - Add validation checks
   - Create feature engineering pipeline

3. Phase 3: Model Development (Week 3)
   - Implement model training changes
   - Add new evaluation metrics
   - Set up model registry

4. Phase 4: Testing & Documentation (Week 4)
   - Unit tests
   - Integration tests
   - Documentation updates

## Testing Strategy

### Unit Tests
- Data validation tests
- Preprocessing transformation tests
- Model utility function tests

### Integration Tests
- End-to-end training pipeline
- MLflow logging verification
- Monitoring system integration

### Performance Tests
- Model metrics evaluation
- Processing time benchmarks
- Memory usage profiling

## Documentation Updates
- Update README.md
- Create model cards
- Add API documentation
- Update usage guides

## Next Steps
1. Review and approve plan
2. Set up new development branch
3. Begin Phase 1 implementation
4. Schedule regular progress reviews