# ARISA-MLOps: Credit Card Default Prediction

A production-ready MLOps implementation of a credit card default prediction model, demonstrating modern ML engineering practices including automated training, prediction pipelines, model versioning, and business-focused metrics.

## Project Overview

This project implements an end-to-end MLOps pipeline for credit card default prediction using the UCI ML Credit Card Default Dataset, featuring:
- Automated model retraining with hyperparameter optimization
- Business-focused metrics and cost-sensitive evaluation
- Model versioning and experiment tracking with MLflow
- Champion/Challenger model deployment strategy
- Comprehensive data validation and preprocessing
- Batch prediction capabilities

## Architecture

The system consists of three main pipelines:
1. **Data Pipeline**: Validates and preprocesses credit card data with domain-specific checks
2. **Training Pipeline**: Automatically retrains the model with cost-sensitive optimization
3. **Prediction Pipeline**: Generates probability-based default predictions with explanations

## Project Structure
```
├── ARISA_DSML/          # Main package directory
│   ├── config.py        # Configuration and constants
│   ├── validation.py    # Data validation
│   ├── preproc.py      # Data preprocessing
│   ├── metrics.py      # Business metrics
│   ├── train.py        # Training pipeline
│   ├── predict.py      # Prediction pipeline
│   └── resolve.py      # Model resolution logic
├── data/               # Data directory
├── docs/              # Documentation and model cards
├── .mlflow/           # MLflow tracking
├── models/            # Model artifacts
├── notebooks/         # Development notebooks
└── reports/          # Generated analysis
```

## Setup

### Prerequisites
- Python 3.11.9 or higher
- Kaggle account and API key
- UV package manager

### Local Development
1. Clone the repository:
```bash
git clone <your-repo-url>
cd ARISA-MLOps
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

3. Install UV and dependencies:
```bash
pip install uv
uv pip install -e .
```

4. Set up Kaggle authentication:
- Place your `kaggle.json` in:
  - Windows: `C:\Users\USERNAME\.kaggle`
  - Mac/Linux: `/home/username/.config/kaggle`

## Usage

### Data Pipeline
Process and validate new credit card data:
```bash
# Download and preprocess data
python -m ARISA_DSML.preproc

# Validate specific dataset
python -m ARISA_DSML.validation --input path/to/data.csv
```

### Training Pipeline
Train a new model with optimized hyperparameters:
```bash
# Full training pipeline with hyperparameter optimization
python -m ARISA_DSML.train

# Cross-validation only
python -m ARISA_DSML.train --cv-only

# Quick training with default parameters
python -m ARISA_DSML.train --quick
```

### Prediction Pipeline
Generate default predictions for new customers:
```bash
# Batch predictions
python -m ARISA_DSML.predict --input new_customers.csv --output predictions.csv

# Single prediction with explanation
python -m ARISA_DSML.predict --explain customer_data.json
```

## Model Performance

The model is optimized for business impact using:
- Precision-Recall AUC for imbalanced classification
- Custom cost matrix (FP: wrongly denied credit, FN: default loss)
- Business metrics (approval rate, default rate, avg cost per decision)

Current performance metrics:
- PR-AUC: 0.XX
- F1 Score: 0.XX
- Avg Cost per Decision: $XX.XX

## MLflow Tracking

Access the MLflow UI to:
- Compare experiments and model versions
- View business metrics and performance benchmarks
- Access model artifacts and SHAP explanations
- Monitor model drift and data quality

Key experiments:
- `credit_default_hyperparam_tuning`: Hyperparameter optimization runs
- `credit_default_full_training`: Production model training

## Model Card

See [docs/model_card.md](docs/model_card.md) for detailed information about:
- Model characteristics and architecture
- Training data and preprocessing
- Performance benchmarks and metrics
- Intended use cases and limitations
- Fairness considerations and bias analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT

## Contact

Piotr Gryko

## Acknowledgments

- UCI ML Credit Card Default Dataset
- MLOps architecture inspired by [ml-ops.org](https://ml-ops.org/)