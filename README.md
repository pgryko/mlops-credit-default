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
├── creditrisk/          # Main package directory
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

   To create and set up your Kaggle API key:
   - Log in to your Kaggle account at [kaggle.com](https://www.kaggle.com/)
   - Go to "Account" by clicking on your profile picture in the top-right corner
   - Scroll down to the "API" section
   - Click "Create New API Token" - this will download a `kaggle.json` file
   - Place your `kaggle.json` in:
     - Windows: `C:\Users\USERNAME\.kaggle`
     - Mac/Linux: `/home/username/.config/kaggle`
   - Ensure the permissions are secure: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

5. Set up GitHub Personal Access Token (for workflows):

   To create a GitHub PAT for workflow automation:
   - Log in to your GitHub account
   - Go to "Settings" → "Developer settings" → "Personal access tokens"
   - Click "Generate new token" (choose "Fine-grained tokens" for better security)
   - Give your token a descriptive name
   - Set an appropriate expiration date
   - Select required permissions (typically "repo" for full repository access)
   - Click "Generate token"
   - **IMPORTANT**: Copy and save your token securely - GitHub will only show it once!
   - Store it as a repository secret:
     - Go to your repository → "Settings" → "Secrets and variables" → "Actions"
     - Click "New repository secret"
     - Name it `WORKFLOW_PAT` (or your preferred name)
     - Paste your token and click "Add secret"

## Usage

### Data Pipeline
Process and validate new credit card data:
```bash
# Download and preprocess data
python -m creditrisk.preproc

# Validate specific dataset
python -m creditrisk.validation --input path/to/data.csv
```

### Training Pipeline
Train a new model with optimized hyperparameters:
```bash
# Full training pipeline with hyperparameter optimization
python -m creditrisk.train

# Cross-validation only
python -m creditrisk.train --cv-only

# Quick training with default parameters
python -m creditrisk.train --quick
```

### Prediction Pipeline
Generate default predictions for new customers:
```bash
# Batch predictions
python -m creditrisk.predict --input new_customers.csv --output predictions.csv

# Single prediction with explanation
python -m creditrisk.predict --explain customer_data.json
```

## CI/CD and Automated Workflows

This project implements continuous integration and continuous delivery through GitHub Actions workflows that automate key MLOps processes.

### Automated Model Retraining

The system automatically retrains the model whenever relevant changes are detected, using the `retrain_on_change.yml` workflow:

#### Workflow Triggers
- Push to main branch affecting:
  - Data files (`data/raw/train.csv`, `data/processed/train.csv`)
  - Python code in the `creditrisk` package
  - Model hyperparameter files (`models/best_params.pkl`)
  - Workflow file itself
- Manual trigger via GitHub Actions UI

#### Required Secrets
- `KAGGLE_KEY`: Your Kaggle API credentials in JSON format (see Setup section)
- `WORKFLOW_PAT`: GitHub Personal Access Token with repo permissions (see Setup section)

#### Workflow Steps
1. **Preprocessing Job**:
   - Sets up Python environment
   - Configures Kaggle authentication
   - Runs data preprocessing with automatic retries
   - Uploads processed data as artifact
   - Creates an issue if preprocessing fails

2. **Training Job**:
   - Downloads processed data artifact
   - Sets up Python and MLflow
   - Runs model training with automatic retries
   - Temporarily disables branch protection using the PAT
   - Commits and pushes model artifacts, MLflow data, and reports
   - Restores branch protection rules
   - Triggers prediction workflow
   - Creates an issue if training fails

### Setting Up Required Secrets

For the workflow to function properly, you must add the following secrets to your GitHub repository:

1. **KAGGLE_KEY**: The entire contents of your `kaggle.json` file
2. **WORKFLOW_PAT**: A GitHub Personal Access Token with `repo` scope

See the Setup section above for detailed instructions on creating these secrets.

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