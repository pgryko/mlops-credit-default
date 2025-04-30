# ARISA-MLOps: Titanic Survival Prediction

A production-ready MLOps implementation of the Titanic survival prediction model, demonstrating modern ML engineering practices including automated training, prediction pipelines, and model versioning.

## Project Overview

This project implements an end-to-end MLOps pipeline for the classic Titanic survival prediction problem, featuring:
- Automated model retraining on data/code changes
- Automated prediction pipeline
- Model versioning and experiment tracking with MLflow
- Champion/Challenger model deployment strategy

## Architecture

The system consists of two main pipelines:
1. **Training Pipeline**: Automatically retrains the model when training data or code changes
2. **Prediction Pipeline**: Generates new predictions when a model is updated


## Project Structure
```
├── ARISA_DSML/          # Main package directory
│   ├── config.py        # Configuration and constants
│   ├── predict.py       # Prediction pipeline
│   ├── preproc.py      # Data preprocessing
│   ├── resolve.py      # Model resolution logic
│   └── train.py        # Training pipeline
├── data/               # Data directory
├── .mlflow/            # MLflow tracking
├── models/             # Model artifacts
├── notebooks/          # Development notebooks
└── reports/           # Generated analysis
```

## Setup

### Prerequisites
- Python 3.11
- Kaggle account and API key

### Local Development
1. Clone the repository:
```bash
git clone <your-repo-url>
cd ARISA-MLOps
```

2. Create and activate virtual environment:
```bash
py -3.11 -m venv .venv
# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

3. Install dependencies:
```bash
make requirements
```

4. Set up Kaggle authentication:
- Place your `kaggle.json` in:
  - Windows: `C:\Users\USERNAME\.kaggle`
  - Mac/Linux: `/home/username/.config/kaggle`

### Cloud Infrastructure Setup

1. **AWS RDS (Metadata Store)**:
   - Create PostgreSQL database
   - Configure public access
   - Note connection details

2. **AWS S3 (Artifact Store)**:
   - Create bucket
   - Configure appropriate access

3. **GitHub Secrets**:
   Add the following secrets to your repository:
   - `KAGGLE_KEY`
   - `WORKFLOW_PAT`

## Usage

### Training Pipeline
The training pipeline automatically triggers when:
- Training data changes
- Model code changes
- Manual workflow dispatch

```bash
make train
```

### Prediction Pipeline
The prediction pipeline runs when:
- A new model is trained
- Prediction code changes
- Manual workflow dispatch

```bash
make predict
```

### Local Development
For local development and testing:
```bash
# Download and preprocess data
make preprocess

# Train model
make train

# Generate predictions
make predict
```

## MLflow Tracking

Access the MLflow UI through your configured tracking server to:
- Compare experiments
- View model metrics
- Access model artifacts
- Monitor model versions

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

- Original Titanic dataset from Kaggle
- MLOps architecture inspired by [ml-ops.org](https://ml-ops.org/)