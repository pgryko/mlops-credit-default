#!/usr/bin/env python3
"""Create test dataset from training data for prediction workflow."""

import os
from pathlib import Path

import pandas as pd

processed_dir = Path("data/processed")
train_file = processed_dir / "train.csv"

if os.path.exists(train_file):
    # Read the first 100 rows of training data
    train_df = pd.read_csv(train_file, nrows=100)

    # Save as test.csv
    test_file = processed_dir / "test.csv"
    train_df.to_csv(test_file, index=False)
    print(f"Created test dataset with {len(train_df)} samples at {test_file}")
else:
    print("Training data not found at", train_file)
