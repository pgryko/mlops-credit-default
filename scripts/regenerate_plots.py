"""Script to regenerate plots from existing CV results."""

from pathlib import Path

import pandas as pd

from creditrisk.core.config import FIGURES_DIR
from creditrisk.models.train import plot_error_scatter


def main():
    """Regenerate plots from existing CV results."""
    # Path to CV results
    cv_results_path = Path("models/cv_results.csv")

    # Ensure the figures directory exists
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load CV results
    cv_results = pd.read_csv(cv_results_path)

    # Regenerate F1 score plot
    fig1 = plot_error_scatter(
        df_plot=cv_results,
        name="Mean F1 Score",
        title="Cross-Validation (N=5) Mean F1 score with Error Bands",
        xtitle="Training Steps",
        ytitle="Performance Score",
        yaxis_range=[0.43, 0.49],  # Adjusted to better show the actual data range (0.43-0.48)
    )

    # Regenerate Logloss plot
    fig2 = plot_error_scatter(
        cv_results,
        x="iterations",
        y="test-Logloss-mean",
        err="test-Logloss-std",
        name="Mean Logloss",
        title="Cross-Validation (N=5) Mean Logloss with Error Bands",
        xtitle="Training Steps",
        ytitle="Logloss",
    )

    print(f"Plots regenerated and saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
