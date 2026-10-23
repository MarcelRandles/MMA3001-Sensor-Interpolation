"""Evaluate the unchanged interpolation methods on the reserved period."""

from pathlib import Path

import run_experiments


def main():
    """Run the same experiment on May–June data."""
    project_folder = Path(__file__).resolve().parents[1]

    # Use complete days with enough surrounding measurements.
    # All context for these outages remains inside May–June.
    run_experiments.START_DATE = "2024-05-02"
    run_experiments.END_DATE = "2024-06-12"
    run_experiments.PERIOD_LABEL = "May–June 2024: final validation"

    # Keep the original March–April results.
    run_experiments.RESULTS_FOLDER = (
        project_folder / "results" / "validation"
    )

    print("Starting final validation on May–June data.")
    print("Methods, outage lengths and context settings are unchanged.")

    run_experiments.main()

    print("\nValidation files are in results/validation/")


if __name__ == "__main__":
    main()