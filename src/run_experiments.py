"""Compare interpolation methods across repeated artificial sensor outages."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_FOLDER / "data/processed/temperature_clean.csv"
RESULTS_FOLDER = PROJECT_FOLDER / "results"

GAP_HOURS = [0.5, 1, 2, 4, 8]


def calculate_errors(actual, predicted):
    """Calculate prediction errors in degrees Celsius."""
    errors = predicted - actual

    return {
        "mae_c": float(np.mean(np.abs(errors))),
        "rmse_c": float(np.sqrt(np.mean(errors**2))),
        "max_absolute_error_c": float(np.max(np.abs(errors))),
    }


def main():
    """Hide measurements, reconstruct them, and save comparison results."""
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(INPUT_PATH)
    data["time"] = pd.to_datetime(
        data["time_utc"], utc=True
    ).dt.tz_convert("Australia/Melbourne")

    data = data.sort_values("timestamp_seconds").reset_index(drop=True)

    seconds = data["timestamp_seconds"].to_numpy(dtype=float)
    temperatures = data["temperature_c"].to_numpy(dtype=float)

    if not np.all(np.isfinite(seconds)):
        raise ValueError("Timestamps contain invalid values.")

    if not np.all(np.isfinite(temperatures)):
        raise ValueError("Temperatures contain invalid values.")

    if not np.all(np.diff(seconds) > 0):
        raise ValueError("Timestamps must be unique and increasing.")

    # One preselected start per day, with different times of day.
    # No start is selected using the resulting prediction errors.
    days = pd.date_range(
        "2024-03-02",
        "2024-04-29",
        freq="D",
        tz="Australia/Melbourne",
    )

    rows = []
    skipped_cases = 0

    for day_number, day in enumerate(days):
        start_hour = [0, 6, 12, 18][day_number % 4]
        gap_start = day + pd.Timedelta(hours=start_hour)

        for duration in GAP_HOURS:
            gap_end = gap_start + pd.Timedelta(hours=duration)

            # Supply each method with six hours of observations
            # on either side of the simulated outage.
            window_start = gap_start - pd.Timedelta(hours=6)
            window_end = gap_end + pd.Timedelta(hours=6)

            window = data.loc[
                (data["time"] >= window_start)
                & (data["time"] <= window_end)
            ].copy()

            hidden = (
                (window["time"] >= gap_start)
                & (window["time"] < gap_end)
            ).to_numpy()

            known = ~hidden

            if hidden.sum() == 0 or known.sum() < 4:
                skipped_cases += 1
                continue

            window_seconds = window["timestamp_seconds"].to_numpy(
                dtype=float
            )
            x = (window_seconds - window_seconds[0]) / 3600
            y = window["temperature_c"].to_numpy(dtype=float)

            x_known = x[known]
            y_known = y[known]
            x_hidden = x[hidden]
            actual = y[hidden]

            # Interpolation needs measurements on both sides.
            if (
                x_hidden.min() <= x_known.min()
                or x_hidden.max() >= x_known.max()
            ):
                skipped_cases += 1
                continue

            linear_predictions = np.interp(
                x_hidden, x_known, y_known
            )

            spline = CubicSpline(
                x_known,
                y_known,
                bc_type="natural",
                extrapolate=False,
            )
            spline_predictions = spline(x_hidden)

            predictions = {
                "Linear": linear_predictions,
                "Natural cubic spline": spline_predictions,
            }

            for method, predicted in predictions.items():
                if not np.all(np.isfinite(predicted)):
                    raise ValueError(
                        f"Invalid predictions for {method} at {gap_start}"
                    )

                rows.append(
                    {
                        "gap_start": gap_start.isoformat(),
                        "gap_hours": duration,
                        "method": method,
                        "hidden_readings": int(hidden.sum()),
                        **calculate_errors(actual, predicted),
                    }
                )

    if not rows:
        raise ValueError("No suitable gaps were found.")

    detailed_results = pd.DataFrame(rows)
    detailed_results.to_csv(
        RESULTS_FOLDER / "gap_experiment_results.csv",
        index=False,
    )

    # Each outage receives equal weight in these averages.
    summary = (
        detailed_results.groupby(["gap_hours", "method"])
        .agg(
            number_of_gaps=("mae_c", "size"),
            mean_mae_c=("mae_c", "mean"),
            median_mae_c=("mae_c", "median"),
            mean_rmse_c=("rmse_c", "mean"),
            worst_absolute_error_c=("max_absolute_error_c", "max"),
        )
        .reset_index()
    )

    summary.to_csv(
        RESULTS_FOLDER / "gap_experiment_summary.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for method in ["Linear", "Natural cubic spline"]:
        method_results = summary.loc[summary["method"] == method]

        axes[0].plot(
            method_results["gap_hours"],
            method_results["mean_mae_c"],
            marker="o",
            label=method,
        )
        axes[1].plot(
            method_results["gap_hours"],
            method_results["mean_rmse_c"],
            marker="o",
            label=method,
        )

    for axis, ylabel in zip(
        axes,
        ["Mean outage MAE (°C)", "Mean outage RMSE (°C)"],
    ):
        axis.set_xlabel("Simulated outage duration (hours)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(GAP_HOURS)
        axis.grid(alpha=0.3)
        axis.legend()

    fig.suptitle("Temperature reconstruction: March–April 2024")
    fig.tight_layout()
    fig.savefig(
        RESULTS_FOLDER / "gap_experiment_errors.png",
        dpi=200,
    )
    plt.close(fig)

    print("\nResults: each outage has equal weight.\n")
    print(summary.to_string(index=False, float_format="%.4f"))
    print(f"\nSkipped outage cases: {skipped_cases}")
    print("\nCreated:")
    print("results/gap_experiment_results.csv")
    print("results/gap_experiment_summary.csv")
    print("results/gap_experiment_errors.png")
    print("\nMay and June were not used in this experiment.")


if __name__ == "__main__":
    main()