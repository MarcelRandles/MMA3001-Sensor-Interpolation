"""Compare interpolation methods on one artificial temperature-data gap."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_FOLDER / "data" / "processed" / "temperature_clean.csv"
RESULTS_FOLDER = PROJECT_FOLDER / "results"


def calculate_errors(actual, predicted):
    """Calculate reconstruction errors at withheld measurement times.

    Args:
        actual: Array of withheld measured temperatures in degrees Celsius.
        predicted: Array of estimated temperatures at the same times.

    Returns:
        Dictionary containing MAE, RMSE and maximum absolute error in °C.
    """
    error = np.asarray(predicted) - np.asarray(actual)

    return {
        "mae_c": float(np.mean(np.abs(error))),
        "rmse_c": float(np.sqrt(np.mean(error ** 2))),
        "max_absolute_error_c": float(np.max(np.abs(error))),
    }


def main():
    """Hide readings, reconstruct them and save a comparison plot and table."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            "Cleaned data is missing. Run src/inspect_data.py first."
        )

    data = pd.read_csv(INPUT_PATH)

    # Read UTC times consistently, then convert for display and selection.
    data["time"] = pd.to_datetime(
        data["time_utc"], utc=True
    ).dt.tz_convert("Australia/Melbourne")

    data = data.sort_values("timestamp_seconds").reset_index(drop=True)

    # This demonstration uses March only.
    # Leave May and June out of this experiment for later evaluation.
    window_start = pd.Timestamp(
        "2024-03-04 00:00", tz="Australia/Melbourne"
    )
    window_end = window_start + pd.Timedelta(days=1)
    gap_start = window_start + pd.Timedelta(hours=12)
    gap_end = gap_start + pd.Timedelta(hours=2)

    window = data[
        (data["time"] >= window_start)
        & (data["time"] < window_end)
    ].copy()

    if len(window) < 4:
        raise ValueError("Not enough readings in the selected day.")

    # Use actual elapsed hours, preserving irregular measurement intervals.
    seconds = window["timestamp_seconds"].to_numpy(dtype=float)
    x = (seconds - seconds[0]) / 3600
    y = window["temperature_c"].to_numpy(dtype=float)

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Data contains missing or nonfinite values.")

    if not np.all(np.diff(x) > 0):
        raise ValueError("Measurement times must be unique and increasing.")

    # True means this reading is hidden from ALL interpolation methods.
    hidden = (
        (window["time"] >= gap_start)
        & (window["time"] < gap_end)
    ).to_numpy()

    known = ~hidden

    if hidden.sum() == 0:
        raise ValueError("No measurements fall inside the selected gap.")

    if known.sum() < 4:
        raise ValueError("At least four known readings are required.")

    if x[hidden].min() <= x[known].min() or x[hidden].max() >= x[known].max():
        raise ValueError("The gap must have known readings on both sides.")

    # Linear interpolation joins neighbouring known readings with straight lines.
    linear_prediction = np.interp(x[hidden], x[known], y[known])

    # A natural cubic spline joins cubic curves smoothly.
    # 'Natural' specifies zero second derivative at the window endpoints.
    spline = CubicSpline(
        x[known],
        y[known],
        bc_type="natural",
        extrapolate=False,
    )
    spline_prediction = spline(x[hidden])

    predictions = {
        "Linear": linear_prediction,
        "Natural cubic spline": spline_prediction,
        "PCHIP": PchipInterpolator(
            x[known], y[known], extrapolate=False
        )(x[hidden]),
    }

    rows = []

    for method, predicted in predictions.items():
        rows.append({
            "method": method,
            "hidden_readings": int(hidden.sum()),
            "simulated_outage_hours": 2,
            **calculate_errors(y[hidden], predicted),
        })

    scores = pd.DataFrame(rows)
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

    scores_path = RESULTS_FOLDER / "interpolation_demo_scores.csv"
    scores.to_csv(scores_path, index=False)

    # Save the withheld values and estimates for checking the calculations.
    comparison = pd.DataFrame({
        "time_utc": window.loc[hidden, "time_utc"].to_numpy(),
        "actual_temperature_c": y[hidden],
        "linear_prediction_c": linear_prediction,
        "spline_prediction_c": spline_prediction,
        "pchip_prediction_c": predictions["PCHIP"],
    })

    predictions_path = RESULTS_FOLDER / "interpolation_demo_predictions.csv"
    comparison.to_csv(predictions_path, index=False)

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        window["time"],
        y,
        color="grey",
        linewidth=1,
        label="Recorded temperature (reference)",
    )

    ax.scatter(
        window.loc[known, "time"],
        y[known],
        color="black",
        s=12,
        label="Readings available to methods",
    )

    ax.scatter(
        window.loc[hidden, "time"],
        y[hidden],
        color="grey",
        marker="x",
        s=35,
        label="Hidden readings",
    )

    for method, predicted in predictions.items():
        ax.plot(
            window.loc[hidden, "time"],
            predicted,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=method,
        )

    ax.axvspan(
        gap_start,
        gap_end,
        color="orange",
        alpha=0.12,
        label="Artificial two-hour outage",
    )

    ax.set_title("First interpolation comparison — 4 March 2024")
    ax.set_xlabel("Melbourne local time")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()

    plot_path = RESULTS_FOLDER / "interpolation_demo.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print("\nErrors calculated ONLY at hidden measurement times:")
    print(scores.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nCreated:")
    for path in [scores_path, predictions_path, plot_path]:
        print(path.relative_to(PROJECT_FOLDER))

    print("\nOne example cannot establish which method is generally better.")


if __name__ == "__main__":
    main()
