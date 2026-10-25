"""Plot recorded temperatures against four predictions for a fixed outage."""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator

from newton_interpolation import local_newton
from run_experiments import calculate_errors

PROJECT_FOLDER = Path(__file__).resolve().parents[1]
# First scheduled main-study start, selected without inspecting its errors.
GAP_START = "2024-03-02 00:00:00"
GAP_HOURS = 24


def main():
    """Read the clean CSV and save a fixed-case plot, predictions and scores.

    Uses six hours of available readings on each side, matching the main
    experiment. Hidden readings are reference values only, never fit inputs.
    Raises ValueError for unsupported data or insufficient bracketing readings.
    """
    results = PROJECT_FOLDER / "results"
    results.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(PROJECT_FOLDER / "data/processed/temperature_clean.csv")
    data["time"] = pd.to_datetime(data["time_utc"], utc=True).dt.tz_convert(
        "Australia/Melbourne"
    )
    data = data.sort_values("timestamp_seconds")
    start = pd.Timestamp(GAP_START, tz="Australia/Melbourne")
    end = start + pd.Timedelta(hours=GAP_HOURS)
    window = data.loc[
        (data["time"] >= start - pd.Timedelta(hours=6))
        & (data["time"] <= end + pd.Timedelta(hours=6))
    ].copy()
    if window.empty:
        raise ValueError("No measurements in the selected window.")
    hidden = ((window["time"] >= start) & (window["time"] < end)).to_numpy()
    known = ~hidden
    seconds = window["timestamp_seconds"].to_numpy(dtype=float)
    y = window["temperature_c"].to_numpy(dtype=float)
    if not np.isfinite(seconds).all() or not np.isfinite(y).all():
        raise ValueError("Times and temperatures must be finite.")
    if not np.all(np.diff(seconds) > 0):
        raise ValueError("Measurement times must be unique and increasing.")
    x = (seconds - seconds[0]) / 3600
    if hidden.sum() == 0 or known.sum() < 4:
        raise ValueError("Need hidden readings and four available observations.")
    if (np.sum(x[known] < x[hidden].min()) < 2
            or np.sum(x[known] > x[hidden].max()) < 2):
        raise ValueError("Need at least two available readings on each side.")
    predictions = {
        "Linear": np.interp(x[hidden], x[known], y[known]),
        "Natural cubic spline": CubicSpline(
            x[known], y[known], bc_type="natural", extrapolate=False
        )(x[hidden]),
        "PCHIP": PchipInterpolator(
            x[known], y[known], extrapolate=False
        )(x[hidden]),
        "Local cubic Newton": local_newton(x[known], y[known], x[hidden]),
    }
    rows = []
    exported = pd.DataFrame({
        "time_utc": window.loc[hidden, "time_utc"].to_numpy(),
        "actual_temperature_c": y[hidden],
    })
    names = ["linear", "spline", "pchip", "newton"]
    for (method, predicted), name in zip(predictions.items(), names):
        if not np.isfinite(predicted).all():
            raise ValueError(f"Nonfinite predictions from {method}.")
        rows.append({"method": method, **calculate_errors(y[hidden], predicted)})
        exported[f"{name}_prediction_c"] = predicted
    # If the main experiment is present, verify this illustration reproduces it.
    saved_path = results / "gap_experiment_results.csv"
    if saved_path.is_file():
        saved = pd.read_csv(saved_path)
        starts = pd.to_datetime(saved["gap_start"], utc=True)
        case = saved.loc[(starts == start.tz_convert("UTC"))
                         & (saved["gap_hours"] == GAP_HOURS)]
        if len(case) != 4:
            raise ValueError("Expected four matching main-experiment scores.")
        for row in rows:
            match = case.loc[case["method"] == row["method"]]
            if len(match) != 1:
                raise ValueError("Missing or duplicate method score.")
            for metric in ["mae_c", "rmse_c", "max_absolute_error_c"]:
                np.testing.assert_allclose(
                    row[metric], match.iloc[0][metric], rtol=1e-7, atol=1e-9
                )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axvspan(start, end, color="#edb658", alpha=0.18,
               label="Simulated outage", zorder=0)
    # Grey context is available to the methods; black is the hidden reference.
    ax.plot(window.loc[known & (window["time"] < start).to_numpy(), "time"],
            y[known & (window["time"] < start).to_numpy()],
            color="0.55", marker=".", markersize=4, linewidth=1,
            label="Available recorded temperatures")
    after = known & (window["time"] >= end).to_numpy()
    ax.plot(window.loc[after, "time"], y[after], color="0.55",
            marker=".", markersize=4, linewidth=1)
    colors = ["#2468ad", "#d24d2c", "#218251", "#8250a5"]
    styles = ["-", "--", "-.", ":"]
    for (method, predicted), color, style in zip(predictions.items(), colors, styles):
        ax.plot(window.loc[hidden, "time"], predicted,
                color=color, linestyle=style, linewidth=2, label=method)
    ax.plot(window.loc[hidden, "time"], y[hidden], color="black",
            linewidth=2, label="Hidden recorded temperatures", zorder=5)
    ax.axvline(start, color="0.5", linestyle=":", linewidth=1)
    ax.axvline(end, color="0.5", linestyle=":", linewidth=1)
    ax.set_title(f"Recorded temperatures and predictions for a {GAP_HOURS}-hour outage\n"
                 f"Starting {start:%d %B %Y at %H:%M} | sensor 6012002000869")
    ax.set_xlabel("Date and time (Melbourne local time)")
    ax.set_ylabel("Temperature (°C)")
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18], tz=start.tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz=start.tz))
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(results / "example_outage_comparison.png", dpi=220)
    plt.close(fig)
    exported.to_csv(results / "example_outage_predictions.csv", index=False)
    scores = pd.DataFrame(rows)
    scores.to_csv(results / "example_outage_scores.csv", index=False)
    print(scores.to_string(index=False, float_format="%.4f"))
    print("\nCreated results/example_outage_comparison.png")
    print("Created results/example_outage_predictions.csv")
    print("Created results/example_outage_scores.csv")
    print("\nExample chosen by schedule, not by prediction error.")


if __name__ == "__main__":
    main()
