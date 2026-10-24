"""Plot the outage containing the largest cubic-spline prediction error."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, PchipInterpolator, BarycentricInterpolator
from newton_interpolation import local_newton

from run_experiments import calculate_errors


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
RESULTS_FOLDER = PROJECT_FOLDER / "results"


def plot_case(scores, data, output_folder, target_method):
    """Plot all methods on the selected method's largest saved error case."""
    spline_scores = scores.loc[scores["method"] == target_method]
    if spline_scores.empty:
        raise ValueError(f"Missing results for {target_method}; rerun experiments.")
    # Select by largest individual absolute error, not average error.
    worst = spline_scores.loc[
        spline_scores["max_absolute_error_c"].idxmax()
    ]

    gap_start = pd.Timestamp(worst["gap_start"]).tz_convert(
        "Australia/Melbourne"
    )
    duration = float(worst["gap_hours"])
    gap_end = gap_start + pd.Timedelta(hours=duration)

    # Use the same six hours of context on either side as before.
    window = data.loc[
        (data["time"] >= gap_start - pd.Timedelta(hours=6))
        & (data["time"] <= gap_end + pd.Timedelta(hours=6))
    ].copy()

    hidden = (
        (window["time"] >= gap_start)
        & (window["time"] < gap_end)
    ).to_numpy()
    known = ~hidden

    seconds = window["timestamp_seconds"].to_numpy(dtype=float)
    x = (seconds - seconds[0]) / 3600
    y = window["temperature_c"].to_numpy(dtype=float)

    if hidden.sum() == 0 or known.sum() < 4:
        raise ValueError("Insufficient readings to reproduce this outage.")

    if (
        x[hidden].min() <= x[known].min()
        or x[hidden].max() >= x[known].max()
    ):
        raise ValueError("Readings are needed on both sides of the gap.")

    predictions = {
        "Linear": np.interp(x[hidden], x[known], y[known]),
        "Natural cubic spline": CubicSpline(
            x[known],
            y[known],
            bc_type="natural",
            extrapolate=False,
        )(x[hidden]),
        "PCHIP": PchipInterpolator(
            x[known], y[known], extrapolate=False
        )(x[hidden]),
    }

    predictions["Local cubic Newton"] = local_newton(
        x[known], y[known], x[hidden]
    )

    # Independently evaluate the same four-point polynomial in barycentric form.
    # Agreement distinguishes polynomial behaviour from a divided-difference bug.
    before = np.flatnonzero(x[known] < x[hidden].min())[-2:]
    after = np.flatnonzero(x[known] > x[hidden].max())[:2]
    selected = np.concatenate([before, after])
    nodes = x[known][selected]
    values = y[known][selected]
    scaled_nodes = (nodes - nodes[0]) / (nodes[-1] - nodes[0])
    scaled_queries = (x[hidden] - nodes[0]) / (nodes[-1] - nodes[0])
    independent = BarycentricInterpolator(scaled_nodes, values)(scaled_queries)
    np.testing.assert_allclose(
        predictions["Local cubic Newton"], independent,
        rtol=1e-7, atol=1e-7,
    )
    slug = "newton" if target_method == "Local cubic Newton" else "spline"
    output_folder.mkdir(parents=True, exist_ok=True)
    score_rows = []

    for method, predicted in predictions.items():
        errors = calculate_errors(y[hidden], predicted)

        # Check that this script reproduces the saved experiment.
        saved = scores.loc[
            (scores["gap_start"] == worst["gap_start"])
            & (scores["gap_hours"] == duration)
            & (scores["method"] == method)
        ]

        if len(saved) != 1:
            raise ValueError("Expected exactly one matching saved result.")

        for metric, value in errors.items():
            if not np.isclose(
                value, float(saved.iloc[0][metric]),
                rtol=1e-7, atol=1e-9,
            ):
                raise ValueError(
                    f"{method}: {metric} differs from the saved experiment."
                )

        score_rows.append({"method": method, **errors})

    comparison = pd.DataFrame({
        "time_utc": window.loc[hidden, "time_utc"].to_numpy(),
        "actual_temperature_c": y[hidden],
        "linear_prediction_c": predictions["Linear"],
        "spline_prediction_c": predictions["Natural cubic spline"],
        "pchip_prediction_c": predictions["PCHIP"],
        "newton_prediction_c": predictions["Local cubic Newton"],
    })
    comparison.to_csv(
        output_folder / f"worst_{slug}_predictions.csv",
        index=False,
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True
    )

    axes[0].plot(
        window["time"], y,
        color="grey", linewidth=1,
        label="Recorded temperature (reference)",
    )
    axes[0].scatter(
        window.loc[known, "time"], y[known],
        color="black", s=12,
        label="Readings available to methods",
    )
    axes[0].scatter(
        window.loc[hidden, "time"], y[hidden],
        color="grey", marker="x", s=22,
        label="Hidden readings",
    )

    for method, predicted in predictions.items():
        axes[0].plot(
            window.loc[hidden, "time"], predicted,
            marker="o", markersize=3, label=method,
        )
        axes[1].plot(
            window.loc[hidden, "time"],
            predicted - y[hidden],
            marker="o", markersize=3, label=method,
        )

    for axis in axes:
        axis.axvspan(
            gap_start, gap_end,
            color="orange", alpha=0.12,
        )
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title(
        f"Largest {target_method} error: {output_folder.parent.name}\n"
        f"{gap_start:%d %B %Y, %H:%M} — {duration:g}-hour outage"
    )

    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("Prediction − recorded value (°C)")
    axes[1].set_xlabel("Melbourne local time")
    axes[1].xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%d %b\n%H:%M",
            tz=window["time"].dt.tz,
        )
    )

    fig.tight_layout()
    fig.savefig(
        output_folder / f"worst_{slug}_comparison.png",
        dpi=200,
    )
    plt.close(fig)

    print("\nSelected outage:", gap_start)
    print("Duration:", duration, "hours")
    print("\nResults match the original experiment.\n")
    print(
        pd.DataFrame(score_rows).to_string(
            index=False, float_format="%.4f"
        )
    )
    print("\nCreated:")
    print(output_folder / f"worst_{slug}_comparison.png")
    print(output_folder / f"worst_{slug}_predictions.csv")
    print("Newton agrees with an independent barycentric polynomial calculation.")
    print("Recorded gap range:", float(y[hidden].min()), float(y[hidden].max()))
    for method, predicted in predictions.items():
        print(method, "prediction range:", float(predicted.min()), float(predicted.max()))
    print("\nThis is a deliberately selected failure example.")
    print("Use the full experiment to judge average performance.")


def main():
    """Diagnose spline and Newton extremes in both observation periods."""
    data = pd.read_csv(PROJECT_FOLDER / "data/processed/temperature_clean.csv")
    data["time"] = pd.to_datetime(data["time_utc"], utc=True).dt.tz_convert(
        "Australia/Melbourne"
    )
    data = data.sort_values("timestamp_seconds")
    for folder in [RESULTS_FOLDER, RESULTS_FOLDER / "validation"]:
        scores = pd.read_csv(folder / "gap_experiment_results.csv")
        for method in ["Natural cubic spline", "Local cubic Newton"]:
            plot_case(scores, data, folder / "diagnostics", method)


if __name__ == "__main__":
    main()
