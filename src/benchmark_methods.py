"""Measure interpolation construction and prediction time on one fixed gap."""

from pathlib import Path
import platform
from timeit import repeat

import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import CubicSpline, PchipInterpolator

from newton_interpolation import local_newton

PROJECT_FOLDER = Path(__file__).resolve().parents[1]


def main():
    """Benchmark all methods using identical available readings and queries."""
    data = pd.read_csv(
        PROJECT_FOLDER / "data/processed/temperature_clean.csv"
    )
    data["time"] = pd.to_datetime(
        data["time_utc"], utc=True
    ).dt.tz_convert("Australia/Melbourne")

    data = data.sort_values("timestamp_seconds")

    # A fixed eight-hour gap from the main experiment.
    gap_start = pd.Timestamp(
        "2024-04-24 06:00", tz="Australia/Melbourne"
    )
    gap_end = gap_start + pd.Timedelta(hours=8)

    window = data.loc[
        (data["time"] >= gap_start - pd.Timedelta(hours=6))
        & (data["time"] <= gap_end + pd.Timedelta(hours=6))
    ]

    hidden = (
        (window["time"] >= gap_start)
        & (window["time"] < gap_end)
    ).to_numpy()

    seconds = window["timestamp_seconds"].to_numpy(dtype=float)
    x = (seconds - seconds[0]) / 3600
    y = window["temperature_c"].to_numpy(dtype=float)

    x_known = x[~hidden]
    y_known = y[~hidden]
    x_hidden = x[hidden]

    if len(x_known) < 4 or len(x_hidden) == 0:
        raise ValueError("Not enough readings for the benchmark.")

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Benchmark data contains invalid values.")

    if not np.all(np.diff(x) > 0):
        raise ValueError("Timestamps must be unique and increasing.")

    if (
        x_hidden.min() <= x_known.min()
        or x_hidden.max() >= x_known.max()
    ):
        raise ValueError("Known readings are needed on both sides.")

    # Each call includes constructing the interpolant where needed
    # and predicting all hidden readings.
    methods = {
        "Linear": lambda: np.interp(
            x_hidden, x_known, y_known
        ),
        "Natural cubic spline": lambda: CubicSpline(
            x_known, y_known,
            bc_type="natural", extrapolate=False,
        )(x_hidden),
        "PCHIP": lambda: PchipInterpolator(
            x_known, y_known, extrapolate=False
        )(x_hidden),
        "Local cubic Newton": lambda: local_newton(
            x_known, y_known, x_hidden
        ),
    }

    calls_per_batch = 200
    batches = 7
    rows = []

    for name, operation in methods.items():
        # Warm up before measuring.
        predicted = operation()
        if not np.isfinite(predicted).all():
            raise ValueError(f"Invalid predictions from {name}.")

        batch_seconds = np.array(
            repeat(
                operation,
                number=calls_per_batch,
                repeat=batches,
            )
        )

        milliseconds = batch_seconds / calls_per_batch * 1000

        rows.append({
            "method": name,
            "known_readings": len(x_known),
            "predicted_readings": len(x_hidden),
            "calls_per_batch": calls_per_batch,
            "batches": batches,
            "median_ms_per_outage": float(np.median(milliseconds)),
            "minimum_ms_per_outage": float(milliseconds.min()),
            "maximum_ms_per_outage": float(milliseconds.max()),
        })

    results = pd.DataFrame(rows)
    output_folder = PROJECT_FOLDER / "results"
    output_folder.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        output_folder / "method_runtime.csv", index=False
    )

    environment = (
        f"Platform: {platform.platform()}\n"
        f"Processor: {platform.processor()}\n"
        f"Python: {platform.python_version()}\n"
        f"NumPy: {np.__version__}\n"
        f"SciPy: {scipy.__version__}\n"
        f"pandas: {pd.__version__}\n"
        "Timing includes interpolation construction and prediction.\n"
        "CSV loading, gap selection, scoring and plotting are excluded.\n"
        "Results describe this fixed case on this computer.\n"
    )

    (output_folder / "benchmark_environment.txt").write_text(
        environment, encoding="utf-8"
    )

    print(results.to_string(index=False, float_format="%.6f"))
    print("\nTimes are milliseconds per reconstructed outage.")
    print("Created results/method_runtime.csv")
    print("Created results/benchmark_environment.txt")


if __name__ == "__main__":
    main()