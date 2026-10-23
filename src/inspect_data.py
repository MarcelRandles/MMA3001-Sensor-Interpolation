"""Inspect temperature measurements and prepare a clean time series."""

import csv
import io
import json
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Find the project folder from this script's location.
PROJECT_FOLDER = Path(__file__).resolve().parents[1]

ZIP_PATH = (
    PROJECT_FOLDER
    / "data"
    / "raw"
    / "5EnvSensor_MayToDec2024_180kRows.zip"
)

SENSOR_ID = "6012002000869"


def read_temperature(zip_path, sensor_id):
    """Read one sensor's temperature measurements directly from a ZIP.

    Args:
        zip_path: Path to the ZIP containing one CSV file.
        sensor_id: Identifier of the environmental sensor.

    Returns:
        DataFrame containing measurement timestamps and temperatures.

    Raises:
        FileNotFoundError: If the ZIP does not exist.
        ValueError: If the CSV selection or measurements are invalid.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"Cannot find the data ZIP: {zip_path}")

    measurements = []

    with zipfile.ZipFile(zip_path) as archive:
        csv_files = [
            name for name in archive.namelist()
            if name.lower().endswith(".csv") and not name.startswith("__MACOSX/")
        ]

        if len(csv_files) != 1:
            raise ValueError("Expected exactly one CSV file inside the ZIP.")

        with archive.open(csv_files[0]) as csv_file:
            text_file = io.TextIOWrapper(csv_file, encoding="utf-8-sig")
            reader = csv.DictReader(text_file)

            for row in reader:
                if row["sensorid"] != sensor_id:
                    continue

                # Each CSV row contains a JSON list of sensor variables.
                variables = json.loads(row["jsondata"])

                for item in variables:
                    if item["variable"]["name"] != "Temperature":
                        continue

                    if item["variable"]["unit"] != "°C":
                        raise ValueError("Expected temperature units of °C.")

                    for reading in item["values"]:
                        measurements.append({
                            "timestamp_seconds": reading["time"],
                            "temperature_c": reading["value"],
                        })

    if not measurements:
        raise ValueError(f"No temperature readings found for {sensor_id}.")

    return pd.DataFrame(measurements)


def clean_temperature(data):
    """Validate readings, remove identical duplicates and sort by time.

    Args:
        data: DataFrame returned by read_temperature.

    Returns:
        DataFrame with unique timestamps, temperatures and UTC/local dates.

    Raises:
        ValueError: If readings are missing, nonnumeric, nonfinite,
            or disagree at the same timestamp.
    """
    data = data.copy()

    for column in ["timestamp_seconds", "temperature_c"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("Missing or nonfinite readings require investigation.")

    # Never silently choose between conflicting measurements.
    values_per_timestamp = data.groupby(
        "timestamp_seconds"
    )["temperature_c"].nunique()

    if (values_per_timestamp > 1).any():
        raise ValueError(
            "Different temperatures share a timestamp. Investigate before cleaning."
        )

    clean = (
        data.drop_duplicates(subset="timestamp_seconds")
        .sort_values("timestamp_seconds")
        .reset_index(drop=True)
    )

    # Use the measurement time, rather than the database processing date.
    clean["time_utc"] = pd.to_datetime(
        clean["timestamp_seconds"], unit="s", utc=True
    )
    clean["time_local"] = clean["time_utc"].dt.tz_convert(
        "Australia/Melbourne"
    )

    return clean


def save_temperature_plot(data, output_path):
    """Save full-period and example-week plots without changing readings.

    Args:
        data: Clean temperature DataFrame.
        output_path: Destination PNG path.

    Returns:
        None.
    """
    first_day = data["time_local"].iloc[0].normalize()
    week_start = first_day + pd.Timedelta(days=1)
    week_end = week_start + pd.Timedelta(days=7)

    week = data[
        (data["time_local"] >= week_start)
        & (data["time_local"] < week_end)
    ]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))

    for ax, subset, title in [
        (axes[0], data, "Full recorded period"),
        (axes[1], week, "First seven-day window after the initial day"),
    ]:
        ax.plot(
            subset["time_local"],
            subset["temperature_c"],
            linewidth=0.8,
        )
        ax.set_title(title)
        ax.set_ylabel("Temperature (°C)")
        ax.set_xlabel("Melbourne local time")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(f"Recorded temperature — sensor {SENSOR_ID}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    """Read, audit and export the selected sensor's temperature data."""
    processed_folder = PROJECT_FOLDER / "data" / "processed"
    results_folder = PROJECT_FOLDER / "results"

    processed_folder.mkdir(parents=True, exist_ok=True)
    results_folder.mkdir(parents=True, exist_ok=True)

    print("Reading the environmental ZIP. Please wait...")
    original = read_temperature(ZIP_PATH, SENSOR_ID)
    clean = clean_temperature(original)

    intervals = clean["timestamp_seconds"].diff().dropna()

    summary = {
        "sensor_id": SENSOR_ID,
        "original_readings": len(original),
        "unique_readings": len(clean),
        "identical_duplicates_removed": len(original) - len(clean),
        "start_local": str(clean["time_local"].iloc[0]),
        "end_local": str(clean["time_local"].iloc[-1]),
        "minimum_temperature_c": float(clean["temperature_c"].min()),
        "maximum_temperature_c": float(clean["temperature_c"].max()),
        "median_interval_minutes": float(intervals.median() / 60),
        "largest_interval_minutes": float(intervals.max() / 60),
        "intervals_over_15_minutes": int((intervals > 900).sum()),
    }

    csv_path = processed_folder / "temperature_clean.csv"
    summary_path = results_folder / "data_summary.json"
    plot_path = results_folder / "temperature_overview.png"

    clean.to_csv(csv_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    save_temperature_plot(clean, plot_path)

    print("\nDATA CHECK")
    for name, value in summary.items():
        print(f"{name}: {value}")

    print("\nCreated:")
    for path in [csv_path, summary_path, plot_path]:
        print(path.relative_to(PROJECT_FOLDER))


if __name__ == "__main__":
    main()