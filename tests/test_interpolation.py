"""Check error calculations and the experiment using known temperatures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Allow the tests to import scripts from the project's src folder.
PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER / "src"))

import run_experiments


def test_perfect_predictions_have_zero_error():
    """Predictions matching every measurement must have zero error."""
    actual = np.array([20.0, 21.0, 22.0])

    errors = run_experiments.calculate_errors(actual, actual.copy())

    assert errors["mae_c"] == pytest.approx(0)
    assert errors["rmse_c"] == pytest.approx(0)
    assert errors["max_absolute_error_c"] == pytest.approx(0)


def test_errors_match_hand_calculation():
    """Check against independently calculated expected errors."""
    actual = np.array([20.0, 20.0, 20.0])
    predicted = np.array([19.0, 22.0, 20.0])

    # Errors are -1, +2 and 0 degrees.
    # MAE = (1 + 2 + 0) / 3 = 1.
    # RMSE = sqrt((1 + 4 + 0) / 3) = sqrt(5/3).
    errors = run_experiments.calculate_errors(actual, predicted)

    assert errors["mae_c"] == pytest.approx(1.0)
    assert errors["rmse_c"] == pytest.approx(np.sqrt(5 / 3))
    assert errors["max_absolute_error_c"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "rise_per_hour",
    [0.0, 0.1],
    ids=["constant_temperature", "steadily_rising_temperature"],
)
def test_experiment_reconstructs_known_signals(
    tmp_path, monkeypatch, rise_per_hour
):
    """Run the actual experiment on temporary synthetic sensor data."""
    times = pd.date_range(
        "2024-03-01",
        "2024-03-08",
        freq="10min",
        tz="Australia/Melbourne",
    )

    elapsed_hours = (
        (times - times[0]).total_seconds().to_numpy() / 3600
    )

    # These signals have known answers throughout every hidden gap.
    temperatures = 20.0 + rise_per_hour * elapsed_hours

    synthetic_data = pd.DataFrame({
        "time_utc": times.tz_convert("UTC").astype(str),
        "timestamp_seconds": [
            time.timestamp() for time in times
        ],
        "temperature_c": temperatures,
    })

    input_path = tmp_path / "synthetic_temperature.csv"
    output_folder = tmp_path / "results"
    synthetic_data.to_csv(input_path, index=False)

    # Redirect this test to temporary files.
    # Your real data and results are not changed.
    monkeypatch.setattr(
        run_experiments, "INPUT_PATH", input_path
    )
    monkeypatch.setattr(
        run_experiments, "RESULTS_FOLDER", output_folder
    )

    run_experiments.main()

    detailed_path = output_folder / "gap_experiment_results.csv"
    summary_path = output_folder / "gap_experiment_summary.csv"
    graph_path = output_folder / "gap_experiment_errors.png"

    assert detailed_path.is_file()
    assert summary_path.is_file()
    assert graph_path.is_file()
    assert graph_path.stat().st_size > 0

    results = pd.read_csv(detailed_path)

    assert not results.empty
    assert set(results["method"]) == {
        "Linear",
        "Natural cubic spline",
        "PCHIP",
        "Local cubic Newton",
    }
    assert set(results["gap_hours"]) == set(range(1, 49))
    assert (results["hidden_readings"] > 0).all()

    # Every tested outage must have a result from all three methods.
    paired = results.groupby(["gap_start", "gap_hours"])
    assert (paired["method"].nunique() == 4).all()
    assert (paired.size() == 4).all()

    # Tiny rounding differences are normal in numerical calculations.
    for metric in [
        "mae_c", "rmse_c", "max_absolute_error_c"
    ]:
        np.testing.assert_allclose(
            results[metric],
            0.0,
            atol=1e-9,
            rtol=0,
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_cleaning_rejects_nonfinite_temperatures(bad_value):
    """Invalid temperatures must not silently enter the analysis."""
    from inspect_data import clean_temperature
    data = pd.DataFrame({
        "timestamp_seconds": [1709251200, 1709251800],
        "temperature_c": [20.0, bad_value],
    })
    with pytest.raises(ValueError):
        clean_temperature(data)


def test_cleaning_rejects_conflicting_duplicates():
    """Two different temperatures at one timestamp need investigation."""
    from inspect_data import clean_temperature
    data = pd.DataFrame({
        "timestamp_seconds": [1709251200, 1709251200],
        "temperature_c": [20.0, 21.0],
    })
    with pytest.raises(ValueError, match="Different temperatures"):
        clean_temperature(data)


def test_cleaning_removes_identical_duplicates_and_sorts():
    """Check the cleaning behaviour against independently specified rows."""
    from inspect_data import clean_temperature
    data = pd.DataFrame({
        "timestamp_seconds": [1709251800, 1709251200, 1709251200],
        "temperature_c": [21.0, 20.0, 20.0],
    })
    result = clean_temperature(data)
    assert result["timestamp_seconds"].tolist() == [1709251200, 1709251800]
    assert result["temperature_c"].tolist() == [20.0, 21.0]
