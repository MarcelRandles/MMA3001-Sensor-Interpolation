"""Test Newton interpolation against independently known answers."""

import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_FOLDER / "src"))

from newton_interpolation import local_newton


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_reconstructs_polynomials_up_to_degree_three(degree):
    """Four nodes should reproduce any polynomial of degree at most three."""
    # Irregular timestamps, with two known points on each side.
    known_times = np.array([0.0, 0.7, 4.2, 5.0])
    hidden_times = np.array([1.0, 2.3, 3.8])

    known_values = 20.0 + 0.1 * known_times**degree
    expected = 20.0 + 0.1 * hidden_times**degree

    predicted = local_newton(
        known_times, known_values, hidden_times
    )

    np.testing.assert_allclose(
        predicted, expected, rtol=1e-10, atol=1e-10
    )


def test_uses_only_the_four_nearest_surrounding_points():
    """More distant temperatures must not affect this local method."""
    known_times = np.array([-5.0, 0.0, 1.0, 4.0, 5.0, 10.0])
    known_values = np.array([999.0, 20.0, 21.0, 24.0, 25.0, -999.0])
    hidden_times = np.array([2.0, 3.0])

    predicted = local_newton(
        known_times, known_values, hidden_times
    )

    np.testing.assert_allclose(predicted, [22.0, 23.0])


def test_rejects_insufficient_readings_before_gap():
    """Do not silently use extrapolation or a different point selection."""
    with pytest.raises(ValueError, match="two known readings"):
        local_newton(
            [0.0, 4.0, 5.0, 6.0],
            [20.0, 24.0, 25.0, 26.0],
            [1.0, 2.0],
        )


def test_rejects_duplicate_times():
    """Divided differences require distinct measurement times."""
    with pytest.raises(ValueError, match="unique and increasing"):
        local_newton(
            [0.0, 0.0, 4.0, 5.0],
            [20.0, 20.0, 24.0, 25.0],
            [1.0, 2.0],
        )


def test_rejects_nonfinite_temperature():
    """Invalid temperatures must not enter the calculation."""
    with pytest.raises(ValueError, match="finite"):
        local_newton(
            [0.0, 1.0, 4.0, 5.0],
            [20.0, np.nan, 24.0, 25.0],
            [2.0, 3.0],
        )