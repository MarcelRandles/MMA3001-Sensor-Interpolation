"""Reconstruct a gap using a local cubic Newton polynomial."""

import numpy as np


def local_newton(x_known, y_known, x_hidden):
    """Estimate hidden temperatures from four surrounding observations.

    Args:
        x_known: Increasing available measurement times, in elapsed hours.
        y_known: Available temperatures in degrees Celsius.
        x_hidden: Increasing times inside one missing-data gap.

    Returns:
        Estimated temperatures in degrees Celsius at x_hidden.

    Raises:
        ValueError: If inputs are invalid or fewer than two available
            observations exist on either side of the gap.

    The four selected points define a polynomial of degree at most three.
    Coefficients are calculated using Newton divided differences.
    """
    x_known = np.asarray(x_known, dtype=float)
    y_known = np.asarray(y_known, dtype=float)
    x_hidden = np.asarray(x_hidden, dtype=float)

    for values in [x_known, y_known, x_hidden]:
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("Inputs must be finite one-dimensional arrays.")

    if len(x_known) != len(y_known):
        raise ValueError("Known times and temperatures must have equal lengths.")

    if len(x_known) < 4 or len(x_hidden) == 0:
        raise ValueError("Need four known readings and at least one query.")

    if not np.all(np.diff(x_known) > 0):
        raise ValueError("Known times must be unique and increasing.")

    if not np.all(np.diff(x_hidden) > 0):
        raise ValueError("Hidden times must be unique and increasing.")

    # The query points must belong to one contiguous missing block.
    if np.any(
        (x_known >= x_hidden[0]) & (x_known <= x_hidden[-1])
    ):
        raise ValueError("Known readings cannot lie inside the query block.")

    before = np.flatnonzero(x_known < x_hidden[0])
    after = np.flatnonzero(x_known > x_hidden[-1])

    if len(before) < 2 or len(after) < 2:
        raise ValueError("Need two known readings on each side of the gap.")

    selected = np.concatenate([before[-2:], after[:2]])
    nodes = x_known[selected]
    values = y_known[selected]

    # Shift and scale time to improve numerical conditioning.
    # This changes coordinates without changing the interpolating polynomial.
    origin = nodes[0]
    scale = nodes[-1] - origin
    nodes = (nodes - origin) / scale
    queries = (x_hidden - origin) / scale

    # Build the divided-difference coefficients.
    coefficients = values.copy()

    for order in range(1, len(nodes)):
        coefficients[order:] = (
            coefficients[order:] - coefficients[order - 1:-1]
        ) / (nodes[order:] - nodes[:-order])

    # Evaluate Newton's nested polynomial.
    predicted = np.full_like(queries, coefficients[-1])

    for index in range(len(nodes) - 2, -1, -1):
        predicted = (
            coefficients[index]
            + (queries - nodes[index]) * predicted
        )

    if not np.isfinite(predicted).all():
        raise ValueError("Newton interpolation produced invalid predictions.")

    return predicted