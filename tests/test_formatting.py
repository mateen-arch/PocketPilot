"""
Unit tests for formatting helpers in pocketpilot.inference.formatting.
"""

import numpy as np
import pytest

from pocketpilot.inference.formatting import format_clock, format_hours, format_probability


def test_format_clock_valid():
    """Verify format_clock on standard, fractional, next-day, and zero hours."""
    assert format_clock(16.0) == "16:00"
    assert format_clock(22.5) == "22:30"
    assert format_clock(25.5) == "01:30 (+1 day)"
    assert format_clock(0.0) == "00:00"
    assert format_clock(49.25) == "01:15 (+2 days)"


def test_format_clock_invalid():
    """Verify format_clock raises ValueError for negative or non-finite inputs."""
    with pytest.raises(ValueError):
        format_clock(-1.0)
    with pytest.raises(ValueError):
        format_clock(float("nan"))
    with pytest.raises(ValueError):
        format_clock(float("inf"))
    with pytest.raises(ValueError):
        format_clock("not_a_number")


def test_format_probability_valid():
    """Verify format_probability on edge bounds and rounded whole numbers."""
    assert format_probability(0.005) == "under 1%"
    assert format_probability(0.0) == "under 1%"
    assert format_probability(0.995) == "over 99%"
    assert format_probability(1.0) == "over 99%"
    assert format_probability(0.146) == "15%"
    assert format_probability(0.50) == "50%"
    assert format_probability(0.01) == "1%"
    assert format_probability(0.99) == "99%"


def test_format_probability_invalid():
    """Verify format_probability raises ValueError for out-of-range or non-finite inputs."""
    with pytest.raises(ValueError):
        format_probability(-0.01)
    with pytest.raises(ValueError):
        format_probability(1.01)
    with pytest.raises(ValueError):
        format_probability(float("nan"))
    with pytest.raises(ValueError):
        format_probability(float("inf"))


def test_format_hours_valid():
    """Verify format_hours correctly formats 1 decimal place."""
    assert format_hours(9.65) == "9.7 h"
    assert format_hours(0.0) == "0.0 h"
    assert format_hours(12.34) == "12.3 h"


def test_format_hours_invalid():
    """Verify format_hours raises ValueError for negative or non-finite inputs."""
    with pytest.raises(ValueError):
        format_hours(-0.5)
    with pytest.raises(ValueError):
        format_hours(float("nan"))
    with pytest.raises(ValueError):
        format_hours(float("inf"))
