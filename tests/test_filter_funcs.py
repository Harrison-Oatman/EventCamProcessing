# author of first function: Harrison
# author of other functions: Joanna Van Liew
import eventcamprocessing


# Harrison's test of the isolated noise filter
def test_isolated_noise_filter():
    import numpy as np

    from eventcamprocessing.filter_funcs import isolated_noise_filter

    dtype = np.dtype([("x", "i4"), ("y", "i4"), ("t", "i8"), ("p", "i1")])
    evs = np.array(
        [
            (10, 10, 1000, 1),
            (10, 11, 1000, 1),
            (10, 10, 1001, 1),
            (50, 50, 2000, -1),  # isolated event
            (11, 10, 1002, 1),
        ],
        dtype=dtype,
    )

    filtered_evs = isolated_noise_filter(
        evs, spatial_radius=5, time_window=10, min_neighbors=2
    )

    assert len(filtered_evs) == 4
    assert all(
        (50, 50, 2000, -1) != (ev["x"], ev["y"], ev["t"], ev["p"])
        for ev in filtered_evs
    )


# Joanna's tests of the low-pass, hot-pixel, and opposite-polarity filter functions
import numpy as np
from conftest import array_events

from eventcamprocessing.filter_funcs import (
    hot_pixel_filter,
    low_pass_filter,
    opposite_polarity_filter,
)


def test_low_pass_filter_flickering():
    """
    Test that the low pass filter is applied correctly and removes rapidly flickering  'high energy'
    pixels.

    This test creates two arbitrary sets of events, one with closely spaced timestamps and one with wide timestamps.
    This test verifies that the events from the high-frequency flicker pixel are removed, only keeping
    the low-frequency pixels.
    """
    fast = [(5, 5, t, 1) for t in range(0, 1000, 50)]
    slow = [(6, 6, t, 1) for t in range(0, 2000, 500)]

    arr = array_events(fast + slow)
    out = low_pass_filter(arr, min_dt=300, min_count=5)

    assert not any((out["x"] == 5) & (out["y"] == 5))
    assert any((out["x"] == 6) & (out["y"] == 6))


def test_hot_pixel_filter_detection():
    """
    Test that hot pixels (above a certain time threshold) are removed, while normal pixels with a regular duration
    are kept.

    This test starts with two arbitrary pixels with repeated events where the first is active for a long time,
    while the second is active for a short time. After applying the hot pixel filter, it verifies that the events
    from the hot pixel are removed.
    """
    events = []
    events += [(5, 5, t, 1) for t in range(0, 10000, 1000)]
    events += [(6, 6, t, 1) for t in range(0, 4000, 1000)]

    arr = array_events(events)
    out = hot_pixel_filter(arr, min_duration=8000)

    assert not any((out["x"] == 5) & (out["y"] == 5))
    assert any((out["x"] == 6) & (out["y"] == 6))


def test_filter_opposite_polarity_keeps_pairs():
    """
    Test that opposite polarity events are kept while isolated events are removed.

    This tests starts with a nearby ON/OFF event pair (arbitrary) that is close in both space and time,
    and a distant event with no opposite polarity match. The opposite polarity filter is then tested and
    this test verifies that the valid pair of ON/OFF events are kept while the isolated one is removed.
    """
    on = (10, 10, 1000, 1)
    off = (12, 11, 1005, -1)
    far = (200, 200, 1000, -1)

    arr = array_events([on, off, far])
    out = opposite_polarity_filter(arr, spatial_radius=10, time_scale=1e-3)

    assert any((out["x"] == 10) & (out["y"] == 10))
    assert any((out["x"] == 12) & (out["y"] == 11))
    assert not any((out["x"] == 200) & (out["y"] == 200))
