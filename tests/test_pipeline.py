# author = Joanna Van Liew

import sys
import types

# Create a fake metavision_core package to run tests without having to download the metavision_core package on your own system"""
metavision_core = types.ModuleType("metavision_core")
event_io = types.ModuleType("metavision_core.event_io")

# Fake EventsIterator inside event_io
event_io.EventsIterator = object

# Wire package structure together
metavision_core.event_io = event_io

# Register in sys.modules
sys.modules["metavision_core"] = metavision_core
sys.modules["metavision_core.event_io"] = event_io

import numpy as np
from conftest import array_events
from skimage.measure import label, regionprops

from eventcamprocessing import ev_particlefinder
from eventcamprocessing.filter_funcs import (
    accumulate_events,
    isolated_noise_filter,
    opposite_polarity_filter,
)


def test_simple_pipeline():
    """
    Test that an event processing pipeline works as expected without error.

    It constructs two small arbitrary chunks of event data and uses the accumulate_event function
    to combine them in a time window. It then applies two of the filter, the noise and polarity filters
    before running it through the particle finder function. The main outcome of this test is assurance that
    the full pipeline can run together and return an array-like output.
    """
    chunk1 = array_events([(10, 10, 1000, 1), (11, 10, 1001, 1), (50, 50, 1002, -1)])
    chunk2 = array_events([(10, 11, 2000, 1), (11, 11, 2001, 1)])

    window = accumulate_events(None, chunk1, t_accum_us=5000)
    window = accumulate_events(window, chunk2, t_accum_us=5000)

    window = isolated_noise_filter(
        window, spatial_radius=2, time_window=1000, min_neighbors=1
    )
    window = opposite_polarity_filter(window, spatial_radius=5, time_scale=1)

    parts = ev_particlefinder(window, min_area=2, h=128, w=128)

    assert isinstance(parts, np.ndarray)
