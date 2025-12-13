"""
This script contains 5 functions which can be used in the filtering/finding/tracking algorithm for the event camera
Completed for the APC524 Group Assignment: Documentation of a File
"""

import numpy as np
from scipy.spatial import KDTree


### Function 1: Shift Window
def accumulate_events(window, new_chunk, t_accum_us):
    """
    Call inside an EventsIterator loop to shift the accumulation window forward,
    appending the new event chunk and discarding the old one.

    Parameters
    ----------
    window : np.ndarray or None
        Numpy array of accumulated events from previous iteration of
        EventsIterator. If window is None, then the new chunk will initialize
        a new window.
    new_chunk : np.ndarray
        Numpy array of newly loaded chunk of events by EventsIterator.
    t_accum_us : int
        Timespan (in us) of the rolling accumulation window.

    Returns
    -------
    new_window : ndarray
        Updated window of accumulated events, including new_chunk and omitting
        an equally-sized chunk at the trailing end of the window.

    Notes
    -----
    RAW event files can be very bulky, so it's helpful for processing
    to load the event stream in chunks using EventsIterator. EventsIterator
    has an input, delta_t, which defines the discrete time window with
    which to load events. We will also want to have a rolling (accumulation)
    window larger than delta_t for doing all of our analysis
    (filtering/clustering/tracking). Once a new iteration is started, we
    will want to add our new chunk of events to the accumulation window of
    events, while also discarding the oldest delta_t chunk of events at the
    tail end of the window.

    Examples
    --------
    >>> from metavision_core.event_io import EventsIterator
    >>> import numpy as np

    >>> window = []
    >>> t_accum_us = 10000
    >>> delta_t = 1000  # us
    >>> raw_file = "Eventfile.raw"
    >>> mv_iterator = EventsIterator(raw_file, delta_t)

    >>> for ev_chunk in mv_iterator:
    >>>     window = accumulate_events(window, ev_chunk, t_accum_us)
    >>>     # ***Perform filtering, etc. on window***
    """

    # If the window hasn't been initialized yet, create the window
    if window is None or len(window) == 0:
        return new_chunk

    combined = np.concatenate([window, new_chunk])  # add new chunk to window
    cutoff_time = combined["t"][-1] - t_accum_us  # determine a cutoff for old events
    new_window = combined[
        combined["t"] >= cutoff_time
    ]  # throw out old events no longer in the window

    return new_window


# function 2: Isolated Noise Filter
def isolated_noise_filter(
    evs, spatial_radius=20, time_window=1000, min_neighbors=3
) -> np.ndarray:
    """
    Filter out events that do not have a minimum number of neighboring events
    within a specified spatial radius and time window.

    Parameters
    ----------
    evs : np.ndarray
        Numpy array with N-events, containing fields ['x', 'y', 't', 'p'].
    spatial_radius : float
        Pixel neighborhood radius to search for neighboring events.
    time_window : float
        Time window (in microseconds) to search for neighboring events.
    min_neighbors : int
        Minimum number of neighboring events required to keep an event.

    Returns
    -------
    filtered_evs : np.ndarray
        Filtered events containing only those with sufficient neighbors.

    Notes
    -----
    We rescale each spatial dimension for efficiency
    """

    points = np.stack(
        [
            evs["x"] / spatial_radius,
            evs["y"] / spatial_radius,
            evs["t"] / time_window,
        ],
        axis=1,
    )

    tree = KDTree(np.stack(points))

    n_neighbors = tree.query_ball_point(points, r=1.0, p=np.inf, return_length=True)
    mask = n_neighbors > min_neighbors

    filtered_evs = evs[mask]
    return filtered_evs


def low_pass_filter(window, min_dt, min_count):
    """
    Low-pass temporal noise filter

    This function removes "flickering" pixels that activate too frequently
    within the accumulation window.

    This filter should be called after
    1. Loading event data from EventsIterator
    2. Updating the accumulation window with 'accumulate_events'

    Parameters
    ----------
    window : np.ndarray
        Accumulated event window including fields ['t', 'x', 'y', 'p'].
        This outputs from 'accumulate_events'

    min_dt: float
        Minimum average inter-event duration in microseconds for a pixel to be valid.
        If the pixel flickers faster than this threshold, it is filtered out.
        For example, if min_dt = 300, it filters out the pixels firing faster than this threshold.

    min_count: int
        Minimum number of events at a pixel before flicker classification.
        The goal of this is to prevent filtering legitimate sparse motion.
        For example, if min_count = 5, then the pixel must fire at least 5x before
        flicker classification.

    Returns
    -------
    filtered_evs: np.ndarray
        The event window with noisy pixels removed.

    Examples
    --------
    >>> import numpy as np
    >>> #Take window = accumulate_events(window, new_chunk, t_accum_us) from function 1.
    >>> min_dt = 300
    >>> min_count = 5
    >>> window = low_pass_filter(window, min_dt, min_count)
    """

    if len(window) == 0:
        return window

    window = np.sort(window, order="t")
    pixel_id = window["x"].astype(np.int32) << 16 | window["y"].astype(np.int32)
    unique_pixel_id, inverse = np.unique(pixel_id, return_inverse=True)
    remove_pixels = np.zeros(len(unique_pixel_id), dtype=bool)

    for i, _p in enumerate(unique_pixel_id):
        idx = np.where(inverse == i)[0]

        if len(idx) < min_count:
            continue

        ts = window["t"][idx]
        dt = np.diff(ts)

        if np.mean(dt) < min_dt:
            remove_pixels[i] = True

    keep = ~remove_pixels[inverse]
    return window[keep]


def hot_pixel_filter(window, min_duration):
    """
    Hot pixel filter

    This function removes pixels that have a sustained on or off period
    over a minimum duration within the accumulation window.
    A pixel is a 'hot pixel' if the timestamps of consecutive events with
    the same polarity span at least min_duration.

    If a pixel remains on or off for the majority of the time, it is likely
    sensor noise rather than good data and should be filtered out.

    Parameters
    ----------
    window : np.ndarray
        Event window including fields ['t', 'x', 'y', 'p'].
        This outputs from 'accumulate_events'

    min_duration: float
        Minimum duration in microseconds of same-polarity events required to
        classify it as a hot pixel.

    Returns
    -------
    filtered_evs : np.ndarray
        The event window with hot pixels removed.

    Examples
    --------
    >>> import numpy as np
    >>> from scripts.filter_funcs import hot_pixel_filter
    >>> min_duration = 4000
    >>> window = hot_pixel_filter(window, min_duration)
    """

    if window is None or len(window) == 0:
        return window

    window = np.sort(window, order="t")

    pixel_ids = window["x"].astype(np.int32) << 16 | window["y"].astype(np.int32)

    unique_pixel_id, inverse = np.unique(pixel_ids, return_inverse=True)
    remove_mask = np.zeros(len(unique_pixel_id), dtype=bool)

    for i in range(len(unique_pixel_id)):
        idx = np.where(inverse == i)[0]

        ts1 = window["t"][idx]
        ps1 = window["p"][idx]

        change_points = np.where(np.diff(ps1) != 0)[0] + 1
        runs = np.split(ts1, change_points)

        for run in runs:
            if len(run) >= 2:
                duration = run[-1] - run[0]
                if duration >= min_duration:
                    remove_mask[i] = True
                    break

    mask = ~remove_mask[inverse]
    return window[mask]


### Function 5: Opposite Polarity Filter
def opposite_polarity_filter(evs, spatial_radius=20, time_scale=1):
    """
    Pass events that have at least one opposite polarity neighbor nearby in space and time,
    using a KD-tree for efficient search.


    Parameters
    ----------
    evs : np.ndarray
        Numpy array with N-events, containing fields ['x', 'y', 't', 'p'].
    spatial_radius : int
        Pixel neighborhood radius to search for opposite polarity events.
    time_scale : float
        Scaling factor for time coordinates in the distance calculations.

    Returns
    -------
    filtered_evs : np.ndarray
        Filtered events containing only those with opposite polarity neighbors.

    Notes
    -----
    The time_scale variable will likely have a large impact on the quality of
    the distance searches. This will be a function of the particle speed, so it
    won't be the same in every analysis. The filter could be processed with
    higher accuracy using cdist, but the kd-tree will help significantly with
    efficiency.
    """

    # sort events by polarity
    on_events = evs[evs["p"] == 1]
    off_events = evs[evs["p"] == -1]

    # break if no opposite polarity events are found
    if len(on_events) == 0 or len(off_events) == 0:
        print("Found no opposite-polarity events.")
        return np.empty(0, dtype=evs.dtype)

    # build KD-tree in (x, y, t) space with scaled time coordinate
    on_coords = np.stack(
        [on_events["x"], on_events["y"], on_events["t"] * time_scale], axis=1
    )
    off_coords = np.stack(
        [off_events["x"], off_events["y"], off_events["t"] * time_scale], axis=1
    )
    tree_on = KDTree(on_coords)
    tree_off = KDTree(off_coords)

    # search for ON events with nearby OFF events, p=2 for Euclidean distance
    on_indices = tree_off.query_ball_point(on_coords, r=spatial_radius, p=2)
    keep_on = np.array([len(neigh) > 0 for neigh in on_indices])

    # search for OFF events with nearby ON events
    off_indices = tree_on.query_ball_point(off_coords, r=spatial_radius, p=2)
    keep_off = np.array([len(neigh) > 0 for neigh in off_indices])

    # combine filtered ON and OFF events
    filtered_on = on_events[keep_on]
    filtered_off = off_events[keep_off]
    filtered_events = np.concatenate([filtered_on, filtered_off])

    # re-sort by timestamp
    filtered_events = np.sort(filtered_events, order="t")

    return filtered_events
