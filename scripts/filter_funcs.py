'''
This script contains 5 functions which can be used in the filtering/finding/tracking algorithm for the event camera
Completed for the APC524 Group Assignment: Documentation of a File
'''

import numpy as np


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
    window[keep] : np.ndarray
        The event window with noisy pixels removed.

    Example Usage
    -------------
    >>> import numpy as np
    >>> #Take window = accumulate_events(window, new_chunk, t_accum_us) from function 1.
    >>> min_dt = 300
    >>> min_count = 5
    >>> window = low_pass_filter(window, min_dt, min_count)
    """

    if len(window) == 0:
        return window

    window = np.sort(window, order = 't')
    pixel_id = window['x'].astype(np.int32) << 16 | window['y'].astype(np.int32)
    unique_pixel_id, inverse = np.unique(pixel_id, return_inverse=True)
    remove_pixels = (np.zeros(len(unique_pixel_id), dtype=bool))

    for i, p in enumerate(unique_pixel_id):
        idx = np.where(inverse == i)[0]

        if len(idx) < min_count:
            continue

        ts = window['t'][idx]
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
    window[mask] : np.ndarray
        The event window with hot pixels removed.

    Example Usage
    -------------
    >>> import numpy as np
    >>> from scripts.filter_funcs import hot_pixel_filter
    >>> min_duration = 4000
    >>> window = hot_pixel_filter(window, min_duration)
    """

    if window is None or len(window) == 0:
        return window

    window = np.sort(window, order = 't')

    pixel_ids = window['x'].astype(np.int32) << 16 | window['y'].astype(np.int32)

    unique_pixel_id, inverse = np.unique(pixel_ids, return_inverse=True)
    remove_mask = (np.zeros(len(unique_pixel_id), dtype=bool))

    for i in range(len(unique_pixel_id)):
        idx = np.where(inverse == i)[0]

        ts1 = window['t'][idx]
        ps1 = window['p'][idx]

        change_points = np.where(np.diff(ps1) != 0)[0]+1
        runs = np.split(ts1, change_points)

        for run in runs:
            if len(run) >=2:
                duration = run[-1]-run[0]
                if duration >= min_duration:
                    remove_mask[i] = True
                    break

    mask = ~remove_mask[inverse]
    return window[mask]
