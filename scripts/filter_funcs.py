'''
This script contains 5 functions which can be used in the filtering/finding/tracking algorithm for the event camera
Completed for the APC524 Group Assignment: Documentation of a File
'''

### Function 1: Shift Window
def accumulate_events(window, new_chunk, t_accum_us):
    """
    Call inside an EventsIterator loop to shift the accumulation window forward,
    appending the new event chunk and discarding the old one.

    RAW event files can be very bulky, so it's helpful for processing 
    to load the event stream in chunks using EventsIterator. EventsIterator 
    has an input, delta_t, which defines the discrete time window with
    which to load events. We will also want to have a rolling (accumulation) 
    window larger than delta_t for doing all of our analysis 
    (filtering/clustering/tracking). Once a new iteration is started, we 
    will want to add our new chunk of events to the accumulation window of 
    events, while also discarding the oldest delta_t chunk of events at the 
    tail end of the window.

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

    Example Usage
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
    cutoff_time = combined['t'][-1] - t_accum_us  # determine a cutoff for old events
    new_window = combined[combined['t'] >= cutoff_time]  # throw out old events no longer in the window

    return new_window

### Function 2: Isolated Space-Time Noise Filter
# This should be a simple filter that checks each event in the window and determines if it
# has significant neighbors in space and time. The inputs would likely be some spatial
# search radius, some temporal search window, and some minimum number of neighbors. If the
# function doesn't find enough events in the radius or window, the event should be marked
# to be filtered out as noise.

### Function 3: Low-Pass Filter
# This should be a simple filter that determines if a pixel is being activated at too high
# of a frequency to be reliable for analysis. The inputs would likely be some minimum
# flicker frequency (or 1/freq = timestep) and some minimum flicker duration. If the
# function finds that a pixel meets both of these minimum criteria, the pixel should be
# marked to be filtered out as noise.

### Function 4: Hot-Pixel Filter
# This should be fairly similar to the Low-Pass Filter, but deals instead with pixels
# demonstrating a sustained on/off period. The only input would likely be some minimum
# duration for a sustained on or off event. If the function finds that a pixel sustains
# either an on or off state for the minimum duration, the pixel should be marked to be
# filtered out as noise.