import numpy as np
import pandas as pd


def numpify_df(events_df: pd.DataFrame):
    """
    Convert a pandas DataFrame of events to a structured NumPy array.
    """
    dtype = np.dtype([("x", "i4"), ("y", "i4"), ("t", "i8"), ("p", "i1")])

    events = np.zeros(len(events_df), dtype=dtype)
    events["x"] = events_df["x"].to_numpy()
    events["y"] = events_df["y"].to_numpy()
    events["t"] = events_df["t"].to_numpy()
    events["p"] = events_df["p"].to_numpy()
    return events


def basic_event_iterator(events: np.ndarray, t_step: float, t_window: float):
    """
    Basic event iterator that yields events in time windows.
    Moves the center of the window by t_step each iteration.
    Collects events within t_window/2 before and after the center time.
    """

    start_time = events["t"].min()
    end_time = events["t"].max()

    center_times = np.arange(start_time, end_time, t_step)
    for center_time in center_times:
        window_start = center_time - t_window / 2
        window_end = center_time + t_window / 2

        mask = (events["t"] >= window_start) & (events["t"] < window_end)
        yield events[mask]


def collapse_2d(events: np.ndarray, shape: tuple[int, int]):
    """
    Collapse events into a 2D histogram based on x and y coordinates.
    """
    img = np.zeros(shape, dtype=np.int32)
    for event in events:
        img[event["y"], event["x"]] += 1
    return img

def collapse_2d_polarity(events: np.ndarray, shape: tuple[int, int]):
    """
    Collapse events into two 2D histograms based on x and y coordinates and polarity.
    Returns a tuple of (positive_polarity_image, negative_polarity_image).
    """
    img_pos = np.zeros(shape, dtype=np.int32)
    img_neg = np.zeros(shape, dtype=np.int32)
    for event in events:
        if event["p"] > 0:
            img_pos[event["y"], event["x"]] += 1
        else:
            img_neg[event["y"], event["x"]] += 1
    return img_pos, img_neg

"""
Taken directly from conftest.py in tests folder
"""

# author = Joanna Van Liew
# helper functions to be used in tests

event_dtype = np.dtype([("x", "i4"), ("y", "i4"), ("t", "i8"), ("p", "i1")])


def make_event(x, y, t, p=1):
    a = np.zeros(1, dtype=event_dtype)
    a["x"][0] = x
    a["y"][0] = y
    a["t"][0] = t
    a["p"][0] = p
    return a


def array_events(list_of_events):
    arr = np.zeros(len(list_of_events), dtype=event_dtype)
    for i, (x, y, t, p) in enumerate(list_of_events):
        arr["x"][i] = x
        arr["y"][i] = y
        arr["t"][i] = t
        arr["p"][i] = p
    return arr


def event():
    return event_dtype

