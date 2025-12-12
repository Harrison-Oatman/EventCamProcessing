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
