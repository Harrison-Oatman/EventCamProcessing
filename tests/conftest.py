#author = Joanna Van Liew
#helper functions to be used in tests

import numpy as np
import pytest

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