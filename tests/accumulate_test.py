#author = Joanna Van Liew
#tests for the accumulate_event function

import numpy as np
from eventcamprocessing.filter_funcs import accumulate_events
from conftest import array_events

def test_accumulate_events_initialization():
    """
    Test that the accumulation of events is initialized correctly when no window exists.

    This test starts with an arbitrary chunk of events that are passed into the accumulator with
    'window=None' and verifies that the output window contains the events from the new chunk.
    """
    chunk = array_events([(10,10,1000,1), (11,10,1500,1)])
    out = accumulate_events(window=None, new_chunk = chunk, t_accum_us=2000)
    assert len(out) == 2
    assert np.all(out["t"] == chunk["t"])

def test_accumulate_events():
    """
    Test that event accumulation drops events outside the accumulation window.

    This test starts with an accumulation window with an existing chunk of events then adds a new chunk
    of event data. This test shows that the older events are removed and the new events within a time window
    remain.
    """
    chunk1 = array_events([(1,1,0,1), (2,2,500,1)])
    chunk2 = array_events([(3, 3, 10000, 1)])
    window = accumulate_events(window=chunk1, new_chunk=chunk2, t_accum_us=2000)
    assert len(window) == 1
    assert window["x"][0] ==3