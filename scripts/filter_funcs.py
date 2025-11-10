'''
This script contains 5 functions which can be used in the filtering/finding/tracking algorithm for the event camera
Completed for the APC524 Group Assignment: Documentation of a File
'''

### Function 1: Load Events
# This should be a super simple function. RAW event files can be very bulky, so it's
# helpful for processing to load the event stream in chunks using EventsIterator.
# EventsIterator has an input, delta_t, which defines the discrete time window with
# which to load events. We will also want to have a rolling (accumulation) window larger
# than delta_t for doing all of our analysis (filtering/clustering/tracking). This
# function would be called inside an EventsIterator chunk, and would convert the
# event stream into a numpy array.

### Function 2: Shift Window
# Again, this should be a fairly simple function. After the "Load Events" function is
# called, we will want to add our new chunk of events to the accumulation window of events,
# while also discarding the oldest delta_t chunk of events at the tail end of the window.
# This function should perform that operation, so that our window retains its size when a
# new chunk is loaded.

### Function 3: Isolated Space-Time Noise Filter
# This should be a simple filter that checks each event in the window and determines if it
# has significant neighbors in space and time. The inputs would likely be some spatial
# search radius, some temporal search window, and some minimum number of neighbors. If the
# function doesn't find enough events in the radius or window, the event should be marked
# to be filtered out as noise.

### Function 4: Low-Pass Filter
# This should be a simple filter that determines if a pixel is being activated at too high
# of a frequency to be reliable for analysis. The inputs would likely be some minimum
# flicker frequency (or 1/freq = timestep) and some minimum flicker duration. If the
# function finds that a pixel meets both of these minimum criteria, the pixel should be
# marked to be filtered out as noise.

### Function 5: Hot-Pixel Filter
# This should be fairly similar to the Low-Pass Filter, but deals instead with pixels
# demonstrating a sustained on/off period. The only input would likely be some minimum
# duration for a sustained on or off event. If the function finds that a pixel sustains
# either an on or off state for the minimum duration, the pixel should be marked to be
# filtered out as noise.