### Below is an example script for running each of
### the functions to track particles in a .raw file

import numpy as np

from metavision_core.event_io import EventsIterator
from eventcamprocessing.particle_detection import ev_particlefinder
from eventcamprocessing.particle_tracking import ev_particletracker
from eventcamprocessing.filter_funcs import accumulate_events


raw_file = "data/events_cut.raw"

h, w = 720, 1280  # pixel height and width
t_accum_us = 20000  # accumulation time
dt = 10000  #
max_disp = 8  # maximum displacement (for tracks of length 1)
t_start = None
window = []
all_particles = []


print(" ")
print("Detecting particles...")
mv_iterator = EventsIterator(raw_file, delta_t=dt)

for evs in mv_iterator:
    if t_start is None and len(evs) > 0:  # mark the starting timestamp of the recording
        t_start = evs["t"][0]

    # update accumulation window
    window = accumulate_events(window=window, new_chunk=evs, t_accum_us=t_accum_us)

    #### ADD FILTERS HERE ###

    # detect particles
    particles = ev_particlefinder(
        evs=window,
        min_area=100,
        h=h,
        w=w,
    )

    for p in particles:
        all_particles.append(p)

dtype = np.dtype([("x", "f4"), ("y", "f4"), ("t", "f8"), ("area", "i4")])
all_particles = np.array(
    all_particles, dtype=dtype
)  # reformat particle info to structured array

print(f"Finished detecting particles! Found {len(all_particles)} particles in total.")
print(" ")
print("========================================================")
print(" ")
print("Tracking Particles...")

# create array of tracking timesteps
t_end = evs["t"][-1]  # last timestamp of the recording
t_len = t_end - t_start
t_ = np.arange(t_start, t_end + dt, dt)

# track particles
track_info = ev_particletracker(all_particles, max_disp)
