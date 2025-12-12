import numpy as np
from skimage.measure import label, regionprops


def ev_particlefinder(evs, min_area, h=720, w=1280):
    """
    Call inside an EventsIterator loop to detect particles in an event chunk.
    Particles are determined using an 8-connected components method, where
    adjacent events (by edge OR corner) are grouped together. Clusters must
    be larger than a prescribed min_area to be considered particles.

    Imports
    -------
    from metavision_core.event_io import EventsIterator
    from skimage.measure import label, regionprops
    import numpy as np

    Parameters
    ----------
    evs : np.ndarray
        Numpy array of current event window (updated by accumulate_events).
    min_area : int
        Minimum area (event count) for an event cluster to be considered a particle.
    h, w : int
        Height and width of the EVK sensor in pixels.

    Returns
    -------
    particle_info : ndarray
        Array of tuples. Each tuple has the following fields, pertaining
        to an identified particle: x (centroid), y (centroid), t (centroid),
        area (# of events)
    """

    ON_events = evs[evs["p"] == 1]  # use ON events for detecting particles
    # binary frame for clustering
    binary_frame = np.zeros((h, w), dtype=np.uint8)
    binary_frame[ON_events["y"], ON_events["x"]] = 1

    # timestamp and count accumulator frames (for t-centroid)
    ts_frame = np.zeros((h, w), dtype=np.float64)
    count_frame = np.zeros((h, w), dtype=np.uint32)
    ts_frame[ON_events["y"], ON_events["x"]] += ON_events["t"]
    count_frame[ON_events["y"], ON_events["x"]] += 1

    # cluster events based on 8-connected components
    label_ = label(binary_frame, connectivity=2)
    regions = regionprops(label_)

    particles = []
    for region in regions:
        if region.area < min_area:  # filter out particles that are too small
            continue

        # particle centroid
        y, x = region.centroid

        # particle's individual event coordinates
        y_coords = region.coords[:, 0]
        x_coords = region.coords[:, 1]

        # calculate time centroid
        t_sum = ts_frame[y_coords, x_coords].sum()
        t_count = count_frame[y_coords, x_coords].sum()
        t_centroid = t_sum / t_count

        # append new particle info
        particles.append((x, y, t_centroid, region.area))

    dtype = np.dtype([("x", "f4"), ("y", "f4"), ("t", "f8"), ("area", "i4")])
    particle_info = np.array(
        particles, dtype=dtype
    )  # reformat particle info to structured array

    if len(particle_info) != 0:
        print(
            f"Found {len(particles)} particles at t = {round(particle_info['t'][-1] / 10e6, 5)} s."
        )

    return particle_info
