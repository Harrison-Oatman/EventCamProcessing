from metavision_core.event_io import EventsIterator
import numpy as np

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
    particle_info = np.array(particles, dtype=dtype)  # reformat particle info to structured array

    if len(particle_info) != 0:
        print(f"Found {len(particles)} particles at t = {round(particle_info['t'][-1] / 10e6, 5)} s.")

    return particle_info



def plot_last_frame(raw_path, accum_time, min_area,
                               height=720, width=1280):
    """
    Plot the last pseudoframe of a .raw recording, with 
    bounding boxes around identified particles
    """

    # load events from file
    mv_it = EventsIterator(raw_path, delta_t=2000)
    window = []

    for evs in mv_it:
        # update accumulation window
        window = accumulate_events(window=window,
                                   new_chunk=evs,
                                   t_accum_us=accum_time
                                   )

        # create binary frame
        ON_events = window[window["p"] == 1]
        frame = np.zeros((height, width), dtype=np.uint8)
        frame[ON_events["y"], ON_events["x"]] = 1

        # cluster events in frame
        label_img = label(frame, connectivity=2)
        regions = regionprops(label_img)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.imshow(frame, cmap='gray', interpolation='none', vmin=0, vmax=1)
    ax.set_title(f"still frame with bounding boxes, acc={accum_time/1000} ms, min area={min_area}")
    ax.axis("off")

    for reg in regions:
        if reg.area < min_area:
            continue

        miny, minx, maxy, maxx = reg.bbox
        bb_buffer = 5  # edge buffer for bounding box
        rect = plt.Rectangle(
            (minx - bb_buffer, miny - bb_buffer),
            maxx - minx + bb_buffer,
            maxy - miny + bb_buffer,
            fill=False,
            edgecolor="red",
            linewidth=1
        )
        ax.add_patch(rect)

    plt.show()