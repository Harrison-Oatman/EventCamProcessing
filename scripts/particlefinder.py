import matplotlib.pyplot as plt
import numpy as np
from metavision_core.event_io import EventsIterator
from skimage.measure import label, regionprops

from eventcamprocessing.filter_funcs import accumulate_events


def plot_last_frame(raw_path, accum_time, min_area, height=720, width=1280):
    """
    Plot the last pseudoframe of a .raw recording, with
    bounding boxes around identified particles
    """

    # load events from file
    mv_it = EventsIterator(raw_path, delta_t=2000)
    window = []

    for evs in mv_it:
        # update accumulation window
        window = accumulate_events(window=window, new_chunk=evs, t_accum_us=accum_time)

        # create binary frame
        ON_events = window[window["p"] == 1]
        frame = np.zeros((height, width), dtype=np.uint8)
        frame[ON_events["y"], ON_events["x"]] = 1

        # cluster events in frame
        label_img = label(frame, connectivity=2)
        regions = regionprops(label_img)

    # Plot results
    _fig, ax = plt.subplots(figsize=(10, 6))
    plt.imshow(frame, cmap="gray", interpolation="none", vmin=0, vmax=1)
    ax.set_title(
        f"still frame with bounding boxes, acc={accum_time / 1000} ms, min area={min_area}"
    )
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
            linewidth=1,
        )
        ax.add_patch(rect)

    plt.show()
