import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import label, regionprops

from eventcamprocessing.filter_funcs import accumulate_events


def ev_particletracker(all_particles, max_disp, time_array):
    """
    Call after ev_particlefinder has detected all particles in an event
    recording and stored information in a global array. Places particles
    in tracks based on predicted displacements, a cost function, and a
    maximum displacement criterion. Returns information about length and
    position for all tracks.

    Parameters
    ----------
    all_particles : np.ndarray
        Array of tuples. Each tuple has the following fields, pertaining
        to an identified particle: x (centroid), y (centroid), t (centroid),
        area (# of events)
    max_disp : float
        Maximum displacement in (x, y)-space for particles to be linked
        in the same track (Only applied to tracks of length 1).
    time_array: np.ndarray
        1D array of timesteps corresponding to the delta_t used in EventsIterator.
        Most usefully constructed as time_array = np.arange(t_start, t_end + dt,
        dt), where t_start and t_end are the timestamps of the first and last
        events in the recording, and dt is the timestep used in EventsIterator.

    Returns
    -------
    track_info : list
        List of every track, each containing the following fields--
        "L" : (int) track length (# of coordinates found)
        "X" : (np.ndarray) X-position at each coordinate
        "Y" : (np.ndarray) Y-position at each coordinate
        "T" : (np.ndarray) T-position at each coordinate
    """

    # sort particles by increasing time
    p_sorted = np.asarray(sorted(all_particles, key=lambda p: p["t"]))

    # using particles from first window, initialize tracks
    ps_1 = p_sorted[(p_sorted["t"] > time_array[0]) & (p_sorted["t"] <= time_array[1])]
    track_info = []
    if len(ps_1) > 0:
        for particle in ps_1:
            track = {
                "L": 1,
                "X": [float(particle["x"])],
                "Y": [float(particle["y"])],
                "T": [float(particle["t"])],
            }
            track_info.append(track)
    else:
        pass

    # log tracks that are active
    num_active = len(ps_1)
    active = np.arange(0, num_active, 1, dtype=int)
    print(
        f"(1/{len(time_array)}): During times "
        f"t = {[float(round(time_array[0] / 10e6, 5)), float(round(time_array[1] / 10e6, 5))]} s, "
        f"there were {num_active} active tracks and {len(track_info)} total tracks."
    )

    # loop over each window to track particles
    for tt in range(1, len(time_array) - 1):
        # get current and previous locations for each active track
        current = np.zeros((num_active, 3))
        prev = np.zeros((num_active, 3))
        for tr in range(num_active):
            # print(active[tr])
            track = track_info[active[tr]]
            current[tr, 0] = track["X"][track["L"] - 1]
            current[tr, 1] = track["Y"][track["L"] - 1]
            current[tr, 2] = track["T"][track["L"] - 1]
            if track["L"] > 1:
                prev[tr, 0] = track["X"][track["L"] - 2]
                prev[tr, 1] = track["Y"][track["L"] - 2]
                prev[tr, 2] = track["T"][track["L"] - 2]
            else:
                prev[tr, :] = current[tr, :]

        # estimate the position of the matching particle (for each track),
        # based on previous trajectory
        delta = current - prev
        pos_est = current + delta

        # initialize arrays for cost and linking
        costs = np.zeros(num_active)
        pairs = np.zeros(num_active)

        new_ps = p_sorted[
            (p_sorted["t"] > time_array[tt]) & (p_sorted["t"] <= time_array[tt + 1])
        ]
        if len(new_ps) > 0:
            # loop over active tracks to determine costs and pairs
            for tr in range(num_active):
                if (
                    track_info[active[tr]]["L"] > 1
                ):  # enhance prediction if previous displacement is known
                    all_dists = (
                        ((pos_est[tr, 0] - new_ps["x"]) / delta[tr, 0]) ** 2
                        + ((pos_est[tr, 1] - new_ps["y"]) / delta[tr, 1]) ** 2
                        + ((pos_est[tr, 2] - new_ps["t"]) / delta[tr, 2]) ** 2
                    )
                    max_disp_error = np.sqrt(
                        3
                    )  # assuming each component above is < ~O(1) for reliable tracks

                else:  # otherwise, use a simple displacement error
                    all_dists = (pos_est[tr, 0] - new_ps["x"]) ** 2 + (
                        pos_est[tr, 1] - new_ps["y"]
                    ) ** 2
                    max_disp_error = (
                        max_disp  # use global prescribed maximum displacement
                    )

                # if all distances are greater than max_disp, don't link
                costs[tr] = min(all_dists)
                if costs[tr] > max_disp_error**2:
                    continue

                # if there are two particles that minimize cost, end the track
                if len(np.where(all_dists == costs[tr])[0]) > 1:
                    continue

                best_match = np.where(all_dists == costs[tr])[0][
                    0
                ]  # choose the best match

                # check if this particle was claimed by another track
                other_claims = np.where(pairs == best_match)[0]
                if len(other_claims) > 0:
                    if (
                        costs[other_claims[0]] > costs[tr]
                    ):  # give particle to better-fitting track
                        pairs[other_claims[0]] = 0
                    else:
                        continue
                pairs[tr] = best_match

            # add particles to tracks
            paired = np.zeros(len(new_ps))
            for tr in range(num_active):
                if pairs[tr] != 0:
                    ind = int(pairs[tr])
                    track = track_info[active[tr]]
                    track["L"] += 1
                    track["X"].append(float(new_ps[ind]["x"]))
                    track["Y"].append(float(new_ps[ind]["y"]))
                    track["T"].append(float(new_ps[ind]["t"]))

                    paired[ind] = 1
            active = active[
                pairs != 0
            ]  # remove unpaired tracks from the list of active tracks

            # create new tracks with unpaired particles
            unpaired = np.where(paired == 0)[0]
            new_tracks = []
            for ii in range(len(unpaired)):
                particle = new_ps[unpaired[ii]]
                track = {
                    "L": 1,
                    "X": [float(particle["x"])],
                    "Y": [float(particle["y"])],
                    "T": [float(particle["t"])],
                }
                new_tracks.append(track)

        else:  # if no new particles
            active = []
            new_tracks = []
            unpaired = []

        if isinstance(active, np.ndarray):
            active = active.tolist()
        if len(new_tracks) > 0:
            for ii in range(len(new_tracks)):
                active.append(len(track_info) + ii)
        active = np.asarray(active)
        num_active = len(active)
        if len(new_tracks) > 0:
            for ii in range(len(new_tracks)):
                track_info.append(new_tracks[ii])

        print(
            f"({tt + 1}/{len(time_array)}): During times "
            f"t = {[float(round(time_array[tt] / 10e6, 5)), float(round(time_array[tt + 1] / 10e6, 5))]} s, "
            f"there were {num_active} active tracks, {len(new_tracks)} new tracks, "
            f"and {len(track_info)} total tracks."
        )

    return track_info


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
