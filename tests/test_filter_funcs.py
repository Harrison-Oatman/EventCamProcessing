import eventcamprocessing


def test_isolated_noise_filter():
    import numpy as np

    from eventcamprocessing.filter_funcs import isolated_noise_filter

    dtype = np.dtype([("x", "i4"), ("y", "i4"), ("t", "i8"), ("p", "i1")])
    evs = np.array(
        [
            (10, 10, 1000, 1),
            (10, 11, 1000, 1),
            (10, 10, 1001, 1),
            (50, 50, 2000, -1),  # isolated event
            (11, 10, 1002, 1),
        ],
        dtype=dtype,
    )

    filtered_evs = isolated_noise_filter(
        evs, spatial_radius=5, time_window=10, min_neighbors=2
    )

    assert len(filtered_evs) == 4
    assert all(
        (50, 50, 2000, -1) != (ev["x"], ev["y"], ev["t"], ev["p"])
        for ev in filtered_evs
    )
