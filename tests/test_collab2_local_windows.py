from itertools import product

import numpy as np
import pandas as pd

from collab2.foraging.toolkit import dataObject, generate_local_windows, rescale_to_grid


def test_local_windows():

    num_frames = 8
    num_foragers = 3
    grid_size = 30
    n_nans = 2  # number of nan rows
    n_missing = 2  # number of missing rows
    gridMin = -1
    gridMax = 1

    np.random.seed(0)

    # generate random test data
    theta = 2 * np.pi * np.random.rand(num_frames * num_foragers)
    r = np.random.rand(num_frames * num_foragers)

    data = {
        "x": r * np.cos(theta),
        "y": r * np.sin(theta),
        "time": np.tile(np.arange(num_frames), num_foragers),
        "forager": np.concatenate(
            [i * np.ones(num_frames) for i in range(num_foragers)]
        ),
    }

    foragersDF = pd.DataFrame(data)

    # add nan values
    nan_idx = np.random.randint(0, num_frames * num_foragers, size=n_nans)
    missing_loc = foragersDF.loc[nan_idx, ["time", "forager"]].values.astype(int)
    foragersDF.loc[nan_idx, ["x", "y"]] = np.nan

    # remove values for certain time points
    drop_idx = np.random.randint(0, num_frames * num_foragers, size=n_missing)
    missing_loc = np.concatenate(
        (missing_loc, foragersDF.loc[drop_idx, ["time", "forager"]].values.astype(int))
    )
    foragersDF = foragersDF.drop(drop_idx)

    # scaling and object creation
    foragersDF_scaled = rescale_to_grid(
        foragersDF, size=grid_size, gridMin=gridMin, gridMax=gridMax
    )
    foragers_object = dataObject(
        foragersDF_scaled, grid_size=grid_size, frames=num_frames
    )

    # local windows, skipping incomplete frames
    local_windows_kwargs = {
        "window_size": 10,
        "sampling_fraction": 0.5,
        "skip_incomplete_frames": True,
    }
    foragers_object.local_windows_kwargs = local_windows_kwargs
    local_windows_skip_incomplete = generate_local_windows(foragers_object)

    # check shape
    assert len(local_windows_skip_incomplete) == num_foragers
    for f in range(num_foragers):
        assert len(local_windows_skip_incomplete[f]) == num_frames

    # check that incomplete (missing/nan) frames are skipped for all foragers
    for t, _ in missing_loc:
        for f in range(num_foragers):
            assert local_windows_skip_incomplete[f][t] is None

    # check one random element
    complete_frames = set(range(num_frames)) - set(missing_loc[:, 0])
    f = np.random.randint(0, num_foragers)
    t = np.random.choice(list(complete_frames))
    test_element = local_windows_skip_incomplete[f][t]
    grid_distances = np.sqrt(
        (
            test_element["x"]
            - foragers_object.foragers[f].query("time == @t")["x"].values
        )
        ** 2
        + (
            test_element["y"]
            - foragers_object.foragers[f].query("time == @t")["y"].values
        )
        ** 2
    )
    assert grid_distances.max() <= local_windows_kwargs["window_size"]

    # local windows, without skipping incomplete frames
    local_windows_kwargs = {
        "window_size": 10,
        "sampling_fraction": 0.5,
        "skip_incomplete_frames": False,
    }
    foragers_object.local_windows_kwargs = local_windows_kwargs
    local_windows_no_skip = generate_local_windows(foragers_object)

    # check shape
    assert len(local_windows_no_skip) == num_foragers
    for f in range(num_foragers):
        assert len(local_windows_no_skip[f]) == num_frames

    # check that missing (nan/drop) frames have no local windows for the specific forager
    for t, f in missing_loc:
        assert local_windows_no_skip[f][t] is None

    # check one random element
    valid_combs = list(
        set(product(range(num_frames), range(num_foragers)))
        - set(map(tuple, missing_loc))
    )
    t, f = valid_combs[np.random.randint(len(valid_combs))]
    test_element = local_windows_no_skip[f][t]
    grid_distances = np.sqrt(
        (
            test_element["x"]
            - foragers_object.foragers[f].query("time == @t")["x"].values
        )
        ** 2
        + (
            test_element["y"]
            - foragers_object.foragers[f].query("time == @t")["y"].values
        )
        ** 2
    )
    assert grid_distances.max() <= local_windows_kwargs["window_size"]
