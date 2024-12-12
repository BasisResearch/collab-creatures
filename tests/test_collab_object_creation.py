import numpy as np
import pandas as pd

from collab.foraging.toolkit import (
    dataObject,
    rescale_to_grid,
    subsample_frames_constant_frame_rate,
    subset_frames_evenly_spaced,
)


def test_dataObject():
    np.random.seed(42)

    # generate random test data
    num_frames = 30
    num_foragers = 3

    data = {
        "x": np.random.randn(num_frames * num_foragers),
        "y": np.random.randn(num_frames * num_foragers),
        "time": np.tile(np.arange(num_frames), num_foragers),
        "forager": np.concatenate(
            [i * np.ones(num_frames) for i in range(num_foragers)]
        ),
    }

    foragersDF = pd.DataFrame(data)

    # include nans and missing data
    nan_ind = np.random.randint(0, num_frames * num_foragers, size=3)
    drop_ind = np.random.randint(0, num_frames * num_foragers, size=3)
    foragersDF.loc[nan_ind, ["x", "y"]] = np.nan
    foragersDF = foragersDF.drop(drop_ind)

    # data processing
    grid_size = 5
    frame_spacing = 2
    desired_frames = int(num_frames / frame_spacing)
    gridMin = -2
    gridMax = 2

    foragersDF_scaled = rescale_to_grid(
        foragersDF, size=grid_size, gridMin=gridMin, gridMax=gridMax
    )
    foragersDF_scaled_subsampled1 = subset_frames_evenly_spaced(
        foragersDF_scaled, desired_frames
    )
    foragersDF_scaled_subsampled2 = subsample_frames_constant_frame_rate(
        foragersDF_scaled, frame_spacing
    )

    foragers_object1 = dataObject(foragersDF_scaled_subsampled1)
    assert foragers_object1.foragersDF["time"].min() == 0  # time starts at 0
    assert foragers_object1.num_frames == desired_frames  # check subsampling
    assert (
        foragers_object1.foragersDF.loc[:, ["x", "y"]].min(axis=None) == 0
    )  # grid starts at 0
    assert foragers_object1.grid_size == grid_size  # check rescaling

    # check that missing data was filled
    for f in range(num_foragers):
        assert foragers_object1.foragers[f].shape == (desired_frames, 4)

    # repeat for foragers_object2
    foragers_object2 = dataObject(foragersDF_scaled_subsampled2)
    assert foragers_object2.foragersDF["time"].min() == 0  # time starts at 0
    assert foragers_object2.num_frames == desired_frames  # check subsampling
    assert (
        foragers_object2.foragersDF.loc[:, ["x", "y"]].min(axis=None) == 0
    )  # grid starts at 0
    assert foragers_object2.grid_size == grid_size  # check rescaling

    # check that missing data was filled
    for f in range(num_foragers):
        assert foragers_object2.foragers[f].shape == (desired_frames, 4)
