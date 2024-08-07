import copy
from itertools import product
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd


def get_grid(
    grid_size: int = 90,
    sampling_fraction: float = 1.0,
    random_seed: int = 0,
    grid_constraint: Optional[
        Callable[[pd.DataFrame, pd.DataFrame, Optional[dict]], pd.DataFrame]
    ] = None,
    grid_constraint_params: Optional[dict] = None,
) -> pd.DataFrame:
    # generate grid of all points
    mesh = product(range(grid_size), repeat=2)
    grid = pd.DataFrame(mesh, columns=["x", "y"])

    # only keep accessible points
    if grid_constraint is not None:
        grid = grid[grid_constraint(grid["x"], grid["y"], grid_constraint_params)]

    # subsample the grid
    np.random.seed(random_seed)
    drop_ind = np.random.choice(grid.index, int(len(grid) * (1 - sampling_fraction)))
    grid = grid.drop(drop_ind)

    return grid


def _generate_local_windows(
    foragers: List[pd.DataFrame],
    grid_size: int,
    num_foragers: int,
    num_frames: int,
    window_size: float,
    sampling_fraction: float = 1.0,
    random_seed: int = 0,
    skip_incomplete_frames: bool = False,
    grid_constraint: Optional[
        Callable[[pd.DataFrame, pd.DataFrame, Any], pd.DataFrame]
    ] = None,
    grid_constraint_params: Optional[dict] = None,
) -> pd.DataFrame:

    # Note: args `grid_size`, `num_foragers`, `num_frames` are not exposed to users but set to
    # values inherited from `foragers_object`` by `generate_local_windows`.

    # initialize a common grid
    grid = get_grid(
        grid_size=grid_size,
        sampling_fraction=sampling_fraction,
        random_seed=random_seed,
        grid_constraint=grid_constraint,
        grid_constraint_params=grid_constraint_params,
    )

    f_present_frames = []
    for f in range(num_foragers):
        tracked_idx = foragers[f].loc[:, ["x", "y"]].notna().all(axis=1)
        f_present_frames.append(foragers[f]["time"].loc[tracked_idx].to_list())

    # identify time points where ALL foragers are presen
    f_present_frames_set = [set(_) for _ in f_present_frames]
    all_present_frames = set.intersection(*f_present_frames_set)

    # calculate local_windows for each forager
    local_windows = []
    for f in range(num_foragers):
        # initialize local_windows_f to None
        local_windows_f = [None for _ in range(num_frames)]

        # find frames for which local windows need to be computed
        if skip_incomplete_frames:
            compute_frames = all_present_frames
        else:
            compute_frames = f_present_frames[f]

        for t in compute_frames:
            # copy grid
            g = copy.deepcopy(grid)

            # calculate distance of points in g to the current position of forager f
            g["distance_to_f"] = np.sqrt(
                (g["x"] - foragers[f].query("time == @t")["x"].values) ** 2
                + (g["y"] - foragers[f].query("time == @t")["y"].values) ** 2
            )

            # select grid points with distance < window_size
            g = g[g["distance_to_f"] <= window_size]

            # add forager and time info to the DF
            # TODO : using assign here because everything else triggers a copy on write warning. Revisit if needed.
            g = g.assign(time=t)
            g = g.assign(forager=f)

            # update the corresponding element of local_windows_f
            local_windows_f[t] = g

        # add local_windows_f to local_windows
        local_windows.append(local_windows_f)

    return local_windows


def generate_local_windows(foragers_object) -> pd.DataFrame:
    # grab parameters specific to local_windows
    params = foragers_object.local_windows_kwargs

    # call hidden function with keyword arguments
    local_windows = _generate_local_windows(
        foragers=foragers_object.foragers,
        grid_size=foragers_object.grid_size,
        num_frames=foragers_object.num_frames,
        num_foragers=foragers_object.num_foragers,
        **params
    )

    return local_windows
