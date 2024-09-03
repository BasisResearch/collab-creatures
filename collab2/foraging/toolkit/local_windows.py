import copy
from itertools import product
from typing import Any, Callable, List, Optional
from collab2.foraging.toolkit.utils import dataObject
import numpy as np
import pandas as pd


def _get_grid(
    grid_size: int,
    sampling_fraction: Optional[float] = 1.0,
    random_seed: Optional[int] = 0,
    grid_constraint : Optional[Callable[[pd.DataFrame,Any],pd.DataFrame]] = None,
    **grid_constraint_params,
) -> pd.DataFrame:
    """
    A helper function that generates a grid of size `grid_size` with options to subsample and 
    apply geometric constraints
    :param grid_size: size of grid
    :param sampling_fraction: fraction of grid points to keep while subsampling
    :param random_seed: random state (for reproducibility of subsampling)
    :param grid_constraint: an optional callable that implements the desired geometric constraint. 
        Takes as inputs current grid and any other kwargs. Eg:
            def circular_constraint_func(grid, c_x,c_y,R):
                ind = ((grid["x"] - c_x) ** 2 + (grid["y"]- c_y) ** 2) < R**2
                return grid.loc[ind]
    :param grid_constraint_params: optional kwargs for grid_constraint
    
    :return: computed grid, as DataFrame with "x","y" columns 
    """
    # generate grid of all points
    mesh = product(range(grid_size), repeat=2)
    grid = pd.DataFrame(mesh, columns=["x", "y"])

    # only keep accessible points
    if grid_constraint is not None:
        grid = grid_constraint(grid, **grid_constraint_params)

    # subsample the grid
    np.random.seed(random_seed)
    drop_ind = np.random.choice(grid.index, int(len(grid) * (1 - sampling_fraction)))
    grid = grid.drop(drop_ind)

    return grid


def _generate_local_windows(
    foragers: List[pd.DataFrame],
    grid_size: int,
    window_size: float,
    sampling_fraction: float = 1.0,
    random_seed: int = 0,
    skip_incomplete_frames: bool = False,
    grid_constraint : Optional[Callable[[pd.DataFrame,Any],pd.DataFrame]] = None,
    **grid_constraint_params,
) -> List[List[pd.DataFrame]]:
    """
    A function that calculates local_windows, i.e. grid points to compute predictors over, 
    for each forager at each time step.
    :param foragers: list of DataFrames containing forager trajectory, grouped by forager index
    :param grid_size: size of grid used to discretize positional data. Note that this argument is not 
        exposed to users, but inherited from `foragers_object` in `generate_local_windows`
    :param window_size: radius of local_windows 
    :param sampling_fraction: fraction of grid points to sample. It may be advisable to subsample 
        grid points for speed
    :param random_seed: random state for subsampling 
    :param skip_incomplete_frames: If True, `local_windows` for *all* foragers are set to `None`
        whenever tracks for *any* forager is missing. This implies that frames with incomplete
        tracking would be skipped entirely from subsequent predictor/score computations. If False (default
        behavior) `local_windows` are set to `None` only for the missing foragers, and computations proceed as normal
        for foragers in frame
    :param grid_constraint: Optional callable to model inaccessible points in the grid. This function takes as arguments
        the grid (as a pd.DataFrame) and any additional kwargs, and returns a DataFrame of accessible grid points 
    :param grid_constrain_params: optional additional kwargs for `grid_constraint`

    :return: Nested list of local_windows (DataFrames with "x","y" columns) grouped by forager index and time
    """

    # initialize a common grid
    grid = _get_grid(
        grid_size=grid_size,
        sampling_fraction=sampling_fraction,
        random_seed=random_seed,
        grid_constraint=grid_constraint,
        **grid_constraint_params,
    )
    num_foragers = len(foragers)
    num_frames = len(foragers[0])
    f_present_frames = []
    for f in range(num_foragers):
        tracked_idx = foragers[f].loc[:, ["x", "y"]].notna().all(axis=1)
        f_present_frames.append(foragers[f]["time"].loc[tracked_idx].to_list())

    # identify time points where ALL foragers are present
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


def generate_local_windows(foragers_object : dataObject) -> List[List[pd.DataFrame]]:
    """
    A wrapper function that calculates `local_windows` for a dataObject by calling `_generate_local_windows` 
    with parameters inherited from the dataObject. 
    :param foragers_object: dataObject containing foragers trajectory data. 
        Must have `local_windows_kwargs` as an attribute

    :return: Nested list of local_windows (DataFrames with "x","y" columns) grouped by forager index and time
    """
    # grab parameters specific to local_windows
    params = foragers_object.local_windows_kwargs

    # call hidden function with keyword arguments
    local_windows = _generate_local_windows(
        foragers=foragers_object.foragers,
        grid_size=foragers_object.grid_size,
        **params
    )

    return local_windows
