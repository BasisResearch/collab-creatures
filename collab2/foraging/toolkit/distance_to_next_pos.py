import copy
from typing import List

import numpy as np
import pandas as pd

from collab2.foraging.toolkit import dataObject


def _generate_distance_to_next_pos(
    foragers: List[pd.DataFrame],
    local_windows: List[List[pd.DataFrame]],
    window_size: int,
):
    """
    A function that computes a score for how far grid points are from the next position of a forager.
    If the next position of the forager is unavailable, nan values are assigned to the scores.

    Parameters:
        - foragers : list of DataFrames containing forager positions, grouped by forager index
        - local_windows :  Nested list of DataFrames containing grid points to compute predictor over,
            grouped by forager index and time
        - window_size : Radius of local windows

    Returns:
        - distance_to_next_pos : Nested list of calculated scores, grouped by foragers and time
    """

    num_foragers = len(foragers)
    num_frames = len(foragers[0])
    distance_to_next_pos = copy.deepcopy(local_windows)

    for f in range(num_foragers):
        for t in range(num_frames - 1):
            if distance_to_next_pos[f][t] is not None:
                x_new = foragers[f].at[t + 1, "x"]
                y_new = foragers[f].at[t + 1, "y"]
                if np.isfinite(x_new) and np.isfinite(y_new):
                    distance_to_next_pos[f][t]["raw_distance_to_next_pos"] = np.sqrt(
                        (distance_to_next_pos[f][t]["x"] - x_new) ** 2
                        + (distance_to_next_pos[f][t]["y"] - y_new) ** 2
                    )
                    distance_to_next_pos[f][t]["scored_raw_distance_to_next_pos"] = (
                        1
                        - (
                            distance_to_next_pos[f][t]["raw_distance_to_next_pos"]
                            / (2 * window_size)
                        )
                        ** 2
                    )
                    distance_to_next_pos[f][t][
                        "rescaled_scored_raw_distance_to_next_pos"
                    ] = (
                        distance_to_next_pos[f][t]["scored_raw_distance_to_next_pos"]
                        - distance_to_next_pos[f][t][
                            "scored_raw_distance_to_next_pos"
                        ].min()
                    ) / (
                        distance_to_next_pos[f][t][
                            "scored_raw_distance_to_next_pos"
                        ].max()
                        - distance_to_next_pos[f][t][
                            "scored_raw_distance_to_next_pos"
                        ].min()
                    )
                    distance_to_next_pos[f][t]["rescaled_distance_to_next_pos"] = (
                        distance_to_next_pos[f][t]["raw_distance_to_next_pos"]
                        - distance_to_next_pos[f][t]["raw_distance_to_next_pos"].min()
                    ) / (
                        distance_to_next_pos[f][t]["raw_distance_to_next_pos"].max()
                        - distance_to_next_pos[f][t]["raw_distance_to_next_pos"].min()
                    )
                    distance_to_next_pos[f][t][
                        "scored_rescaled_distance_to_next_pos"
                    ] = (
                        1 - distance_to_next_pos[f][t]["rescaled_distance_to_next_pos"]
                    )
                else:
                    distance_to_next_pos[f][t]["raw_distance_to_next_pos"] = np.nan
                    distance_to_next_pos[f][t][
                        "scored_raw_distance_to_next_pos"
                    ] = np.nan
                    distance_to_next_pos[f][t][
                        "rescaled_scored_raw_distance_to_next_pos"
                    ] = np.nan
                    distance_to_next_pos[f][t]["rescaled_distance_to_next_pos"] = np.nan
                    distance_to_next_pos[f][t][
                        "scored_rescaled_distance_to_next_pos"
                    ] = np.nan

        # save nans for last frame
        distance_to_next_pos[f][num_frames - 1]["raw_distance_to_next_pos"] = np.nan
        distance_to_next_pos[f][num_frames - 1][
            "scored_raw_distance_to_next_pos"
        ] = np.nan
        distance_to_next_pos[f][num_frames - 1][
            "rescaled_scored_raw_distance_to_next_pos"
        ] = np.nan
        distance_to_next_pos[f][num_frames - 1][
            "rescaled_distance_to_next_pos"
        ] = np.nan
        distance_to_next_pos[f][num_frames - 1][
            "scored_rescaled_distance_to_next_pos"
        ] = np.nan

    return distance_to_next_pos


def generate_distance_to_next_pos(foragers_object: dataObject):
    """
    A wrapper function that computes `distance_to_next_pos` only taking `foragers_object` as argument,
    and calling `_generate_distance_to_next_pos` under the hood

    Parameters:
        - foragers_object : dataObject containing positional data and necessary kwargs

    Returns:
        - distance_to_next_pos : Nested list of calculated scores, grouped by foragers and time
    """
    window_size = foragers_object.local_windows_kwarg["window_size"]
    return _generate_distance_to_next_pos(
        foragers_object.foragers, foragers_object.local_windows, window_size
    )
