import copy
from typing import List, Optional

import numpy as np
import pandas as pd

from collab2.foraging.toolkit.utils import dataObject


def _generate_next_step_score(
    foragers: List[pd.DataFrame],
    local_windows: List[List[pd.DataFrame]],
    n: Optional[float] = 1.0,
):
    """
    A function that computes a score for how far grid points are from the next position of a forager.
    If the next position of the forager is unavailable, nan values are assigned to the scores.

    Parameters:
        - foragers : list of DataFrames containing forager positions, grouped by forager index
        - local_windows :  Nested list of DataFrames containing grid points to compute predictor over,
            grouped by forager index and time
        - n : Nonlinearity in  `next_step_score`. Default value n=1

    Returns:
        - next_step_score : Nested list of calculated scores, grouped by foragers and time
    """

    num_foragers = len(foragers)
    num_frames = len(foragers[0])
    next_step_score = copy.deepcopy(local_windows)

    for f in range(num_foragers):
        for t in range(num_frames - 1):
            if next_step_score[f][t] is not None:
                x_new = foragers[f].at[t + 1, "x"]
                y_new = foragers[f].at[t + 1, "y"]
                if np.isfinite(x_new) and np.isfinite(y_new):
                    next_step_score[f][t]["distance_to_next_step"] = np.sqrt(
                        (next_step_score[f][t]["x"] - x_new) ** 2
                        + (next_step_score[f][t]["y"] - y_new) ** 2
                    )
                    next_step_score[f][t]["scaled_distance_to_next_step"] = (
                        next_step_score[f][t]["distance_to_next_step"]
                        - next_step_score[f][t]["distance_to_next_step"].min()
                    ) / (
                        next_step_score[f][t]["distance_to_next_step"].max()
                        - next_step_score[f][t]["distance_to_next_step"].min()
                    )

                    next_step_score[f][t]["next_step_score"] = 1 - (
                        next_step_score[f][t]["scaled_distance_to_next_step"]
                    ) ** n
                else:
                    next_step_score[f][t]["distance_to_next_step"] = np.nan
                    next_step_score[f][t]["scaled_distance_to_next_pos"] = np.nan
                    next_step_score[f][t]["next_step_score"] = np.nan

        # save nans for last frame
        next_step_score[f][num_frames - 1]["distance_to_next_step"] = np.nan
        next_step_score[f][num_frames - 1]["scaled_distance_to_next_pos"] = np.nan
        next_step_score[f][num_frames - 1]["next_step_score"] = np.nan

    return next_step_score


def generate_next_step_score(foragers_object: dataObject, n):
    """
    A wrapper function that computes `next_step_score` only taking `foragers_object` as argument,
    and calling `_generate_next_step_score` under the hood

    Parameters:
        - foragers_object : dataObject containing positional data, local_windows

    Returns:
        - next_step_score : Nested list of calculated scores, grouped by foragers and time
    """

    return _generate_next_step_score(
        foragers_object.foragers, foragers_object.local_windows, n
    )
