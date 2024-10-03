import copy
from typing import List, Optional

import numpy as np
import pandas as pd

from collab2.foraging.toolkit.utils import dataObject


def _generate_nextStep_score(
    foragers: List[pd.DataFrame],
    local_windows: List[List[pd.DataFrame]],
    score_name: str,
    nonlinearity_exponent: Optional[float] = 1.0,
):
    """
    A function that computes a score for how far grid points are from the next position of a forager.
    If the next position of the forager is unavailable, nan values are assigned to the scores.

    :param foragers: list of DataFrames containing forager positions, grouped by forager index
    :param local_windows:  Nested list of DataFrames containing grid points to compute predictor over,
        grouped by forager index and time
    :param score_name: name of column to save the calculated nextStep scores under
    :param nonlinearity_exponent: Nonlinearity in  nextStep score calculation. Default value n=1

    :return: Nested list of calculated scores, grouped by foragers and time
    """

    num_foragers = len(foragers)
    num_frames = len(foragers[0])
    score = copy.deepcopy(local_windows)

    for f in range(num_foragers):
        for t in range(num_frames - 1):
            if score[f][t] is not None:
                x_new = foragers[f].at[t + 1, "x"]
                y_new = foragers[f].at[t + 1, "y"]
                if np.isfinite(x_new) and np.isfinite(y_new):
                    score[f][t]["distance_to_next_step"] = np.sqrt(
                        (score[f][t]["x"] - x_new) ** 2
                        + (score[f][t]["y"] - y_new) ** 2
                    )
                    d_scaled = (
                        score[f][t]["distance_to_next_step"]
                        - score[f][t]["distance_to_next_step"].min()
                    ) / (
                        score[f][t]["distance_to_next_step"].max()
                        - score[f][t]["distance_to_next_step"].min()
                    )

                    score[f][t][score_name] = 1 - (d_scaled) ** nonlinearity_exponent

                else:
                    score[f][t]["distance_to_next_step"] = np.nan
                    score[f][t][score_name] = np.nan

        # save nans for last frame
        score[f][num_frames - 1]["distance_to_next_step"] = np.nan
        score[f][num_frames - 1][score_name] = np.nan

    return score


def generate_nextStep_score(foragers_object: dataObject, score_name: str):
    """
    A wrapper function that computes `next_step_score` only taking `foragers_object` as argument,
    and calling `_generate_next_step_score` under the hood.

<<<<<<< HEAD
<<<<<<< HEAD
    The next step score computes a score for how far grid points are from the next position of a forager.
    If the next position of the forager is unavailable, nan values are assigned to the scores.

    The formula for the score is:
        next_step_score(i,t,x,y) = 1 - (d_scaled) ** n
        where d_scaled = (d - min(d)) / (max(d) - min(d))

    Here d is the vector of distances of grid points in the local window of forager i at time t
    from the position of forager i at time t+1.

    :param foragers_object: dataObject containing positional data, local_windows, score_kwargs
    :param score_name : name of column to save the calculated nextStep scores under

    :return: Nested list of calculated scores, grouped by foragers and time

    Keyword arguments:
        :param nonlinearity_exponent: Nonlinearity in  nextStep score calculation. Default value n=1
=======
    :param foragers_object: dataObject containing positional data, local_windows, score_kwargs
    :param score_name : name of column to save the calculated nextStep scores under

    :return: Nested list of calculated scores, grouped by foragers and time
>>>>>>> 4cc7d95b1d5591bb508f2856b4ba680f61f3d756
=======
    :param foragers_object: dataObject containing positional data, local_windows, score_kwargs
    :param score_name : name of column to save the calculated nextStep scores under

    :return: Nested list of calculated scores, grouped by foragers and time
>>>>>>> 4cc7d95b1d5591bb508f2856b4ba680f61f3d756
    """
    params = foragers_object.score_kwargs[score_name]
    return _generate_nextStep_score(
        foragers_object.foragers, foragers_object.local_windows, score_name, **params
    )
