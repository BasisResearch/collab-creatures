import copy
from typing import Callable, List

import pandas as pd

from collab2.foraging.toolkit.point_contribution import (
    _exponential_decay,
    _point_contribution,
)
from collab2.foraging.toolkit.utils import dataObject


def _generate_access_predictor(
    foragers: List[pd.DataFrame],
    local_windows: List[List[pd.DataFrame]],
    predictor_name: str,
    decay_contribution_function: Callable = _exponential_decay,
    **decay_contribution_function_kwargs,
) -> List[List[pd.DataFrame]]:

    num_foragers = len(foragers)
    num_frames = len(foragers[0])
    predictor = copy.deepcopy(local_windows)

    for f in range(num_foragers):
        for t in range(num_frames):
            if predictor[f][t] is not None:

                predictor[f][t][predictor_name] = 0

                current_x = foragers[f].loc[foragers[f]["time"] == t, "x"].item()
                current_y = foragers[f].loc[foragers[f]["time"] == t, "y"].item()

                predictor[f][t][predictor_name] += _point_contribution(
                    current_x,
                    current_y,
                    local_windows[f][t],
                    decay_contribution_function,
                    **decay_contribution_function_kwargs,
                )

                max_abs_over_grid = predictor[f][t][predictor_name].abs().max()
                if max_abs_over_grid > 0:
                    predictor[f][t][predictor_name] = (
                        predictor[f][t][predictor_name] / max_abs_over_grid
                    )

    return predictor


def generate_access_predictor(foragers_object: dataObject, predictor_name: str):
    """
    Generates access-based predictors for a group of foragers. Access is defined as the ability of a forager
    to reach a specific location in space. For a homogeneous environment, the value of the predictor is
    inversely proportional to the distance between the forager and the target location. The decay function
    can be customized.

    Arguments:
    :param foragers_object: A data object containing information about the foragers, including their positions,
                            trajectories, and local windows. Such objects can be generated using `object_from_data`.
    :param predictor_name: The name of the access predictor to be generated, used to fetch relevant parameters
                           from `foragers_object.predictor_kwargs` and to store the computed values.

    :return: A list of lists of pandas DataFrames where each DataFrame has been updated with the computed access
             predictor values.

    Predictor-specific keyword arguments:
        :param decay_contribution_function: The decay function used to compute the value of the access predictor.
            The default value is the exponential decay function: f(dist) - exp(-decay_factor * dist).
            The default decay factor is 0.5, it can be customized by passing
            an additional `decay_factor` keyword argument.
    """
    params = foragers_object.predictor_kwargs[predictor_name]

    predictor = _generate_access_predictor(
        foragers=foragers_object.foragers,
        local_windows=foragers_object.local_windows,
        predictor_name=predictor_name,
        **params,
    )

    return predictor
