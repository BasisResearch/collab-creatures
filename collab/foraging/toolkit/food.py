import copy
from typing import Callable, List

import pandas as pd

from collab.foraging.toolkit.point_contribution import (
    _exponential_decay,
    _point_contribution,
)
from collab.foraging.toolkit.utils import dataObject


def _generate_food_predictor(
    rewards: List[pd.DataFrame],  # one frame per t, with columns x and y
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

                rewards_now = rewards[t]

                if len(rewards_now) > 0:
                    for _, row in rewards_now.iterrows():
                        reward_x = row["x"]
                        reward_y = row["y"]

                        predictor[f][t][predictor_name] += _point_contribution(
                            reward_x,
                            reward_y,
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


def generate_food_predictor(foragers_object: dataObject, predictor_name: str):
    """
    Generates food-based predictors for a group of foragers. Food is given
    by the presence of rewards in the environment. The value of the predictor
    is proportional to the rewards in the vicinity of the forager. The decay
    function can be customized.

    Arguments:
    :param foragers_object: A data object containing information about the foragers, including their positions,
                            trajectories, and local windows. Such objects can be generated using `object_from_data`.
    :param predictor_name: The name of the food predictor to be generated, used to fetch relevant parameters
                           from `foragers_object.predictor_kwargs` and to store the computed values.

    :return: A list of lists of pandas DataFrames where each DataFrame has been updated with the computed food
             predictor values.

    Predictor-specific keyword arguments:
        :param decay_contribution_function: The decay function for computing the value for each reward.
        The value of the food predictor will be equal to the total contribution from the
        individual rewards.
            The default value is the exponential decay function: f(dist) = exp(-decay_factor * dist).
            The default decay factor is 0.5, it can be customized by passing
            an additional `decay_factor` keyword argument.
    """
    params = foragers_object.predictor_kwargs[predictor_name]

    predictor = _generate_food_predictor(
        foragers=foragers_object.foragers,
        rewards=foragers_object.rewards,
        local_windows=foragers_object.local_windows,
        predictor_name=predictor_name,
        **params,
    )

    return predictor
