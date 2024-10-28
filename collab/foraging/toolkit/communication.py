import copy
from typing import Any, Callable, List, Optional, Union

import pandas as pd

from collab.foraging.toolkit.filtering import (
    constraint_filter_close_to_reward,
    filter_by_distance,
)
from collab.foraging.toolkit.point_contribution import (
    _exponential_decay,
    _point_contribution,
)
from collab.foraging.toolkit.utils import dataObject  # noqa: F401


def _generate_communication_predictor(
    foragers: List[pd.DataFrame],
    foragersDF: pd.DataFrame,
    rewards: List[pd.DataFrame],
    local_windows: List[List[pd.DataFrame]],
    predictor_name: str,
    interaction_length: float,
    interaction_constraint: Optional[
        Union[
            Callable[[List[int], int, int, pd.DataFrame], List[int]],
            Callable[[list[int], int, int, Any, float, list[int]], list[int]],
        ]
    ] = constraint_filter_close_to_reward,
    interaction_constraint_params: Optional[dict] = None,
    communication_contribution_function: Callable = _exponential_decay,
    **communication_contribution_function_kwargs,
) -> List[List[pd.DataFrame]]:

    if interaction_constraint_params is None:
        interaction_constraint_params = {}

    num_foragers = len(foragers)
    num_frames = len(foragers[0])

    interaction_constraint_params["rewards"] = rewards

    predictor = copy.deepcopy(local_windows)

    for f in range(num_foragers):
        for t in range(num_frames):
            if predictor[f][t] is not None:

                predictor[f][t][predictor_name] = 0

                # find confocals within interaction length
                interaction_partners = filter_by_distance(
                    foragersDF,
                    f,
                    t,
                    interaction_length,
                    interaction_constraint=interaction_constraint,
                    **interaction_constraint_params,
                )

                if len(interaction_partners) > 0:
                    for partner in interaction_partners:
                        partner_x = foragers[partner]["x"].iloc[t]
                        partner_y = foragers[partner]["y"].iloc[t]

                        predictor[f][t][predictor_name] += _point_contribution(
                            partner_x,
                            partner_y,
                            local_windows[f][t],
                            communication_contribution_function,
                            **communication_contribution_function_kwargs,
                        )

                # scaling to abs max (not sum, as this would lead to small numerical values)
                max_abs_over_grid = predictor[f][t][predictor_name].abs().max()  # sum()
                if max_abs_over_grid > 0:
                    predictor[f][t][predictor_name] = (
                        predictor[f][t][predictor_name] / max_abs_over_grid
                    )

    return predictor


def generate_communication_predictor(foragers_object: dataObject, predictor_name: str):
    """
    Generates communication-based predictors for a group of foragers. When a forager
    is in the vicinity of food, it can communicate this information with the other
    foragers. The predictor value is proportional to the proximity of the communicating
    partner, but only if that partner is close to a food source.
    The predictor can be customized by providing a custom communication function
    (default: exponential decay) and/or a custom interaction function (default: closeness to food).


    Arguments:
    :param foragers_object: A data object containing information about the foragers, including their positions,
                            trajectories, and local windows. Such objects can be generated using `object_from_data`.
    :param predictor_name: The name of the food predictor to be generated, used to fetch relevant parameters
                           from `foragers_object.predictor_kwargs` and to store the computed values.

    :return: A list of lists of pandas DataFrames where each DataFrame has been updated with the computed food
             predictor values.

    Predictor-specific keyword arguments:
        :param interaction_length: The maximum distance to the communicating partner.
        :param interaction_constraint: An optional callable that imposes additional constraints on which
                                    foragers can interact based on custom logic. Default is
                                    `constraint_filter_close_to_reward`
        :param interaction_constraint_params: Optional parameters to pass to the `interaction_constraint`
                                            function. For `constraint_filter_close_to_reward`, this
                                            includes `finders_tolerance` - the maximal distance
                                            of the communicating partner to the food source.
        :param communication_contribution_function: The decay function for computing the strength of the communication.
        The value of the communication predictor will be equal to the total contribution from the
        individual communicating partners.
            The default value is the exponential decay function: f(dist) = exp(-decay_factor * dist).
            The default decay factor is 0.5, it can be customized by passing
            an additional `decay_factor` keyword argument.
    """
    params = foragers_object.predictor_kwargs[predictor_name]

    predictor = _generate_communication_predictor(
        foragers_object.foragers,
        foragers_object.foragersDF,
        rewards=foragers_object.rewards,
        local_windows=foragers_object.local_windows,
        predictor_name=predictor_name,
        **params,
    )

    return predictor
