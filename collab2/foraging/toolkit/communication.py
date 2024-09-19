import copy
from typing import Callable, List, Optional

import pandas as pd
import numpy as np

from collab2.foraging.toolkit import filter_by_distance
from collab2.foraging.toolkit.filtering import constraint_filter_close_to_reward
from collab2.foraging.toolkit.point_contribution import (
    _exponential_decay,
    _point_contribution,
)
from collab2.foraging.toolkit.utils import dataObject  # noqa: F401


def _generate_communication_predictor(
    foragers: List[pd.DataFrame],
    foragersDF: pd.DataFrame,
    rewards: List[pd.DataFrame],
    memory: int,
    local_windows: List[List[pd.DataFrame]],
    predictor_name: str,
    interaction_length: float,
    interaction_constraint: Optional[
        Callable[[List[int], int, int, pd.DataFrame], List[int]]
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
                    interaction_constraint=constraint_filter_close_to_reward,
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
