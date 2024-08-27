import copy
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from collab2.foraging.toolkit.utils import dataObject 
from collab2.foraging.toolkit.filtering import filter_by_distance
from collab2.foraging.toolkit.decay import _decaying_contribution, _exponential_decay


def _generate_food_predictor(
    rewards: List[pd.DataFrame],
    foragers: List[pd.DataFrame],
    local_windows: List[List[pd.DataFrame]],
    predictor_name: str,
    decay_function: Callable = _exponential_decay,
    **decay_function_kwargs,
) -> List[List[pd.DataFrame]]:

    num_foragers = len(foragers)
    num_frames = len(foragers[0])
    predictor = copy.deepcopy(local_windows)

    for f in range(num_foragers):
        for t in range(num_frames):
            if predictor[f][t] is not None:

                predictor[f][t][predictor_name] = 0


                if len(rewards) > 0:
                    for reward in rewards:
                        reward_x = reward["x"].iloc[t]
                        reward_y = reward["y"].iloc[t]

                        predictor[f][t][predictor_name] += _decaying_contribution(
                            reward_x,
                            reward_y,
                            local_windows[f][t],
                            decay_function,
                            **decay_function_kwargs,
                        )

                max_abs_over_grid = predictor[f][t][predictor_name].abs().max()
                if max_abs_over_grid > 0:
                    predictor[f][t][predictor_name] = (
                        predictor[f][t][predictor_name] / max_abs_over_grid
                    )

    return predictor

def generate_food_predictor(foragers_object: dataObject, predictor_name: str):
    
    params = foragers_object.predictor_kwargs[predictor_name]

    predictor = _generate_food_predictor(
        foragers=foragers_object.foragers,
        rewards=foragers_object.rewards,
        local_windows=foragers_object.local_windows,
        predictor_name=predictor_name,
        **params,
    )

    return predictor
