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

    params = foragers_object.predictor_kwargs[predictor_name]

    predictor = _generate_access_predictor(
        foragers=foragers_object.foragers,
        local_windows=foragers_object.local_windows,
        predictor_name=predictor_name,
        **params,
    )

    return predictor
