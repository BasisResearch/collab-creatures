import copy
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from collab2.foraging.toolkit import filter_by_distance
from collab2.foraging.toolkit.utils import dataObject  # noqa: F401


def _piecewise_proximity_function(
    distance: Union[float, np.ndarray],
    getting_worse: float = 1.5,
    optimal: float = 4,
    proximity_decay: float = 1,
):
    """
    Computes a piecewise proximity function based on distance, modeling the transition from 
    a suboptimal to an optimal range and beyond.

    The function uses a piecewise approach:
    1. For distances less than or equal to `getting_worse`, it applies a sine function to model 
       an increasing proximity effect, starting at -1 and reaching 0 at `getting_worse`.
    2. For distances between `getting_worse` and a mid-range value derived from `optimal`, it 
       applies another sine function to model proximity improvement.
    3. For distances beyond the optimal range, proximity decays exponentially, representing a 
       diminishing effect.

    :param distance: A float or ndarray representing the distance(s) at which proximity is evaluated.
    :param getting_worse: The distance threshold below which the score becomes negative. Defaults to 1.5.
    :param optimal: The distance where proximity reaches its peak. Defaults to 4.
    :param proximity_decay: The rate at which proximity decays beyond the optimal range. Defaults to 1.

    :return: A float or ndarray representing the computed proximity value(s) based on the input distance.
    """

    cond1 = distance <= getting_worse
    cond2 = (distance > getting_worse) & (
        distance <= getting_worse + 1.5 * (optimal - getting_worse)
    )

    result = np.where(
        cond1,
        np.sin(np.pi / (2 * getting_worse) * (distance + 3 * getting_worse)),
        np.where(
            cond2,
            np.sin(
                np.pi / (2 * (optimal - getting_worse)) * (distance - getting_worse)
            ),
            np.sin(
                np.pi
                / (2 * (optimal - getting_worse))
                * (1.5 * (optimal - getting_worse))
            )
            * np.exp(
                -proximity_decay
                * (distance - optimal - 0.5 * (optimal - getting_worse))
            ),
        ),
    )

    return result


# TODO this can be further generalized to any point-centered contribution
# with some decay/scoring function
def _proximity_predictor_contribution(
    x_other: int,
    y_other: int,
    grid: pd.DataFrame,
    proximity_function: Callable,
    **proximity_function_kwargs,
) -> np.ndarray:
    """
    Computes the proximity score contribution of an agent present at `x_other, y_other` to points present in `grid`.

    This function calculates the Euclidean distance from the point `(x_other, y_other)` to each point 
    in the grid and applies the provided proximity function to determine the proximity score for those points 
    as a function of distance.

    :param x_other: The x-coordinate of an agent.
    :param y_other: The y-coordinate of an agent.
    :param grid: A pandas DataFrame containing the x and y coordinates of the grid points. The DataFrame
                 must have columns named 'x' and 'y'.
    :param proximity_function: A callable function that takes distance as input and computes proximity scores.
    :param proximity_function_kwargs: Additional keyword arguments to pass to the proximity function.

    :return: A numpy ndarray representing the computed proximity scores for each point in the grid.
    """

    distance_to_other = np.sqrt((grid["x"] - x_other) ** 2 + (grid["y"] - y_other) ** 2)
    proximity_score = proximity_function(distance_to_other, **proximity_function_kwargs)
    return proximity_score


def _proximity_predictor(
    foragers: List[pd.DataFrame],
    foragersDF: pd.DataFrame,
    local_windows: List[List[pd.DataFrame]],
    predictor_name: str,
    interaction_length: float,
    interaction_constraint: Optional[
        Callable[[List[int], int, int, pd.DataFrame, Optional[dict]], List[int]]
    ] = None,
    interaction_constraint_params: Optional[dict] = None,
    proximity_function: Callable = _piecewise_proximity_function,
    **proximity_function_kwargs,
) -> List[List[pd.DataFrame]]:

    num_foragers = len(foragers)
    num_frames = len(foragers[0])
    predictor = copy.deepcopy(local_windows)

    for f in range(num_foragers):
        for t in range(num_frames):
            if predictor[f][t] is not None:
                # add column for predictor_name
                predictor[f][t][predictor_name] = 0
                # find confocals within interaction length
                interaction_partners = filter_by_distance(
                    foragersDF,
                    f,
                    t,
                    interaction_length,
                    interaction_constraint,
                    interaction_constraint_params,
                )

                if len(interaction_partners) > 0:
                    for partner in interaction_partners:
                        partner_x = foragers[partner]["x"].iloc[t]
                        partner_y = foragers[partner]["y"].iloc[t]

                        predictor[f][t][
                            predictor_name
                        ] += _proximity_predictor_contribution(
                            partner_x,
                            partner_y,
                            local_windows[f][t],
                            proximity_function,
                            **proximity_function_kwargs,
                        )

                # scaling to abs max (not sum, as this would lead to small numerical values)
                max_abs_over_grid = predictor[f][t][predictor_name].abs().max()  # sum()
                if max_abs_over_grid > 0:
                    predictor[f][t][predictor_name] = (
                        predictor[f][t][predictor_name] / max_abs_over_grid
                    )

    return predictor


def generate_proximity(foragers_object: dataObject, predictor_name: str):

    params = foragers_object.predictor_kwargs[predictor_name]

    predictor = _proximity_predictor(
        foragers_object.foragers,
        foragers_object.foragersDF,
        foragers_object.local_windows,
        predictor_name=predictor_name,
        **params,
    )

    return predictor
