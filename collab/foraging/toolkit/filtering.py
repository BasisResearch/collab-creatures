import copy
from typing import Callable, List, Optional

import numpy as np
import pandas as pd


def filter_by_distance(
    foragersDF: pd.DataFrame,
    f: int,
    t: int,
    interaction_length: float,
    interaction_constraint: Optional[Callable] = None,  # TODO add more type hints?
    interaction_minimal_distance: float = 0,
    **interaction_constraint_params,
) -> List[int]:
    """
    Filters and returns a list of foragers that are within a specified distance of a given forager at a particular time.

    This function calculates the distances between a specified forager and all other foragers at a given time step
    and returns the indices of foragers that fall within the specified `interaction_length`. An optional interaction
    constraint can be applied to further filter the resulting foragers based on custom logic.

    :param foragersDF: A pandas DataFrame containing the foragers' positions and times. Must include columns 'x',
                       'y', 'forager', and 'time'
    :param f: The index of the forager whose neighbors are being filtered
    :param t: The time step at which the filtering is performed.
    :param interaction_length: The maximum distance within which foragers are considered neighbors.
    :param interaction_minimal_distance: The minimum distance at which foragers are considered neighbors.
    :param interaction_constraint: An optional callable that imposes additional filtering criteria. The callable should
                                   take in the list of forager indices, the current forager index `f`, time `t`, the
                                   full foragers DataFrame, and any optional parameters.
    :param interaction_constraint_params: Optional parameters passed to the interaction constraint function.

    :return: A list of forager indices that are within the specified interaction length of forager `f` at time `t`,
             possibly further filtered by the interaction constraint
    """
    interaction_constraint_params = interaction_constraint_params or {}

    positions = copy.deepcopy(foragersDF[foragersDF["time"] == t])
    positions["distance"] = np.sqrt(
        (positions["x"] - positions.loc[positions["forager"] == f, "x"].values) ** 2
        + (positions["y"] - positions.loc[positions["forager"] == f, "y"].values) ** 2
    )
    positions.loc[positions["forager"] == f, "distance"] = np.nan

    mask = (positions["distance"] > interaction_minimal_distance) & (
        positions["distance"] <= interaction_length
    )

    foragers_ind = positions.loc[mask, "forager"].tolist()

    if interaction_constraint is not None:
        foragers_ind = interaction_constraint(
            foragers_ind, f, t, foragersDF, **interaction_constraint_params
        )
    return foragers_ind


def constraint_filter_nearest(
    f_ind: List[int], f: int, t: int, foragersDF: pd.DataFrame, **params
) -> List[int]:
    """
    Filters and returns the index of the nearest forager to a given forager at a specified time step.

    This function identifies the nearest forager from a list of candidate foragers by calculating their
    Euclidean distances to the focal forager `f` at time step `t`. The forager with the smallest distance
    is returned.

    :param f_ind: A list of forager indices to consider as potential neighbors
    :param f: The index of the focal forager
    :param t: The time step at which to compute distances
    :param foragersDF: A pandas DataFrame containing forager positions and times. The DataFrame must include
                       columns 'x', 'y', 'forager', and 'time'
    :param params: Additional parameters that may be required by this or other filter functions. Not used in
                   this implementation but included for compatibility with the filtering pipeline

    :return: A list containing the index of the nearest forager to forager `f` at time `t`
    """
    current_positions = foragersDF.loc[
        np.logical_and(foragersDF["forager"].isin(f_ind), foragersDF["time"] == t)
    ]

    x_f = foragersDF.loc[
        np.logical_and(foragersDF["forager"] == f, foragersDF["time"] == t), "x"
    ].values[0]

    y_f = foragersDF.loc[
        np.logical_and(foragersDF["forager"] == f, foragersDF["time"] == t), "y"
    ].values[0]

    distances = np.sqrt(
        (current_positions["x"] - x_f) ** 2 + (current_positions["y"] - y_f) ** 2
    )

    return current_positions["forager"][distances == distances.min()].to_list()


def constraint_filter_close_to_reward(
    f_ind: List[int],
    f: int,
    t: int,
    foragersDF: pd.DataFrame,
    finders_tolerance: float,
    rewards: List[pd.DataFrame],
) -> List[int]:

    current_positions = foragersDF.loc[
        np.logical_and(foragersDF["forager"].isin(f_ind), foragersDF["time"] == t)
    ]

    on_reward = []

    for _, row in current_positions.iterrows():

        try:
            row_x = row["x"].item()
            row_y = row["y"].item()
        except AttributeError:
            row_x = row["x"]
            row_y = row["y"]

        on_reward.append(
            any(
                np.sqrt(
                    (row_x - rewards[int(t)]["x"]) ** 2
                    + (row_y - rewards[int(t)]["y"]) ** 2
                )
                < finders_tolerance
            )
        )

    f_ind_constr = [f for f, rew in zip(f_ind, on_reward) if rew]

    return [int(f) for f in f_ind_constr]
