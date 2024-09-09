import copy
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from collab2.foraging.toolkit.filtering import filter_by_distance
from collab2.foraging.toolkit.utils import dataObject


def _add_velocity(
    foragers: List[pd.DataFrame], dt: int
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    A function to calculate velocity magnitude and direction from forager positions, and add them to forager DataFrames.
    :param foragers : list of DataFrames containing forager positions, grouped by forager index
    :param dt : time interval (in frames) used to compute velocity
    :return: tuple containing
        - list of DataFrames containing forager positions + computed velocity,
        grouped by forager index
        - flattened DataFrame obtained by concatenating aforementioned list
    """
    foragers_processed = copy.deepcopy(foragers)
    for df in foragers_processed:
        v_ID = f"v_dt={dt}"
        theta_ID = f"theta_dt={dt}"
        if v_ID in df.columns and theta_ID in df.columns:
            # warnings.warn(
            #     """Using existing velocity data.
            #     Delete corresponding columns from foragersDF to re-calculate velocity values."""
            # )
            # warnings.warn(
            #     """Using existing velocity data.
            #     Delete corresponding columns from foragersDF to re-calculate velocity values."""
            # )
            continue
        else:
            # define v_x(t) = (x(t) - x(t-dt))/dt
            v_x = df["x"].diff(periods=dt) / dt
            v_y = df["y"].diff(periods=dt) / dt
            df[v_ID] = np.sqrt(v_x**2 + v_y**2)
            df[theta_ID] = np.arctan2(v_y, v_x)

    return foragers_processed, pd.concat(foragers_processed)


def _velocity_predictor_contribution(
    v_pref: float,
    theta_pref: float,
    x: int,
    y: int,
    grid: pd.DataFrame,
    sigma_v: float,
    sigma_t: float,
) -> np.ndarray:
    """
    A function that calculates Gaussian predictor scores over a grid, given a preferred velocity magnitude/direction
    for the next time-step and the current position of the focal forager.
    :param v_pref : Preferred velocity magnitude
    :param theta_pref : Preferred velocity direction. Must be specified as an angle in [-pi,pi)
    :param x : current x position of focal forager
    :param y : current y position of focal forager
    :param grid : grid to compute predictor scores over.
                For most applications, this would be the relevant `local_windows` for the focal forager
    :param sigma_v : standard deviation of Gaussian for velocity magnitude
    :param sigma_t : standard deviation of Gaussian for velocity direction
    :return: calculated predictor scores for each grid point returned as a DataFrame
    """

    v_implied = np.sqrt((grid["x"] - x) ** 2 + (grid["y"] - y) ** 2)
    theta_implied = np.arctan2(grid["y"] - y, grid["x"] - x)
    P_v = norm.pdf(x=v_implied, loc=v_pref, scale=sigma_v)
    # there is a discontinuity when taking the difference of angles (2pi \equiv 0 !),
    # so always choose the smaller difference
    d_theta = theta_implied - theta_pref
    d_theta[d_theta > np.pi] += -2 * np.pi
    d_theta[d_theta < -np.pi] += 2 * np.pi
    P_theta = norm.pdf(x=d_theta, loc=0, scale=sigma_t)
    return P_v * P_theta


def _generic_velocity_predictor(
    foragers: List[pd.DataFrame],
    foragersDF: pd.DataFrame,
    local_windows: List[List[pd.DataFrame]],
    predictor_name: str,
    interaction_length: float,
    dt: int,
    sigma_v: float,
    sigma_t: float,
    transformation_function: Callable[[pd.DataFrame], pd.DataFrame],
    interaction_constraint: Optional[
        Callable[[List[int], int, int, pd.DataFrame], List[int]]
    ] = None,
    interaction_constraint_params: dict[str, Any] = {},
) -> List[List[pd.DataFrame]]:
    """
    A function that calculates predictor scores for arbitrary velocity alignment mechanisms, as specified by
    `transformation_function`. This function takes the velocities of interaction partners and outputs a transformation
    (eg, average, identity, max) as required by the corresponding mechanism
    Predictors are not calculated for frames where interaction partners have missing velocities.
    In this case, fraction of dropped frames is reported.
    Predictors are normalized by dividing by their max value for each forager & frame.

    :param foragers : List of DataFrames containing forager positions and velocities grouped by forager index
    :param foragersDF : Flattened DataFrame of forager positions and velocities
    :param local_windows : Nested list of DataFrames containing grid points to compute predictor over,
            grouped by forager index and time
    :param predictor_name : Name given to column containing predictor scores in `predictor`
    :param interaction_length : Maximum inter-forager distance for velocity copying interaction
    :param dt : frames skipped in calculation of velocities
            Note: This function requires `foragers` and `foragersDF` to contain
            columns "v_dt={dt}", "theta_dt={dt}"
    :param sigma_v : standard deviation of Gaussian for velocity magnitude
    :param sigma_t : standard deviation of Gaussian for velocity direction
    :param transformation_function : Function that implements a transformation of velocities of interaction partners,
            as stipulated by the chosen velocity alignment mechanism
    :param interaction_constraint : Optional function to model other interaction constraints
    :param interaction_constraint_params : Optional kwargs to be passed to `interaction_constraint`
    :return: Nested list of calculated predictor scores, grouped by foragers and time
    """

    num_foragers = len(foragers)
    num_frames = len(foragers[0])
    predictor = copy.deepcopy(local_windows)

    for f in range(num_foragers):
        for t in range(num_frames):
            if predictor[f][t] is not None:
                # add column for predictor_ID
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

                # check if all interaction partners have valid velocity values before computing predictor
                v_values = foragersDF.loc[
                    np.logical_and(
                        foragersDF["forager"].isin(interaction_partners),
                        foragersDF["time"] == t,
                    ),
                    [f"v_dt={dt}", f"theta_dt={dt}"],
                ]

                if v_values.notna().all(axis=None):
                    v_values = transformation_function(v_values)
                    x = foragers[f].loc[t, "x"]
                    y = foragers[f].loc[t, "y"]
                    # additively combine the influence of all confocals
                    for v_pref, theta_pref in v_values.itertuples(index=False):
                        predictor[f][t][
                            predictor_name
                        ] += _velocity_predictor_contribution(
                            v_pref, theta_pref, x, y, predictor[f][t], sigma_v, sigma_t
                        )
                else:
                    predictor[f][t][predictor_name] = np.nan

                # normalize predictor by dividing by max
                max_val = predictor[f][t][predictor_name].abs().max()
                if max_val > 0:
                    predictor[f][t][predictor_name] = (
                        predictor[f][t][predictor_name] / max_val
                    )

    return predictor


def generate_pairwiseCopying_predictor(
    foragers_object: dataObject, predictor_name: str
):
    """
    A function that calculates the predictor scores associated with random, pairwise velocity copying,
    by specifying an identity transformation to `_generic_velocity_predictor`.
    The necessary parameters from `foragers_object`. Thus, `foragers_object` must contain as attribute
    `predictor_kwargs` : dict, with `predictor_name` as a valid key.

    :param foragers_object : dataObject containing positional data and necessary kwargs
    :param predictorID : Name given to column containing predictor scores in `predictor`
    :return: Nested list of calculated predictor scores, grouped by foragers and time
    """

    # define transformation function
    def transformation_pairwiseCopying(v_values):
        return v_values

    # grab relevant parameters from foragers_object
    params = foragers_object.predictor_kwargs[predictor_name]

    # compute/add velocity
    foragers_object.foragers, foragers_object.foragersDF = _add_velocity(
        foragers_object.foragers, params["dt"]
    )

    # calculate predictor values
    predictor = _generic_velocity_predictor(
        foragers_object.foragers,
        foragers_object.foragersDF,
        foragers_object.local_windows,
        predictor_name,
        transformation_function=transformation_pairwiseCopying,
        **params,
    )

    return predictor


def generate_vicsek_predictor(foragers_object: dataObject, predictor_name: str):
    """
    A function that calculates the predictor scores associated with vicsek flocking,
    by specifying an averaging transformation to `_generic_velocity_predictor`.
    The necessary parameters are taken from `foragers_object`.
    Thus, `foragers_object` must contain as attribute
    `predictor_kwargs` : dict, with `predictor_name` as a valid key.

    :param foragers_object : dataObject containing positional data and necessary kwargs
    :param predictorID : Name given to column containing predictor scores in `predictor`
    :return: Nested list of calculated predictor scores, grouped by foragers and time
    """

    # define transformation function
    def transformation_vicsek(v_values):
        v_x = np.mean(v_values.iloc[:, 0] * np.cos(v_values.iloc[:, 1]))
        v_y = np.mean(v_values.iloc[:, 0] * np.sin(v_values.iloc[:, 1]))
        v_transformed = pd.DataFrame([[np.sqrt(v_x**2 + v_y**2), np.arctan2(v_y, v_x)]])
        return v_transformed

    # grab relevant parameters from foragers_object
    params = foragers_object.predictor_kwargs[predictor_name]

    # compute/add velocity
    foragers_object.foragers, foragers_object.foragersDF = _add_velocity(
        foragers_object.foragers, params["dt"]
    )

    # calculate predictor values
    predictor = _generic_velocity_predictor(
        foragers_object.foragers,
        foragers_object.foragersDF,
        foragers_object.local_windows,
        predictor_name,
        transformation_function=transformation_vicsek,
        **params,
    )

    return predictor


def _generate_velocityDiffusion_predictor(
    foragers: List[pd.DataFrame],
    local_windows: List[List[pd.DataFrame]],
    predictor_name: str,
    dt: int,
    sigma_v: float,
    sigma_t: float,
) -> List[List[pd.DataFrame]]:
    """
    A function that calculates predictor scores for angular/speed diffusion about the current velocity.
    This process models inertia in moving direction/ speed that can arise due to biological constraints.
    Predictors are modelled as 2D polar Gaussians, and normalized by dividing by
    their max value for each forager & frame.

    :param foragers : List of DataFrames containing forager positions and velocities grouped by forager index
    :param foragersDF : Flattened DataFrame of forager positions and velocities
    :param local_windows : Nested list of DataFrames containing grid points to compute predictor over,
            grouped by forager index and time
    :param predictor_name : Name given to column containing predictor scores in `predictor`
    :param dt : frames skipped in calculation of velocities
            Note: This function requires `foragers` and `foragersDF` to contain
            columns "v_dt={dt}", "theta_dt={dt}"
    :param sigma_v : standard deviation of Gaussian for velocity magnitude
    :param sigma_t : standard deviation of Gaussian for velocity direction
    :return: Nested list of calculated predictor scores, grouped by foragers and time
    """

    num_foragers = len(foragers)
    num_frames = len(foragers[0])
    predictor = copy.deepcopy(local_windows)

    for f in range(num_foragers):
        for t in range(num_frames):
            if predictor[f][t] is not None:
                # add column for predictor_ID
                predictor[f][t][predictor_name] = 0
                x = foragers[f].loc[t, "x"]
                y = foragers[f].loc[t, "y"]
                # center Gaussian about current velocity
                v_pref = foragers[f].loc[t, f"v_dt={dt}"]
                theta_pref = foragers[f].loc[t, f"theta_dt={dt}"]

                if np.isfinite(v_pref) and np.isfinite(theta_pref):
                    predictor[f][t][predictor_name] = _velocity_predictor_contribution(
                        v_pref, theta_pref, x, y, predictor[f][t], sigma_v, sigma_t
                    )
                else:
                    predictor[f][t][predictor_name] = np.nan

                # normalize predictor by dividing by max
                max_val = predictor[f][t][predictor_name].abs().max()
                if max_val > 0:
                    predictor[f][t][predictor_name] = (
                        predictor[f][t][predictor_name] / max_val
                    )

    return predictor


def generate_velocityDiffusion_predictor(
    foragers_object: dataObject, predictor_name: str
):
    """
    A function that calculates the predictor scores associated with velocity diffusion,
    by calling `_generate_velocityDiffusion_predictor` and grabbing necessary parameters
    from `foragers_object`. Thus, `foragers_object` must contain as attribute `predictor_kwargs` : dict,
    with `predictor_name` as a valid key.

    :param foragers_object : dataObject containing positional data and necessary kwargs
    :param predictorID : Name given to column containing predictor scores in `predictor`
    :return: Nested list of calculated predictor scores, grouped by foragers and time
    """

    # grab relevant parameters from foragers_object
    params = foragers_object.predictor_kwargs[predictor_name]

    # compute/add velocity
    foragers_object.foragers, foragers_object.foragersDF = _add_velocity(
        foragers_object.foragers, params["dt"]
    )

    # calculate predictor values
    predictor = _generate_velocityDiffusion_predictor(
        foragers_object.foragers,
        foragers_object.local_windows,
        predictor_name,
        **params,
    )

    return predictor
