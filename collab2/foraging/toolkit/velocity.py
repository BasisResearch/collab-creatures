import copy
import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from collab2.foraging.toolkit import filter_by_distance


def add_velocity(
    foragers: List[pd.DataFrame], dt: int
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    A function to calculate velocity magnitude and direction from forager positions, and add them to forager DataFrames.
    Parameters :
        - foragers : list of DataFrames containing forager positions, grouped by forager index
        - dt : time interval (in frames) used to compute velocity
    Returns :
        - foragers_processed : list of DataFrames containing forager positions + velocity magnitude and direction,
                    grouped by forager index
        - foragersDF_processed : flattened DataFrame containing positions + velocity magnitude and direction
                    for all foragers
    """
    foragers_processed = copy.deepcopy(foragers)
    for df in foragers_processed:
        v_ID = f"v_dt={dt}"
        theta_ID = f"theta_dt={dt}"
        if v_ID in df.columns and theta_ID in df.columns:
            warnings.warn(
                """Using existing velocity data.
                Delete corresponding columns from foragersDF to re-calculate velocity values."""
            )
            continue
        else:
            # define v_x(t) = (x(t) - x(t-dt))/dt
            v_x = df["x"].diff(periods=dt) / dt
            v_y = df["y"].diff(periods=dt) / dt
            df[v_ID] = np.sqrt(v_x**2 + v_y**2)
            df[theta_ID] = np.arctan2(v_y, v_x)

    return foragers_processed, pd.concat(foragers_processed)


def velocity_predictor_contribution(
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
    Parameters:
        - v_pref : Preferred velocity magnitude
        - theta_pref : Preferred velocity direction. Must be specified as an angle in [-pi,pi)
        - x : current x position of focal forager
        - y : current y position of focal forager
        - grid : grid to compute predictor scores over.
                For most applications, this would be the relevant `local_windows` for the focal forager
        - sigma_v : standard deviation of Gaussian for velocity magnitude
        - sigma_t : standard deviation of Gaussian for velocity direction
    Returns:
        - calculated predictor scores for each grid point returned as a DataFrame
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


def _generate_pairwise_copying(
    foragers: List[pd.DataFrame],
    foragersDF: pd.DataFrame,
    local_windows: List[List[pd.DataFrame]],
    predictor_ID: str,
    interaction_length: float,
    dt: int,
    sigma_v: float,
    sigma_t: float,
    interaction_constraint: Optional[
        Callable[[List[int], int, int, pd.DataFrame, Optional[dict]], List[int]]
    ] = None,
    interaction_constraint_params: Optional[dict] = None,
) -> List[List[pd.DataFrame]]:
    """
    A function that calculates the predictor scores associated with random, pairwise velocity copying to all foragers.
    Parameters:
        - foragers : List of DataFrames containing forager positions and velocities grouped by forager index
        - foragersDF : Flattened DataFrame of forager positions and velocities
        - local_windows : Nested list of DataFrames containing grid points to compute predictor over,
            grouped by forager index and time
        - predictorID : Name given to column containing predictor scores in `predictor`
        - interaction_length : Maximum inter-forager distance for velocity copying interaction
        - dt : frames skipped in calculation of velocities
            ** Note: This function requires `foragers` and `foragersDF` to contain 
                columns "v_dt={dt}", "theta_dt={dt}" **
        - sigma_v : standard deviation of Gaussian for velocity magnitude
        - sigma_t : standard deviation of Gaussian for velocity direction
        - interaction_constraint : Optional function to model other interaction constraints
        - interaction_constraint_params : Optional dictionary of parameters to be passed to `interaction_constraint`
    Returns:
        - predictor : Nested list of calculated predictor scores, grouped by foragers and time
    """

    num_foragers = len(foragers)
    num_frames = len(foragers[0])
    predictor = copy.deepcopy(local_windows)

    for f in range(num_foragers):
        for t in range(num_frames):
            if predictor[f][t] is not None:
                # add column for predictor_ID
                predictor[f][t][predictor_ID] = 0
                # find confocals within interaction length
                interaction_partners = filter_by_distance(
                    foragersDF,
                    f,
                    t,
                    interaction_length,
                    interaction_constraint,
                    interaction_constraint_params,
                )
                # additively combine their influence
                x = foragers[f].loc[t, "x"]
                y = foragers[f].loc[t, "y"]
                valid_partners = 0  # number of partners with finite v,theta

                for f_i in interaction_partners:
                    v_pref = foragers[f_i].loc[t, f"v_dt={dt}"]
                    theta_pref = foragers[f_i].loc[t, f"theta_dt={dt}"]
                    if np.isfinite(v_pref) and np.isfinite(theta_pref):
                        valid_partners += 1
                        predictor[f][t][
                            predictor_ID
                        ] += velocity_predictor_contribution(
                            v_pref, theta_pref, x, y, predictor[f][t], sigma_v, sigma_t
                        )

                # finally, normalize by number of valid interaction partners
                if valid_partners > 0:
                    predictor[f][t][predictor_ID] = (
                        predictor[f][t][predictor_ID] / valid_partners
                    )

    return predictor
