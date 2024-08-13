from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm


def add_velocity(
    foragers: List[pd.DataFrame], dt: int = 1
) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """
    A function to calculate velocity magnitude and direction from forager positions, and add them to forager DataFrames.
    Parameters :
        - foragers : list of DataFrames containing forager positions, grouped by forager index
        - dt : time interval (in frames) used to compute velocity
    Returns :
        - foragers : list of DataFrames containing forager positions + velocity magnitude and direction,
                    grouped by forager index
        - foragersDF : flattened DataFrame containing positions + velocity magnitude and direction for all foragers
    """
    for df in foragers:
        v_ID = f"v_dt={dt}"
        theta_ID = f"theta_dt={dt}"
        if v_ID in df.columns and theta_ID in df.columns:
            continue
        else:
            # define v_x(t) = (x(t+dt) - x(t))/dt
            # df.diff(period = -dt) computes (x(t) - x(t+dt)), hence need an extra minus sign
            v_x = -df["x"].diff(periods=-dt) / dt
            v_y = -df["y"].diff(periods=-dt) / dt
            df[v_ID] = np.sqrt(v_x**2 + v_y**2)
            df[theta_ID] = np.arctan2(v_y, v_x)

    return foragers, pd.concat(foragers)


def velocity_predictor_contribution(
    v_pref: float,
    theta_pref: float,
    x: int,
    y: int,
    grid: pd.DataFrame,
    sigma_v: float,
    sigma_t: float,
) -> pd.DataFrame:
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
