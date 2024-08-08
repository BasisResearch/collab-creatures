from typing import List

import numpy as np
import pandas as pd


# a function that calculates the velocity magnitude and direction and outputs updated (foragers, foragersDF)
def add_velocity(foragers: List[pd.DataFrame], dt: int = 1):
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


# a function that takes in the "preferred" (v, theta) (set by any mechanism), current position,
# and calculates a predictor score over all grid points using polar gaussians of width (sigma_v, sigma_t)
def velocity_predictor_contribution(
    v_pref: float,
    theta_pref: float,
    x: int,
    y: int,
    grid: pd.DataFrame,
    sigma_v: float,
    sigma_t: float,
):
    """
    Requires theta in [-pi,pi)
    """

    v_implied = np.sqrt((grid["x"] - x) ** 2 + (grid["y"] - y) ** 2)
    theta_implied = np.arctan2(grid["y"] - y, grid["x"] - x)

    d_v = v_implied - v_pref
    # there is a discontinuity when taking the difference of angles (2pi \equiv 0 !),
    # so always choose the smaller difference
    d_theta = theta_implied - theta_pref
    d_theta[d_theta > np.pi] += -2 * np.pi
    d_theta[d_theta < -np.pi] += 2 * np.pi

    P_v = np.exp(-(d_v**2) / (2 * sigma_v**2)) / (np.sqrt(2 * np.pi) * sigma_v)
    P_theta = np.exp(-(d_theta**2) / (2 * sigma_t**2)) / (np.sqrt(2 * np.pi) * sigma_t)

    return P_v * P_theta
