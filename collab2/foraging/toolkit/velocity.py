import warnings
from typing import List

import numpy as np
import pandas as pd


# a function that calculates the velocity magnitude and direction and outputs updated (foragers, foragersDF)
def add_velocity(foragers: List[pd.DataFrame], dt: int = 1):
    for df in foragers:
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

    return foragers, pd.concat(foragers)
