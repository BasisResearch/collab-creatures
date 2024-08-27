import copy
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from collab2.foraging.toolkit import filter_by_distance
from collab2.foraging.toolkit.utils import dataObject  # noqa: F401


def _exponential_decay(distance: Union[float, np.ndarray], 
                      decay_factor: float = 0.5):
    return np.exp(-decay_factor * distance)


def _decaying_contribution(
    x_source: int,
    y_source: int,
    grid: pd.DataFrame,
    decay_function: Callable = _exponential_decay,
    **decay_function_kwargs,
) -> np.ndarray:
  

    distance_to_source = np.sqrt((grid["x"] - x_source) ** 2 + (grid["y"] - y_source) ** 2)
    proximity_score = decay_function(distance_to_source, **decay_function_kwargs)
    return proximity_score
