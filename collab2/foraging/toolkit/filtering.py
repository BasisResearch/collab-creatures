import pandas as pd
import numpy as np
from typing import Callable, Any

def filter_by_distance(foragersDF : pd.Dataframe, f : int, t : int, interaction_length : float, interaction_constraint : Callable[[list,int,int,Any],list] = None, interaction_constraint_params : dict = None):
    positions = foragersDF[foragersDF["time"]==t].copy()
    distances 
    distances = (positions["x"] - positions["x"][f])**2 + (positions["y"] - positions["y"][f])**2
distances[f] = np.nan
forager_ind = positions.loc[distances<interaction_length**2, "forager"].tolist()
if interaction_constraint is not None:
    foragers_ind = interaction_constraint(forager_ind, f, t, foragersDF, **interaction_constraint_parameters)
return foragers_ind