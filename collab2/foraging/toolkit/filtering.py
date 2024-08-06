from typing import Callable, List, Optional

import numpy as np
import pandas as pd


# PP_comment : even with passing constraint params as a dictionary, we might have a limitation : 
# we cannot have a constraint that depends on quantities derived within the predictor function (eg, fastest v)
# One workaround could be to locally (i.e. within the function) modify foragersDF with the derived quantities 
# and pass that to the constraint function via filter_by_distance
def filter_by_distance(
    foragersDF: pd.DataFrame,
    f: int,
    t: int,
    interaction_length: float,
    interaction_constraint: Optional[
        Callable[[List[int], int, int, pd.DataFrame, Optional[dict]], List[int]]
    ] = None,
    interaction_constraint_params: Optional[dict] = None,
) -> List[int]:
    positions = foragersDF[foragersDF["time"] == t].copy()
    positions["distance"] = np.sqrt(
        (positions["x"] - positions.loc[positions["forager"] == f, "x"].values) ** 2
        + (positions["y"] - positions.loc[positions["forager"] == f, "y"].values) ** 2
    )
    positions.loc[positions["forager"] == f, "distance"] = np.nan
    foragers_ind = positions.loc[
        positions["distance"] <= interaction_length, "forager"
    ].tolist()

    if interaction_constraint is not None:
        foragers_ind = interaction_constraint(
            foragers_ind, f, t, foragersDF, interaction_constraint_params
        )
    return foragers_ind
