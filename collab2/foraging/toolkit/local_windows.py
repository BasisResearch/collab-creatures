from itertools import product
import pandas as pd
import numpy as np
from typing import Callable, Any
 
def get_grid(grid_size : int =90, sampling_fraction : int =1, random_seed : int =0, grid_constraint: Callable[[pd.DataFrame, pd.DataFrame, Any], pd.DataFrame] =None, grid_constraint_params : dict = None):
   #generate grid of all points
    grid = list(product(range(1, grid_size+ 1), repeat=2)) 
    grid =pd.DataFrame(grid, columns=["x", "y"])

    #only keep accessible points
    if grid_constraint is not None: 
        grid = grid[grid_constraint(grid["x"],grid["y"], **grid_constraint_params)]
    
    #subsample the grid
    np.random.seed(random_seed)
    drop_ind = np.random.choice(grid.index, int(len(grid)*(1-sampling_fraction)))
    grid = grid.drop(drop_ind)

    return grid