import pandas as pd
from typing import List

def add_velocities_to_foragers(foragers: List[pd.DataFrame]) -> None:
    
    for forager in foragers:        
        forager['velocity_x'] = forager['x'].diff().fillna(0)
        forager['velocity_y'] = forager['y'].diff().fillna(0)

