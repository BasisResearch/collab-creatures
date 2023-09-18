import numpy as np
import pandas as pd
import random
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)
import foraging_toolkit as ft


import pandas as pd
import numpy as np


def add_how_far_squared_scaled(sim):
    birds = sim.birds
    step_size_max = sim.step_size_max
    visibility_range = sim.visibility_range
    how_far = sim.visibility.copy()

    for b in range(sim.num_birds):
        for t in range(sim.num_frames - 1):
            x_new = int(birds[b]["x"][t + 1])
            y_new = int(birds[b]["y"][t + 1])

            # print(how_far[b][t].head(n=1))
            _hf = how_far[b][t]
            _hf["how_far_squared"] = (_hf["x"] - x_new) ** 2 + (
                _hf["y"] - y_new
            ) ** 2
            _hf["how_far_squared_scaled"] = (
                -_hf["how_far_squared"]
                / (2 * (sim.step_size_max + sim.visibility_range) ** 2)
                + 1
            )
            # print(how_far[b][t].head(n=1))

        how_far[b][sim.num_frames - 1]["how_far_squared"] = np.nan
        how_far[b][sim.num_frames - 1]["how_far_squared_scaled"] = np.nan

    sim.how_far = how_far

    birds_how_far = []
    for bird in range(sim.num_birds):
        birds_how_far.append(pd.concat(how_far[bird]))

    sim.how_farDF = pd.concat(birds_how_far)
