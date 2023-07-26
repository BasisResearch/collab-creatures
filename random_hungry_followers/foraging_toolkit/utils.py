from itertools import product

import pandas as pd
import numpy as np


# generating grid
def generate_grid(grid_size):
    grid = list(product(range(1, grid_size + 1), repeat=2))
    return pd.DataFrame(grid, columns=["x", "y"])


# remove rewards eaten by birds in proximity
def update_rewards(sim, birds, start=1, stop=None):
    if stop is None:
        stop = birds[0].shape[0]

    for t in range(start, stop):
        sim.rewards.append(sim.rewards[t - 1].copy())
        eaten = []

        for b in range(len(birds)):
            eaten_b = sim.rewards[t][
                (abs(sim.rewards[t]["x"] - birds[b].iloc[t]["x"]) <= sim.grab_range)
                & (abs(sim.rewards[t]["y"] - birds[b].iloc[t]["y"]) <= sim.grab_range)
            ].index.tolist()
            if eaten_b:
                eaten.extend(eaten_b)

        if eaten:
            sim.rewards[t] = sim.rewards[t].drop(eaten)

        sim.rewards[t]["time"] = t + 1

    sim.rewardsDF = pd.concat(sim.rewards)
