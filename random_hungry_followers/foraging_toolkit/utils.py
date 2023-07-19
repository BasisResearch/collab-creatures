from itertools import product

import pandas as pd
import numpy as np


# generating grid
def generate_grid(grid_size):
    grid = list(product(range(1, grid_size + 1), repeat=2))
    return pd.DataFrame(grid, columns=["x", "y"])


# remove rewards eaten by birds in proximity
def update_rewards(rewards, birds, num_birds, num_frames, grab_range):
    for t in range(1, num_frames):
        rewards.append(rewards[t - 1].copy())
        eaten = []

        for b in range(num_birds):
            eaten_b = rewards[t][
                (abs(rewards[t]["x"] - birds[b].iloc[t]["x"]) <= grab_range)
                & (abs(rewards[t]["y"] - birds[b].iloc[t]["y"]) <= grab_range)
            ].index.tolist()
            if eaten_b:
                eaten.extend(eaten_b)

        if eaten:
            rewards[t] = rewards[t].drop(eaten)

        rewards[t]["time"] = t
    rewards = rewards
    rewardsDF = pd.concat(rewards)

    return {"rewards": rewards, "rewardsDF": rewardsDF}
