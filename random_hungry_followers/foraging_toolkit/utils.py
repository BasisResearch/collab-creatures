from itertools import product

import pandas as pd
import numpy as np


def object_from_data(birdsDF, rewardsDF):
    grid_max = max(max(birdsDF["x"]), max(birdsDF["y"]))
    maxes = [max(birdsDF["time"]), max(rewardsDF["time"])]
    limit = min(maxes)
    birdsDF = birdsDF[birdsDF["time"] <= limit]
    rewardsDF = rewardsDF[rewardsDF["time"] <= limit]

    class EmptyObject:
        pass

    sim = EmptyObject()

    sim.grid_size = int(grid_max)
    sim.num_frames = int(limit)
    sim.birdsDF = birdsDF
    sim.rewardsDF = rewardsDF
    sim.birds = [group for _, group in birdsDF.groupby("bird")]
    sim.rewards = [group for _, group in rewardsDF.groupby("time")]
    sim.num_birds = len(sim.birds)

    step_maxes = []
    for b in range(len(sim.birds)):
        step_maxes.append(
            max(
                max(
                    [
                        abs(sim.birds[b]["x"][t + 1] - sim.birds[b]["x"][t])
                        for t in range(sim.num_frames - 1)
                    ]
                ),
                max(
                    [
                        abs(sim.birds[b]["y"][t + 1] - sim.birds[b]["y"][t])
                        for t in range(sim.num_frames - 1)
                    ]
                ),
            )
        )

    sim.step_size_max = max(step_maxes)

    return sim


# generating grid
def generate_grid(grid_size):
    grid = list(product(range(1, grid_size + 1), repeat=2))
    return pd.DataFrame(grid, columns=["x", "y"])


# remove rewards eaten by birds in proximity
def update_rewards(sim, rewards, birds, start=1, end=None):
    if end is None:
        end = birds[0].shape[0]

    for t in range(start, end):
        # rewards.append(rewards[t - 1].copy())
        rewards[t] = rewards[t - 1].copy()
        eaten = []

        for b in range(len(birds)):
            eaten_b = rewards[t][
                (abs(rewards[t]["x"] - birds[b].iloc[t]["x"]) <= sim.grab_range)
                & (abs(rewards[t]["y"] - birds[b].iloc[t]["y"]) <= sim.grab_range)
            ].index.tolist()
            if eaten_b:
                eaten.extend(eaten_b)

        if eaten:
            rewards[t] = rewards[t].drop(eaten)

        rewards[t]["time"] = t + 1

    rewardsDF = pd.concat(rewards)

    return {"rewards": rewards, "rewardsDF": rewardsDF}
