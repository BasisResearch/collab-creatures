from itertools import product

import pandas as pd


def object_from_data(foragersDF, rewardsDF):
    grid_max = max(max(foragersDF["x"]), max(foragersDF["y"]))  # TODO remove nested max
    maxes = [max(foragersDF["time"]), max(rewardsDF["time"])]
    limit = min(maxes)
    foragersDF = foragersDF[foragersDF["time"] <= limit]
    rewardsDF = rewardsDF[rewardsDF["time"] <= limit]

    class EmptyObject:
        pass

    sim = EmptyObject()

    sim.grid_size = int(grid_max)
    sim.num_frames = int(limit)
    sim.foragersDF = foragersDF
    sim.rewardsDF = rewardsDF
    sim.foragers = [group for _, group in foragersDF.groupby("forager")]
    sim.rewards = [group for _, group in rewardsDF.groupby("time")]
    sim.num_foragers = len(sim.foragers)

    step_maxes = []
    for b in range(len(sim.foragers)):
        step_maxes.append(
            max(
                max(
                    [
                        abs(sim.foragers[b]["x"][t + 1] - sim.foragers[b]["x"][t])
                        for t in range(sim.num_frames - 1)
                    ]
                ),
                max(
                    [
                        abs(sim.foragers[b]["y"][t + 1] - sim.foragers[b]["y"][t])
                        for t in range(sim.num_frames - 1)
                    ]
                ),
            )
        )

    sim.step_size_max = max(step_maxes)

    return sim


def generate_grid(grid_size):
    grid = list(product(range(1, grid_size + 1), repeat=2))
    return pd.DataFrame(grid, columns=["x", "y"])
