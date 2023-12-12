from itertools import product

import pandas as pd


def object_from_data(birdsDF, rewardsDF):
    grid_max = max(max(birdsDF["x"]), max(birdsDF["y"]))  # TODO remove nested max
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


def generate_grid(grid_size):
    grid = list(product(range(1, grid_size + 1), repeat=2))
    return pd.DataFrame(grid, columns=["x", "y"])
