from itertools import product

import pandas as pd


def object_from_data(
    foragersDF,
    grid_size,
    frames=None,
    rewardsDF=None,
    calculate_step_size_max=False,
):
    if frames is None:
        frames = foragersDF["time"].nunique()

    class EmptyObject:
        pass

    sim = EmptyObject()

    sim.grid_size = grid_size
    sim.num_frames = frames
    sim.foragersDF = foragersDF
    sim.foragers = [group for _, group in foragersDF.groupby("forager")]

    if rewardsDF is not None:
        sim.rewardsDF = rewardsDF
        sim.rewards = [group for _, group in rewardsDF.groupby("time")]

    if sim.foragersDF["forager"].min() == 0:
        sim.foragersDF["forager"] = sim.foragersDF["forager"] + 1

    sim.num_foragers = len(sim.foragers)

    # TODO: this doesn't work for foragers that run away
    if calculate_step_size_max:
        step_maxes = []

        for b in range(len(sim.foragers)):
            df = sim.foragers[b]
            step_maxes.append(
                max(
                    max(
                        [
                            abs(df["x"].iloc[t + 1] - df["x"].iloc[t])
                            for t in range(len(df) - 1)
                        ]
                    ),
                    max(
                        [
                            abs(df["y"].iloc[t + 1] - df["y"].iloc[t])
                            for t in range(len(df) - 1)
                        ]
                    ),
                )
            )
        sim.step_size_max = max(step_maxes)

    return sim


# tentatively suspended to test the new version that works with real life foragers
# def object_from_data(foragersDF, rewardsDF):
#     grid_max = max(max(foragersDF["x"]), max(foragersDF["y"]))  # TODO remove nested max
#     maxes = [max(foragersDF["time"]), max(rewardsDF["time"])]
#     limit = min(maxes)
#     foragersDF = foragersDF[foragersDF["time"] <= limit]
#     rewardsDF = rewardsDF[rewardsDF["time"] <= limit]

#     class EmptyObject:
#         pass

#     sim = EmptyObject()

#     sim.grid_size = int(grid_max)
#     sim.num_frames = int(limit)
#     sim.foragersDF = foragersDF
#     sim.rewardsDF = rewardsDF
#     sim.foragers = [group for _, group in foragersDF.groupby("forager")]
#     sim.rewards = [group for _, group in rewardsDF.groupby("time")]
#     sim.num_foragers = len(sim.foragers)

#     step_maxes = []
#     for b in range(len(sim.foragers)):
#         step_maxes.append(
#             max(
#                 max(
#                     [
#                         abs(sim.foragers[b]["x"][t + 1] - sim.foragers[b]["x"][t])
#                         for t in range(sim.num_frames - 1)
#                     ]
#                 ),
#                 max(
#                     [
#                         abs(sim.foragers[b]["y"][t + 1] - sim.foragers[b]["y"][t])
#                         for t in range(sim.num_frames - 1)
#                     ]
#                 ),
#             )
#         )

#     sim.step_size_max = max(step_maxes)

#     return sim


def generate_grid(grid_size):
    grid = list(product(range(1, grid_size + 1), repeat=2))
    return pd.DataFrame(grid, columns=["x", "y"])


# remove rewards eaten by foragers in proximity
def update_rewards(sim, rewards, foragers, start=1, end=None):
    if end is None:
        end = foragers[0].shape[0]

    for t in range(start, end):
        # rewards.append(rewards[t - 1].copy())
        rewards[t] = rewards[t - 1].copy()
        eaten = []

        for b in range(len(foragers)):
            eaten_b = rewards[t][
                (abs(rewards[t]["x"] - foragers[b].iloc[t]["x"]) <= sim.grab_range)
                & (abs(rewards[t]["y"] - foragers[b].iloc[t]["y"]) <= sim.grab_range)
            ].index.tolist()
            if eaten_b:
                eaten.extend(eaten_b)

        if eaten:
            rewards[t] = rewards[t].drop(eaten)

        rewards[t]["time"] = t + 1

    rewardsDF = pd.concat(rewards)

    return {"rewards": rewards, "rewardsDF": rewardsDF}
