import sys


import pandas as pd
import numpy as np
import warnings

from itertools import product

import pandas as pd
import numpy as np

import foraging_toolkit as ft


def add_hungry_birds(
    sim,
    num_hungry_birds=3,
    rewards_decay=0.5,
    visibility_range=10,
):
    """
        A function to add hungry birds to a simulation.

    Args:
        sim (Birds): A Birds object.

    Returns:
        sim (Birds): The same Birds object, but with hungry birds added.
    """

    old_birds = sim.birds.copy()

    # TODO Check if different bird types mix well
    how_many_birds_already = len(old_birds)

    new_birds = sim.generate_random_birds(num_hungry_birds, size=1)["random_birds"]

    for new_bird in new_birds:
        new_bird["bird"] = new_bird["bird"] + how_many_birds_already
        new_bird["type"] = "hungry"

    for t in range(0, 1):  # change to num frames
        _vis = ft.construct_visibility(
            new_birds,
            sim.grid_size,
            visibility_range=visibility_range,
            start=t,
            end=t + 1,
        )["visibility"]

        sim.rewards = ft.update_rewards(
            sim, sim.rewards, new_birds, start=t, end=t + 1
        )["rewards"]

        sim.traces = ft.rewards_to_trace(
            sim.rewards,
            sim.grid_size,
            sim.num_frames,
            rewards_decay,
        )["traces"]

        for b in range(num_hungry_birds):
            options = _vis[b][t].copy()
            options = options.merge(sim.traces[t], how="inner")
            options.sort_values(by="trace", ascending=False, inplace=True)
            # options = options.head(4)
            # chosen_option = options.iloc[np.random.randint(0, 4)]
            chosen_option = options.head(1)
            print("at ", t, "bird", b, "chose \n", chosen_option)
            if t < sim.num_frames - 1:
                new_birds[b].loc[new_birds[b]["time"] == t + 1, "x"] = chosen_option[
                    "x"
                ]
                new_birds[b].loc[new_birds[b]["time"] == t + 1, "y"] = chosen_option[
                    "y"
                ]
    sim.birds.extend(new_birds)
    sim.birdsDF = pd.concat(sim.birds)

    rew = ft.update_rewards(sim, sim.rewards, sim.birds, start=1)

    sim.rewards = rew["rewards"]
    sim.rewardsDF = rew["rewardsDF"]

    tr = ft.rewards_to_trace(
        sim.rewards,
        sim.grid_size,
        sim.num_frames,
        rewards_decay,
    )

    sim.traces = tr["traces"]
    sim.tracesDF = pd.concat(sim.traces)

    vis = ft.construct_visibility(
        sim.birds, sim.grid_size, visibility_range=visibility_range
    )

    sim.visibility = vis["visibility"]
    sim.visibilityDF = vis["visibilityDF"]

    return sim


#
# for b in range(num_hungry_birds):
#     options = sim.visibility[b + how_many_birds_already][t - 1].copy()
#     options = options.merge(sim.traces[t - 1], how="inner")
#     options.sort_values(by="trace", ascending=False, inplace=True)
#     # options = options.head(4)
#     # chosen_option = options.iloc[np.random.randint(0, 4)]
#     chosen_option = options.head(1)
#     print("at ", t, "bird", b, "chose \n",
#           chosen_option)
#     new_birds[b].loc[new_birds[b]["time"] == t, "x"] = chosen_option["x"]
#     new_birds[b].loc[new_birds[b]["time"] == t, "y"] = chosen_option["y"]


# if old_birds:
#     sim.birds = old_birds.extend(new_birds)
# else:
#     sim.birds = new_birds

# sim.birdsDF = pd.concat(sim.birds)

#    return sim

# sim.birds.extend(new_birds)

# ft.update_rewards(sim, sim.birds, start=1)

# tr = ft.rewards_to_trace(
#     sim.rewards,
#     sim.grid_size,
#     sim.num_frames,
#     rewards_decay,
# )
# sim.traces = tr["traces"]
# sim.tracesDF = pd.concat(sim.traces)

# vis = ft.construct_visibility(
#     sim.birds, sim.grid_size, visibility_range=visibility_range
# )
# sim.visibility_range = visibility_range
# sim.visibility = vis["visibility"]
# #    sim.visibilityDF = vis["visibilityDF"]

# for t in range(1, sim.num_frames):
#     for b in range(
#         how_many_birds_already, how_many_birds_already + num_hungry_birds
#     ):
#         options = sim.visibility[b][t - 1].copy()
#         options = options.merge(sim.traces[t - 1], how="inner")
#         options.sort_values(by="trace", ascending=False, inplace=True)
#         # options = options.head(4)
#         # chosen_option = options.iloc[np.random.randint(0, 4)]
#         chosen_option = options.head(1)
#         print("at ", t, "bird", b, "chose \n",
#               chosen_option)
#         sim.birds[b].loc[sim.birds[b]["time"] == t, "x"] = chosen_option["x"]
#         sim.birds[b].loc[sim.birds[b]["time"] == t, "y"] = chosen_option["y"]

#         vis = ft.construct_visibility(
#             sim.birds, sim.grid_size, visibility_range=visibility_range
#         )
#         sim.visibility = vis["visibility"]

# ft.update_rewards(sim, sim.birds, start=1)

# trf = ft.rewards_to_trace(
#     sim.rewards,
#     sim.grid_size,
#     sim.num_frames,
#     rewards_decay,
# )
# tr = trf["traces"]

# sim.visibility_range = visibility_range
# sim.visibility = vis["visibility"]
# sim.visibilityDF = vis["visibilityDF"]

# _tr = tr["traces"].copy()

# for b in range(num_hungry_birds):
#     for t in range(1, sim.num_frames):
#         options = _vis[b][t - 1].copy()
#         options = options.merge(_tr[t - 1], how="inner")
#         options.sort_values(by="trace", ascending=False, inplace=True)
#         options = options.head(5)
#         chosen_option = options.iloc[np.random.randint(0, 5)]
#         new_birds[b].loc[new_birds[b]["time"] == t, "x"] = chosen_option["x"]
#         new_birds[b].loc[new_birds[b]["time"] == t, "y"] = chosen_option["y"]

#         joint_birds = old_birds + new_birds

#         ft.update_rewards(sim, joint_birds)

#         _tr = ft.rewards_to_trace(
#             sim.rewards,
#             sim.grid_size,
#             sim.num_frames,
#             rewards_decay,
#         )["traces"]

#         _vis = ft.construct_visibility(
#             new_birds, sim.grid_size, visibility_range=visibility_range
#         )["visibility"]

# sim.birds = joint_birds

# ft.update_rewards(sim, sim.birds, start=1)

# tr = ft.rewards_to_trace(
#     sim.rewards,
#     sim.grid_size,
#     sim.num_frames,
#     rewards_decay,
# )

# sim.traces = tr["traces"]
# sim.tracesDF = pd.concat(sim.traces)
# # sim.tracesDF = tr["tracesDF"]

# vis = ft.construct_visibility(
#     sim.birds, sim.grid_size, visibility_range=visibility_range
# )
# sim.visibility_range = visibility_range
# sim.visibility = vis["visibility"]
# sim.visibilityDF = vis["visibilityDF"]
# # sim.visibilityDF = vis["visibilityDF"]

# return sim
