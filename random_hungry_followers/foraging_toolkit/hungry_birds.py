import sys

sys.path.insert(0, "..")

import foraging_toolkit as ft

import pandas as pd
import numpy as np
import warnings

from itertools import product

import pandas as pd
import numpy as np


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

    new_birds = sim.generate_random_birds(num_hungry_birds)["random_birds"]

    for new_bird in new_birds:
        new_bird["bird"] = new_bird["bird"] + how_many_birds_already
        new_bird["type"] = "hungry"

    sim.birds.extend(new_birds)

    ft.update_rewards(sim, sim.birds, start=1)

    tr = ft.rewards_to_trace(
        sim.rewards,
        sim.grid_size,
        sim.num_frames,
        rewards_decay,
    )

    vis = ft.construct_visibility(
        sim.birds, sim.grid_size, visibility_range=visibility_range
    )
    sim.visibility_range = visibility_range
    sim.visibility = vis["visibility"]
    sim.visibilityDF = vis["visibilityDF"]

    _tr = tr["traces"].copy()

    for t in range(1, sim.num_frames):
        _new_vis = ft.construct_visibility(
            new_birds, sim.grid_size, visibility_range=visibility_range, end=t + 1
        )["visibility"]

        for b in range(num_hungry_birds):
            options = _new_vis[b][t - 1].copy()
            options = options.merge(_tr[t - 1], how="inner")
            options.sort_values(by="trace", ascending=False, inplace=True)
            options = options.head(5)
            chosen_option = options.iloc[np.random.randint(0, 5)]
            new_birds[b][t] = chosen_option.values

        joint_birds = old_birds + new_birds
        ft.update_rewards(sim, joint_birds, start=1, stop=t)

        _tr = ft.rewards_to_trace(
            sim.rewards,
            sim.grid_size,
            sim.num_frames,
            rewards_decay,
        )["traces"]

    sim.birds = joint_birds
    sim.birdsDF = pd.concat(sim.birds)

    # tr = ft.rewards_to_trace(
    #         sim.rewards,
    #         sim.grid_size,
    #         sim.num_frames,
    #         rewards_decay,
    #     )

    sim.traces = _tr
    sim.tracesDF = pd.concat(sim.traces)

    vis = ft.construct_visibility(
        sim.birds, sim.grid_size, visibility_range=visibility_range
    )
    sim.visibility_range = visibility_range
    sim.visibility = vis["visibility"]
    sim.visibilityDF = vis["visibilityDF"]

    return sim
