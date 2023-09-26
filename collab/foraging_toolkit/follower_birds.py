import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s:  %(message)s")


from .visibility import construct_visibility
from .proximity import generate_proximity_score
from .utils import update_rewards

import pandas as pd
import numpy as np
import warnings

from itertools import product

import pandas as pd
import numpy as np

# import foraging_toolkit as ft


def add_follower_birds(
    sim,
    num_follower_birds=3,
    visibility_range=10,
    getting_worse=1.5,
    optimal=4,
    proximity_decay=1,
):
    """
        A function to add follower birds to a simulation.

    Args:
        sim (Birds): A Birds object.

    Returns:
        sim (Birds): The same Birds object, but with follower birds added.
    """

    old_birds = sim.birds.copy()

    # TODO Check if different bird types mix well
    how_many_birds_already = len(old_birds)

    new_birds = sim.generate_random_birds(num_follower_birds, size=1)["random_birds"]

    for new_bird in new_birds:
        new_bird["bird"] = new_bird["bird"] + how_many_birds_already
        new_bird["type"] = "follower"

    for t in range(0, sim.num_frames):
        if t > 0 and t % 10 == 0:
            logging.info(f"Generating frame {t}/{sim.num_frames} ")
        # change to num frames
        _vis = construct_visibility(
            new_birds,
            sim.grid_size,
            visibility_range=visibility_range,
            start=t,
            end=t + 1,
        )["visibility"]

        _prox = generate_proximity_score(
            new_birds,
            _vis,
            visibility_range=visibility_range,
            getting_worse=getting_worse,
            optimal=optimal,
            proximity_decay=proximity_decay,
            start=0,
            end=1,
        )["proximity"]

        for b in range(num_follower_birds):
            options = _vis[b][0].copy()
            options = options.merge(_prox[b][0], how="inner")
            options.sort_values(by="proximity", ascending=False, inplace=True)
            options = options.head(10)
            chosen_option = options.iloc[np.random.randint(0, 10)]
            # chosen_option = options.head(0)

            if t < sim.num_frames - 1:
                new_x = chosen_option["x"]
                new_y = chosen_option["y"]

                new_row = {
                    "x": new_x,
                    "y": new_y,
                    "time": t + 2,
                    "bird": b + 1,
                    "type": "follower",
                }

                new_birds[b].loc[len(new_birds[b])] = new_row

    sim.birds.extend(new_birds)
    sim.birdsDF = pd.concat(sim.birds)

    rew = update_rewards(sim, sim.rewards, sim.birds, start=1)

    sim.rewards = rew["rewards"]
    sim.rewardsDF = rew["rewardsDF"]

    return sim
