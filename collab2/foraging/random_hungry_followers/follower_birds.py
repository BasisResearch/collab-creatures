import logging

import numpy as np
import pandas as pd

from collab.foraging.toolkit import (
    construct_visibility,
    generate_proximity_score,
    update_rewards,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s:  %(message)s")


def add_follower_foragers(
    sim,
    num_follower_foragers=3,
    visibility_range=10,
    getting_worse=1.5,
    optimal=4,
    proximity_decay=1,
):
    """
        A function to add follower foragers to a simulation.

    Args:
        sim (foragers): A foragers object.

    Returns:
        sim (foragers): The same foragers object, but with follower foragers added.
    """

    old_foragers = sim.foragers.copy()

    # TODO Check if different forager types mix well
    how_many_foragers_already = len(old_foragers)

    new_foragers = sim.generate_random_foragers(num_follower_foragers, size=1)[
        "random_foragers"
    ]

    for new_forager in new_foragers:
        new_forager["forager"] = new_forager["forager"] + how_many_foragers_already
        new_forager["type"] = "follower"

    for t in range(0, sim.num_frames):
        if t > 0 and t % 10 == 0:
            logging.info(f"Generating frame {t}/{sim.num_frames} ")
        # change to num frames
        _vis = construct_visibility(
            new_foragers,
            sim.grid_size,
            visibility_range=visibility_range,
            start=t,
            end=t + 1,
        )["visibility"]

        _prox = generate_proximity_score(
            new_foragers,
            _vis,
            visibility_range=visibility_range,
            getting_worse=getting_worse,
            optimal=optimal,
            proximity_decay=proximity_decay,
            start=0,
            end=1,
        )["proximity"]

        for b in range(num_follower_foragers):
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
                    "time": t + 1,
                    "forager": b ,
                    "type": "follower",
                }

                new_foragers[b].loc[len(new_foragers[b])] = new_row

    sim.foragers.extend(new_foragers)
    sim.foragersDF = pd.concat(sim.foragers)

    rew = update_rewards(sim, sim.rewards, sim.foragers, start=1)

    sim.rewards = rew["rewards"]
    sim.rewardsDF = rew["rewardsDF"]

    return sim
