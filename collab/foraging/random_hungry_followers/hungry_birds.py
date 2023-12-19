import logging

import numpy as np
import pandas as pd

from collab.foraging.toolkit import (
    construct_visibility,
    rewards_to_trace,
    update_rewards,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s:  %(message)s")


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

        sim.rewards = update_rewards(sim, sim.rewards, new_birds, start=t, end=t + 1)[
            "rewards"
        ]

        sim.traces = rewards_to_trace(
            sim.rewards,
            sim.grid_size,
            sim.num_frames,
            rewards_decay,
        )["traces"]

        for b in range(num_hungry_birds):
            options = _vis[b][0].copy()
            options = options.merge(sim.traces[t], how="inner")
            options.sort_values(by="trace", ascending=False, inplace=True)
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
                    "type": "hungry",
                }

                new_birds[b].loc[len(new_birds[b])] = new_row

    sim.birds.extend(new_birds)
    sim.birdsDF = pd.concat(sim.birds)

    rew = update_rewards(sim, sim.rewards, sim.birds, start=1)

    sim.rewards = rew["rewards"]
    sim.rewardsDF = rew["rewardsDF"]

    tr = rewards_to_trace(
        sim.rewards,
        sim.grid_size,
        sim.num_frames,
        rewards_decay,
    )

    sim.traces = tr["traces"]
    sim.tracesDF = pd.concat(sim.traces)

    vis = construct_visibility(
        sim.birds, sim.grid_size, visibility_range=visibility_range
    )

    sim.visibility = vis["visibility"]
    sim.visibilityDF = vis["visibilityDF"]

    return sim
