import logging
import math
import numpy as np
import pandas as pd
from itertools import product

from collab.foraging.toolkit import (
    construct_visibility,
    rewards_to_trace,
    update_rewards,
)





def generate_grid(grid_size):
    grid = list(product(range(1, grid_size + 1), repeat=2))
    return pd.DataFrame(grid, columns=["x", "y"])

def visibility_vs_distance(distance, visibility_range):
    return math.cos((math.pi / visibility_range * distance) / 2)



def construct_visibility(
    foragers,
    grid_size,
    visibility_range,
    start=None,
    end=None,
    time_shift=0,
    grid=None,
):
    num_foragers = len(foragers)
    if start is None:
        start = 0

    if end is None:
        end = len(foragers[0])

    visibility = []

    for forager in range(num_foragers):
        ranges = []
        for frame in range(start, end):
            if grid is None:
                g = generate_grid(grid_size)
            else:
                g = grid.copy()

            g["distance"] = (
                (g["x"] - foragers[forager]["x"].iloc[frame]) ** 2
                + (g["y"] - foragers[forager]["y"].iloc[frame]) ** 2
            ) ** 0.5

            range_df = g[g["distance"] <= visibility_range].copy()
            range_df["distance_x"] = abs(
                range_df["x"] - foragers[forager]["x"].iloc[frame]
            )
            range_df["distance_y"] = abs(
                range_df["y"] - foragers[forager]["y"].iloc[frame]
            )
            range_df["visibility"] = range_df["distance"].apply(
                lambda d: visibility_vs_distance(d, visibility_range)
            )
            range_df["forager"] = forager + 1
            range_df["time"] = frame + 1

            range_df["time"] = range_df["time"] + time_shift
            ranges.append(range_df)

        visibility.append(ranges)

    foragers_visibilities = []
    for forager in range(num_foragers):
        foragers_visibilities.append(pd.concat(visibility[forager]))

    visibility_df = pd.concat(foragers_visibilities)

    return {"visibility": visibility, "visibilityDF": visibility_df}



logging.basicConfig(level=logging.INFO, format="%(asctime)s:  %(message)s")


def add_hungry_foragers(
    sim,
    num_hungry_foragers=3,
    rewards_decay=0.5,
    visibility_range=10,
):
    """
        A function to add hungry foragers to a simulation.

    Args:
        sim (foragers): A foragers object.

    Returns:
        sim (foragers): The same foragers object, but with hungry foragers added.
    """

    old_foragers = sim.foragers.copy()

    # TODO Check if different forager types mix well
    how_many_foragers_already = len(old_foragers)

    new_foragers = sim.generate_random_foragers(num_hungry_foragers, size=1)[
        "random_foragers"
    ]

    for new_forager in new_foragers:
        new_forager["forager"] = new_forager["forager"] + how_many_foragers_already
        new_forager["type"] = "hungry"

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

        sim.rewards = update_rewards(
            sim, sim.rewards, new_foragers, start=t, end=t + 1
        )["rewards"]

        sim.traces = rewards_to_trace(
            sim.rewards,
            sim.grid_size,
            sim.num_frames,
            rewards_decay,
        )["traces"]

        for b in range(num_hungry_foragers):
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
                    "time": t + 1,
                    "forager": b ,
                    "type": "hungry",
                }

                new_foragers[b].loc[len(new_foragers[b])] = new_row

    sim.foragers.extend(new_foragers)
    sim.foragersDF = pd.concat(sim.foragers)

    rew = update_rewards(sim, sim.rewards, sim.foragers, start=1)

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
        sim.foragers, sim.grid_size, visibility_range=visibility_range
    )

    sim.visibility = vis["visibility"]
    sim.visibilityDF = vis["visibilityDF"]

    return sim
