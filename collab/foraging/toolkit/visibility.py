import math

import numpy as np
import pandas as pd

from collab.foraging.toolkit.utils import generate_grid


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


def filter_by_visibility(
    sim,
    subject: int,
    time_shift: int,
    visibility_restriction: str,
    info_time_decay: int = 1,
    finders_tolerance: float = 2.0,
    filter_by_on_reward: bool = False,
):
    other_foragers_df = sim.foragersDF[sim.foragersDF["forager"] != subject]

    myself = sim.foragersDF[sim.foragersDF["forager"] == subject]

    filtered_foragers = []
    for t in range(time_shift + 1, (time_shift + len(sim.foragers[0]))):
        x_series = myself["x"][myself["time"] == t]
        y_series = myself["y"][myself["time"] == t]

        if isinstance(x_series, pd.Series) and len(x_series) == 1:
            myself_x = x_series.item()
            myself_y = y_series.item()
        else:
            myself_x = x_series
            myself_y = y_series

        others_now = other_foragers_df[other_foragers_df["time"] == t].copy()

        others_now["distance"] = np.sqrt(
            (others_now["x"] - myself_x) ** 2 + (others_now["y"] - myself_y) ** 2
        )

        others_now["out_of_range"] = others_now["distance"] > sim.visibility_range

        if visibility_restriction == "invisible":
            others_now = others_now[others_now["out_of_range"]]
        elif visibility_restriction == "visible":
            others_now = others_now[~others_now["out_of_range"]]

        if filter_by_on_reward:
            on_reward = []
            for _, row in others_now.iterrows():
                others_x = row["x"]
                others_y = row["y"]

                on_reward.append(
                    any(
                        np.sqrt(
                            (others_x - sim.rewards[t - time_shift - 1]["x"]) ** 2
                            + (others_y - sim.rewards[t - time_shift - 1]["y"]) ** 2
                        )
                        <= finders_tolerance
                    )
                )

            others_now["on_reward"] = on_reward

        filtered_foragers.append(others_now)
    filtered_foragersDF = pd.concat(filtered_foragers)

    if filter_by_on_reward:
        filtered_foragersDF = filtered_foragersDF[
            filtered_foragersDF["on_reward"]
            == True  # noqa E712   the == True  removal or switching to `is True` leads to failure
        ]

    expansion = [
        filtered_foragersDF.assign(time=filtered_foragersDF["time"] + i)
        for i in range(1, info_time_decay + 1)
    ]

    expansion_df = pd.concat(expansion, ignore_index=True)

    revelant_othersDF = pd.concat([filtered_foragersDF, expansion_df])

    return revelant_othersDF
