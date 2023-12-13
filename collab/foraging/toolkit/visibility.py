import math

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
