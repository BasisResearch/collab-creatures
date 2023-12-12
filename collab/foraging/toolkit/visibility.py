import math
import pandas as pd

from collab.foraging.toolkit.utils import generate_grid


def visibility_vs_distance(distance, visibility_range):
    return math.cos((math.pi / visibility_range * distance) / 2)

def construct_visibility(
    birds,
    grid_size,
    visibility_range,
    start=None,
    end=None,
    time_shift=0,
    grid=None,
):
    num_birds = len(birds)
    if start is None:
        start = 0

    if end is None:
        end = len(birds[0])

    visibility = []

    for bird in range(num_birds):
        ranges = []
        for frame in range(start, end):
            if grid is None:
                g = generate_grid(grid_size)
            else:
                g = grid.copy()

            g["distance"] = (
                (g["x"] - birds[bird]["x"].iloc[frame]) ** 2
                + (g["y"] - birds[bird]["y"].iloc[frame]) ** 2
            ) ** 0.5
            

            range_df = g[g["distance"] <= visibility_range].copy()
            range_df["distance_x"] = abs(range_df["x"] - birds[bird]["x"].iloc[frame])
            range_df["distance_y"] = abs(range_df["y"] - birds[bird]["y"].iloc[frame])
            range_df["visibility"] = range_df["distance"].apply(
                lambda d: visibility_vs_distance(d, visibility_range)
            )
            range_df["bird"] = bird + 1
            range_df["time"] = frame + 1

            range_df["time"] = range_df["time"] + time_shift
            ranges.append(range_df)

        visibility.append(ranges)

    birds_visibilities = []
    for bird in range(num_birds):
        birds_visibilities.append(pd.concat(visibility[bird]))

    visibility_df = pd.concat(birds_visibilities)

    return {"visibility": visibility, "visibilityDF": visibility_df}
