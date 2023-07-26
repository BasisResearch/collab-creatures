import sys

sys.path.insert(0, "..")


import numpy as np
from .utils import generate_grid

# import foraging_toolkit as ft
import math

import math
import pandas as pd
import matplotlib.pyplot as plt


def visibility_vs_distance(distance, visibility_range):
    return math.cos((math.pi / visibility_range * distance) / 2)


# distances = np.linspace(0, 8, 100)
# scores = [visibility_vs_distance(d, 8) for d in distances]

# plt.plot(distances, scores)
# plt.xlabel("Distance")
# plt.ylabel("Visibility Score")
# plt.title("Visibility Score vs. Distance")
# plt.grid(True)
# plt.show()


def construct_visibility(birds, grid_size, visibility_range, end=None):
    num_birds = len(birds)

    if end is None:
        end = len(birds[0])

    visibility = []

    for bird in range(num_birds):
        gridb = []
        ranges = []
        for frame in range(end):
            g = generate_grid(grid_size)
            gridb.append(g)
            gridb[frame]["distance"] = (
                (gridb[frame]["x"] - birds[bird]["x"].iloc[frame]) ** 2
                + (gridb[frame]["y"] - birds[bird]["y"].iloc[frame]) ** 2
            ) ** 0.5

            range_df = gridb[frame][gridb[frame]["distance"] <= visibility_range].copy()
            range_df["distance_x"] = abs(range_df["x"] - birds[bird]["x"].iloc[frame])
            range_df["distance_y"] = abs(range_df["y"] - birds[bird]["y"].iloc[frame])
            range_df["visibility"] = range_df["distance"].apply(
                lambda d: visibility_vs_distance(d, visibility_range)
            )
            range_df["bird"] = bird + 1
            range_df["time"] = frame + 1

            ranges.append(range_df)

        visibility.append(ranges)

    birds_visibilities = []
    for bird in range(num_birds):
        birds_visibilities.append(pd.concat(visibility[bird]))

    visibility_df = pd.concat(birds_visibilities)
    # visibility_df["bird"] = visibility_df["bird"].astype("category")

    return {"visibility": visibility, "visibilityDF": visibility_df}


# def visibility_to_df(visibility):
#     birds_visibilities = []
#     num_birds = len(visibility)

#     for bird in range(num_birds):
#         birds_visibilities.append(pd.concat(visibility[bird]))

#     visibility_df = pd.concat(birds_visibilities)
#     visibility_df["bird"] = visibility_df["bird"].astype("category")

#     return visibility_df
