import math
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt


from .utils import generate_grid
from .visibility import 


def cp_birds_to_bird_distances(obj):
    distances = []
    birds = obj.birds
    birdsDF = obj.birdsDF

    for bird in range(len(birds)):
        bird_distances = []

        times_b = birds[bird]["time"].unique()

        for frame in times_b:
            # frame_index = np.where(times_b == frame)[0][0]

            birds_at_frameDF = birdsDF[birdsDF["time"] == frame]
            birds_at_frameDF.sort_values(by="bird", inplace=True)
            # birds_at_frame = [group for _, group in birds_at_frameDF.groupby("bird")]

            birds_at_frame = birds_at_frameDF["bird"].unique()
            birds_at_frame.sort()

            bird_x = birds[bird][birds[bird]["time"] == frame]["x"].item()

            bird_y = birds[bird][birds[bird]["time"] == frame]["y"].item()

            assert isinstance(bird_x, float) and isinstance(bird_y, float)

            distances_now = []
            for other in birds_at_frame:
                other_x = birds[other - 1][birds[other - 1]["time"] == frame]["x"].item()
                other_y = birds[other - 1][birds[other - 1]["time"] == frame]["y"].item()

                assert isinstance(other_x, float) and isinstance(other_y, float)

                distances_now.append(math.sqrt((bird_x - other_x) ** 2 + (bird_y - other_y) ** 2))

            assert len(distances_now) == len(birds_at_frame)

            distances_now_df = pd.DataFrame({"distance": distances_now, "birds_at_frame": birds_at_frame})

            bird_distances.append(distances_now_df)

        distances.append(bird_distances)

    return distances


def cp_distances_and_peaks(distances, bins=40, x_min=None, x_max=None):
    distances_list = [distance for sublist in distances for df in sublist for distance in df["distance"].tolist()]

    distances_list = list(filter(lambda x: x != 0, distances_list))

    hist, bins, _ = plt.hist(distances_list, bins=40, color="blue", edgecolor="black")
    peaks, _ = find_peaks(hist)

    plt.hist(distances_list, bins=bins, color="blue", edgecolor="black")
    plt.scatter(bins[peaks], hist[peaks], c="red", marker="o", s=50, label="Peaks")

    peak_values = hist[peaks]
    peak_positions = np.round(bins[peaks], 2)

    if x_min is not None:
        plt.xlim(x_min, x_max)

    for i, peak_x in enumerate(bins[peaks]):
        plt.annotate(
            f"{peak_positions[i]}",
            (peak_x, hist[peaks][i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            color="red",
        )


def cp_generate_visibility(
    birds,
    getting_worse=1.5,
    optimal=4,
    proximity_decay=1,
    time_shift=0,
    grid_size=90,
    joint_df=False,
    grid=None,
    visibility_range=90,
    sampling_rate=0.02,
    random_seed=42,
):
    if grid is None:
        grid = generate_grid(grid_size)
        grid = grid.sample(frac=sampling_rate, random_state=random_seed)

    else:
        grid = grid.copy()

    visibility = []

    for bird in range(len(birds)):
        ranges = []
        times_b = birds[bird]["time"].unique()
        for frame in times_b:
            g = grid.copy()

            temporary_x = birds[bird]["x"][birds[bird]["time"] == frame]
            temporary_y = birds[bird]["y"][birds[bird]["time"] == frame]

            g["distance"] = ((g["x"] - temporary_x.item()) * 2 + (g["y"] - temporary_y.item()) * 2) ** 0.5

            range_df = g[g["distance"] <= visibility_range].copy()
            range_df = g[g["distance"] <= visibility_range].copy()
            range_df["distance_x"] = abs(range_df["x"] - temporary_x.item())
            range_df["distance_y"] = abs(range_df["y"] - temporary_y.item())
            range_df["visibility"] = range_df["distance"].apply(
                lambda d: visibility_vs_distance(d, visibility_range)
            )
            range_df["bird"] = bird + 1
            range_df["time"] = frame

            range_df["time"] = range_df["time"] + time_shift
            ranges.append(range_df)

        visibility.append(ranges)

    birds_visibilities = []
    for bird in range(len(birds)):
        birds_visibilities.append(pd.concat(visibility[bird]))

    visibility_df = pd.concat(birds_visibilities)

    return {"visibility": visibility, "visibilityDF": visibility_df}
