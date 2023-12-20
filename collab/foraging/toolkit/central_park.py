import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .proximity import proximity_score
from .utils import generate_grid
from .visibility import visibility_vs_distance


# TODO consider revised version from central park notebook
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
                other_x = birds[other - 1][birds[other - 1]["time"] == frame][
                    "x"
                ].item()
                other_y = birds[other - 1][birds[other - 1]["time"] == frame][
                    "y"
                ].item()

                assert isinstance(other_x, float) and isinstance(other_y, float)

                distances_now.append(
                    math.sqrt((bird_x - other_x) ** 2 + (bird_y - other_y) ** 2)
                )

            assert len(distances_now) == len(birds_at_frame)

            distances_now_df = pd.DataFrame(
                {"distance": distances_now, "birds_at_frame": birds_at_frame}
            )

            bird_distances.append(distances_now_df)

        distances.append(bird_distances)

    return distances


# neded to customize this
# as sps not only migrate in-out
# but also are not in the order of appearance
# TODO sort by bird no at the point of creating objects and grouping


def cp_birds_to_bird_distances_sps(obj):
    distances = []
    birds = obj.birds
    birdsDF = obj.birdsDF
    bird_map = [birds[k]["bird"].unique().item() for k in range(len(birds))]

    for bird in range(len(birds)):
        bird_distances = []

        times_b = birds[bird]["time"].unique()
        bird_name = birds[bird]["bird"].unique().item()

        for frame in times_b:
            birds_at_frameDF = birdsDF[birdsDF["time"] == frame]
            birds_at_frameDF.sort_values(by="bird", inplace=True)

            birds_at_frame = birds_at_frameDF["bird"].unique()
            birds_at_frame.sort()

            bird_x = birds[bird][birds[bird]["time"] == frame]["x"].item()

            bird_y = birds[bird][birds[bird]["time"] == frame]["y"].item()

            assert isinstance(bird_x, float) and isinstance(bird_y, float)

            distances_now = []
            for other in birds_at_frame:
                other_location = bird_map.index(other)

                df = birds[other_location]
                other_x = df[df["time"] == frame]["x"].item()
                other_y = df[df["time"] == frame]["y"].item()

                assert isinstance(other_x, float) and isinstance(other_y, float)

                distances_now.append(
                    math.sqrt((bird_x - other_x) ** 2 + (bird_y - other_y) ** 2)
                )

            distances_now_df = pd.DataFrame(
                {"distance": distances_now, "birds_at_frame": birds_at_frame}
            )

            bird_distances.append(distances_now_df)

        distances.append(bird_distances)

    return distances


def cp_distances_and_peaks(distances, bins=40, x_min=None, x_max=None):
    distances_list = [
        distance
        for sublist in distances
        for df in sublist
        for distance in df["distance"].tolist()
    ]

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

            g["distance"] = (
                (g["x"] - temporary_x.item()) * 2 + (g["y"] - temporary_y.item()) * 2
            ) ** 0.5

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


def cp_generate_proximity_score(
    obj,
    visibility_range=100,
    getting_worse=1.5,
    optimal=4,
    proximity_decay=1,
    start=0,
    end=None,
    time_shift=0,
    joint_df=False,
    bird_distances=None,
):
    birds = obj.birds
    birdsDF = obj.birdsDF

    if end is None:
        end = len(birds[0])

    if bird_distances is None:
        bird_distances = cp_birds_to_bird_distances(obj)

    proximity = obj.visibility.copy()

    for bird in range(len(birds)):
        times_b = birds[bird]["time"].unique()

        for frame in times_b:
            frame_index = np.where(times_b == frame)[0][0]

            birds_at_frameDF = birdsDF[birdsDF["time"] == frame]
            birds_at_frameDF.sort_values(by="bird", inplace=True)

            birds_at_frame = birds_at_frameDF["bird"].unique()
            birds_at_frame.sort()  # perhaps redundant

            # assert bird + 1 in birds_at_frame

            proximity[bird][frame_index]["proximity"] = 0

            dist_bt = bird_distances[bird][frame_index]

            visible_birds = dist_bt["birds_at_frame"][
                dist_bt["distance"] <= visibility_range
            ].tolist()

            if visible_birds:
                for vb in visible_birds:
                    o_x = (
                        birds[vb - 1].loc[birds[vb - 1]["time"] == frame, "x"].values[0]
                    )
                    o_y = (
                        birds[vb - 1].loc[birds[vb - 1]["time"] == frame, "y"].values[0]
                    )

                proximity[bird][frame_index]["proximity"] += [
                    proximity_score(s, getting_worse, optimal, proximity_decay)
                    for s in np.sqrt(
                        (proximity[bird][frame_index]["x"] - o_x) ** 2
                        + (proximity[bird][frame_index]["y"] - o_y) ** 2
                    )
                ]

            proximity[bird][frame_index]["proximity_standardized"] = (
                proximity[bird][frame_index]["proximity"]
                - proximity[bird][frame_index]["proximity"].mean()
            ) / proximity[bird][frame_index]["proximity"].std()

            proximity[bird][frame_index]["proximity_standardized"].fillna(
                0, inplace=True
            )

            proximity[bird][frame_index]["bird"] = bird + 1
            proximity[bird][frame_index]["time"] = frame

    proximityDF = pd.concat([pd.concat(p) for p in proximity])

    return {"proximity": proximity, "proximityDF": proximityDF}


def cp_generate_proximity_score_sps(
    obj,
    visibility_range=100,
    getting_worse=1.5,
    optimal=4,
    proximity_decay=1,
    start=0,
    end=None,
    time_shift=0,
    joint_df=False,
    bird_distances=None,
):
    birds = obj.birds
    birdsDF = obj.birdsDF

    bird_map = [birds[k]["bird"].unique().item() for k in range(len(birds))]

    if end is None:
        end = len(birds[0])

    if bird_distances is None:
        bird_distances = cp_birds_to_bird_distances(obj)

    proximity = obj.visibility.copy()

    for bird in range(len(birds)):
        times_b = birds[bird]["time"].unique()

        for frame in times_b:
            frame_index = np.where(times_b == frame)[0][0]

            birds_at_frameDF = birdsDF[birdsDF["time"] == frame]
            birds_at_frameDF.sort_values(by="bird", inplace=True)

            birds_at_frame = birds_at_frameDF["bird"].unique()
            birds_at_frame.sort()  # perhaps redundant

            proximity[bird][frame_index]["proximity"] = 0

            dist_bt = bird_distances[bird][frame_index]

            visible_birds = dist_bt["birds_at_frame"][
                dist_bt["distance"] <= visibility_range
            ].tolist()

            if visible_birds:
                for vb in visible_birds:
                    vb_loc = bird_map.index(vb)
                    o_x = (
                        birds[vb_loc].loc[birds[vb_loc]["time"] == frame, "x"].values[0]
                    )
                    o_y = (
                        birds[vb_loc].loc[birds[vb_loc]["time"] == frame, "y"].values[0]
                    )

                proximity[bird][frame_index]["proximity"] += [
                    proximity_score(s, getting_worse, optimal, proximity_decay)
                    for s in np.sqrt(
                        (proximity[bird][frame_index]["x"] - o_x) ** 2
                        + (proximity[bird][frame_index]["y"] - o_y) ** 2
                    )
                ]

            proximity[bird][frame_index]["proximity_standardized"] = (
                proximity[bird][frame_index]["proximity"]
                - proximity[bird][frame_index]["proximity"].mean()
            ) / proximity[bird][frame_index]["proximity"].std()

            proximity[bird][frame_index]["proximity_standardized"].fillna(
                0, inplace=True
            )

            proximity[bird][frame_index]["bird"] = bird + 1
            proximity[bird][frame_index]["time"] = frame

    proximityDF = pd.concat([pd.concat(p) for p in proximity])

    return {"proximity": proximity, "proximityDF": proximityDF}


def cp_add_how_far_squared_scaled(sim):
    birds = sim.birds
    step_size_max = sim.step_size_max
    visibility_range = 100
    how_far = sim.proximity.copy()

    for bird in range(sim.num_birds):
        df = birds[bird]
        for frame in range(len(df) - 1):
            try:
                x_new = int(df["x"][frame + 1])
                y_new = int(df["y"][frame + 1])
            except (KeyError, AttributeError):
                x_new = int(df["x"].iloc[frame + 1])
                y_new = int(df["y"].iloc[frame + 1])

            assert isinstance(x_new, int) and isinstance(y_new, int)

            _hf = how_far[bird][frame]
            _hf["how_far_squared"] = (_hf["x"] - x_new) ** 2 + (_hf["y"] - y_new) ** 2
            _hf["how_far_squared_scaled"] = (
                -_hf["how_far_squared"]
                / (2 * (sim.step_size_max + visibility_range) ** 2)
                + 1
            )

        # m maybe not needed
        # how_far[bird][len(df)]["how_far_squared"] = np.nan
        # how_far[bbird][len(df)]["how_far_squared_scaled"] = np.nan

    sim.how_far = how_far

    birds_how_far = []
    for bird in range(sim.num_birds):
        birds_how_far.append(pd.concat(how_far[bird]))

    sim.how_farDF = pd.concat(birds_how_far)
