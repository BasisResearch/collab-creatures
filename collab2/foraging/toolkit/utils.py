import math
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import warnings

#define a class for streamlining object creation 
class dataObject:
    def __init__(self, foragersDF, grid_size=None, rewardsDF=None, frames=None):
        if frames is None:
            frames = foragersDF["time"].nunique()

        if grid_size is None:
            grid_size = int(max(max(foragersDF["x"]), max(foragersDF["y"])))

        self.grid_size = grid_size
        self.num_frames = frames
        self.foragersDF = foragersDF

        if self.foragersDF["forager"].min() == 0:
            self.foragersDF["forager"] = self.foragersDF["forager"] + 1

        self.foragers = [group for _, group in foragersDF.groupby("forager")]

        if rewardsDF is not None:
            self.rewardsDF = rewardsDF
            self.rewards = [group for _, group in rewardsDF.groupby("time")]

        self.num_foragers = len(self.foragers)

        #raise warning if all foragers are not present in any frame 
        for f in range(self.num_foragers):
            missing = set(range(self.num_frames)) - set(self.foragers[f]["time"][self.foragers[f]["x"].notna()].to_list())
            if missing :
                warnings.warn(f"Incomplete frames in data. Specify handling of missing data using skip_incomplete_frames argument to generate_all_predictors")
                break
        
    def calculate_step_size_max(self):
        step_maxes = []

        for b in range(len(self.foragers)):
            df = self.foragers[b]
            step_maxes.append(
                max(
                    max(
                        [
                            abs(df["x"].iloc[t + 1] - df["x"].iloc[t])
                            for t in range(len(df) - 1)
                        ]
                    ),
                    max(
                        [
                            abs(df["y"].iloc[t + 1] - df["y"].iloc[t])
                            for t in range(len(df) - 1)
                        ]
                    ),
                )
            )
        self.step_size_max = max(step_maxes)

def object_from_data(
    foragersDF,
    grid_size=None,
    rewardsDF=None,
    frames=None,
    calculate_step_size_max=False,
):
    if frames is None:
        frames = foragersDF["time"].nunique()

    if grid_size is None:
        grid_size = int(max(max(foragersDF["x"]), max(foragersDF["y"])))

    class EmptyObject:
        pass

    sim = EmptyObject()

    sim.grid_size = grid_size
    sim.num_frames = frames
    sim.foragersDF = foragersDF
    if sim.foragersDF["forager"].min() == 0:
        sim.foragersDF["forager"] = sim.foragersDF["forager"] + 1

    sim.foragers = [group for _, group in foragersDF.groupby("forager")]

    if rewardsDF is not None:
        sim.rewardsDF = rewardsDF
        sim.rewards = [group for _, group in rewardsDF.groupby("time")]

    sim.num_foragers = len(sim.foragers)

    if calculate_step_size_max:
        step_maxes = []

        for b in range(len(sim.foragers)):
            df = sim.foragers[b]
            step_maxes.append(
                max(
                    max(
                        [
                            abs(df["x"].iloc[t + 1] - df["x"].iloc[t])
                            for t in range(len(df) - 1)
                        ]
                    ),
                    max(
                        [
                            abs(df["y"].iloc[t + 1] - df["y"].iloc[t])
                            for t in range(len(df) - 1)
                        ]
                    ),
                )
            )
        sim.step_size_max = max(step_maxes)

    return sim


def foragers_to_forager_distances(obj):
    distances = []
    foragers = obj.foragers
    foragersDF = obj.foragersDF
    forager_map = [foragers[k]["forager"].unique().item() for k in range(len(foragers))]
    for forager in range(len(foragers)):
        forager_distances = []

        times_b = foragers[forager]["time"].unique()

        for frame in times_b:
            foragers_at_frameDF = foragersDF[foragersDF["time"] == frame].copy()
            foragers_at_frameDF.sort_values(by="forager", inplace=True)

            foragers_at_frame = foragers_at_frameDF["forager"].unique()
            foragers_at_frame.sort()

            forager_x = foragers[forager][foragers[forager]["time"] == frame][
                "x"
            ].item()

            forager_y = foragers[forager][foragers[forager]["time"] == frame][
                "y"
            ].item()

            assert isinstance(forager_x, float) and isinstance(forager_y, float)

            distances_now = []
            for other in foragers_at_frame:
                other_location = forager_map.index(other)
                df = foragers[other_location].copy()
                other_x = df[df["time"] == frame]["x"].item()
                other_y = df[df["time"] == frame]["y"].item()

                assert isinstance(other_x, float) and isinstance(other_y, float)

                distances_now.append(
                    math.sqrt((forager_x - other_x) ** 2 + (forager_y - other_y) ** 2)
                )

            distances_now_df = pd.DataFrame(
                {"distance": distances_now, "foragers_at_frame": foragers_at_frame}
            )

            forager_distances.append(distances_now_df)

        distances.append(forager_distances)

    return distances


def distances_and_peaks(distances, bins=40, x_min=None, x_max=None):
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


def generate_grid(grid_size):
    grid = list(product(range(1, grid_size + 1), repeat=2))
    return pd.DataFrame(grid, columns=["x", "y"])


# remove rewards eaten by foragers in proximity
def update_rewards(sim, rewards, foragers, start=1, end=None):
    if end is None:
        end = foragers[0].shape[0]

    for t in range(start, end):
        rewards[t] = rewards[t - 1].copy()
        eaten = []

        for b in range(len(foragers)):
            eaten_b = rewards[t][
                (abs(rewards[t]["x"] - foragers[b].iloc[t]["x"]) <= sim.grab_range)
                & (abs(rewards[t]["y"] - foragers[b].iloc[t]["y"]) <= sim.grab_range)
            ].index.tolist()
            if eaten_b:
                eaten.extend(eaten_b)

        if eaten:
            rewards[t] = rewards[t].drop(eaten)

        rewards[t]["time"] = t + 1

    rewardsDF = pd.concat(rewards)

    return {"rewards": rewards, "rewardsDF": rewardsDF}
