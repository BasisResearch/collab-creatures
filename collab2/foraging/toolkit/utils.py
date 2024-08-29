import math
import warnings
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# define a class to streamline object creation
class dataObject:
    def __init__(
        self,
        foragersDF: pd.DataFrame,
        grid_size: Optional[int] = None,
        rewardsDF: Optional[pd.DataFrame] = None,
        frames: Optional[int] = None,
    ):
        """
        Requirements for foragersDF :
            - Required columns "x" : float, "y" : float, "time" : int, "forager" :int
            - Frame numbers and forager indices must start at 0
        """
        if frames is None:
            frames = foragersDF["time"].max() + 1

        if grid_size is None:
            grid_size = int(foragersDF.loc[:, ["x", "y"]].max(axis=None)) + 1

        self.grid_size = grid_size
        self.num_frames = frames

        # raise warning if nan values in DataFrame
        if foragersDF.isna().any(axis=None):
            warnings.warn(
                """ Nan values in data.
                Specify handling of missing data using `skip_incomplete_frames` argument to `generate_all_predictors`"""
            )

        # ensure that forager index is saved as an integer
        foragersDF.loc[:, "forager"] = foragersDF.loc[:, "forager"].astype(int)

        # group dfs by forager index
        foragers = [group for _, group in foragersDF.groupby("forager")]
        self.num_foragers = len(foragers)

        # add nans for any omitted frames & raise warning
        all_frames = range(self.num_frames)
        for f in range(self.num_foragers):
            missing = set(all_frames) - set(foragers[f]["time"])
            if missing:
                warnings.warn(
                    f"""Missing frames encountered for forager {f}, adding NaN fillers.
                    Specify handling of missing data using `skip_incomplete_frames` argument to
                    `generate_all_predictors`"""
                )
                filler_rows = pd.DataFrame(
                    {"time": list(missing), "forager": [f] * len(missing)}
                )
                foragers[f] = pd.concat(
                    [foragers[f], filler_rows]
                )  # adds nan values for all other columns automatically

            # sort by time
            foragers[f].sort_values("time", ignore_index=True, inplace=True)

        # save to object
        self.foragers = foragers
        self.foragersDF = pd.concat(foragers, ignore_index=True)

        # add rewards
        if rewardsDF is not None:
            self.rewardsDF = rewardsDF
            self.rewards = [group for _, group in rewardsDF.groupby("time")]

        # save placeholders for local_windows, predictors and kwargs
        self.local_windows: List[List[pd.DataFrame]] = [[]]
        self.local_windows_kwargs: dict[str, Any] = {}
        self.score_kwargs: dict[str, dict[str, Any]] = {}
        self.predictor_kwargs: dict[str, dict[str, Any]] = {}
        self.derived_quantities: dict[str, List[List[pd.DataFrame]]] = {}
        self.derivedDF: pd.DataFrame

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
