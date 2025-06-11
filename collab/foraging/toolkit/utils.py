import math
import warnings
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# define a class to streamline object creation
class dataObject:
    """
    Object class containing foragers' trajectory data and other attributes.
    """

    def __init__(
        self,
        foragersDF: pd.DataFrame,
        grid_size: Optional[int] = None,
        rewardsDF: Optional[pd.DataFrame] = None,
    ):
        """
        Initializes an instance of dataObject with trajectory data
        :param foragersDF: DataFrame containing foragers' trajectory data and additional attributes.
            Must contain columns "x" : int, "y" : int, "time" : int, "forager" :int.
            Time and forager indices must start at 0.
        :param grid_size: size of grid used to discretize positional data.
            If argument not provided, grid_size is set to the max "x" and "y" value in `foragersDF`
        :param rewardsDF: location of rewards in grid, if applicable.
            Must contain columns "x" : int, "y" : int, "time" : int, "reward" :int.
        """

        if grid_size is None:
            grid_size = int(foragersDF.loc[:, ["x", "y"]].max(axis=None)) + 1

        self.grid_size = grid_size
        self.num_frames = foragersDF["time"].max() + 1

        # raise warning if nan values in DataFrame
        if foragersDF.isna().any(axis=None):
            warnings.warn(
                """
                NaN values in data. The default behavior of predictor/score generating functions is
                to ignore foragers with missing positional data. To modify, see documentation of
                `derive_predictors_and_scores` and `generate_local_windows`
                """
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
                    f"""
                    Missing frames encountered for forager {f}, adding NaN fillers.
                    The default behavior of predictor/score generating functions is
                    to ignore foragers with missing positional data. To modify, see documentation of
                    `derive_predictors_and_scores` and `generate_local_windows`
                    """
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

        # Get unique forager IDs from the DataFrame
        forager_ids = foragersDF.forager.unique()

        # Check if forager IDs are already consecutive integers starting from 0
        # Expected IDs are the consecutive integers starting from 0
        self.local_forager_ids = sorted(range(len(forager_ids)))
        self.global_forager_ids = sorted(forager_ids)

        # Save the original forager IDs and map to consecutive indices if needed
        if self.local_forager_ids != self.global_forager_ids:
            warnings.warn(
                f"""
                Original forager indices were converted to consecutive integers starting from 0.
                To access the original forager IDs, use the apply_forager_id_mapping() method.
                Original IDs were: {self.global_forager_ids}
                """
            )

            # By default, convert global to local IDs
            self.apply_forager_id_mapping(local_to_global=False)

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

    @property
    def local_to_global_map(self) -> dict:
        return {
            local_id: global_id
            for local_id, global_id in enumerate(self.global_forager_ids)
        }

    @property
    def global_to_local_map(self) -> dict:
        return {
            global_id: local_id
            for local_id, global_id in enumerate(self.global_forager_ids)
        }

    def apply_forager_id_mapping(self, local_to_global: bool = False):
        """
        Apply forager ID mapping to convert between local and global IDs. Applies
        directly to the foragersDF attribute.

        Args:
            local_to_global: If True, converts from local to global IDs. If False, converts from global to local IDs
        """

        # Find current forager IDs and grab mapping
        current_ids = set(self.foragersDF["forager"].unique())

        if local_to_global:
            mapping = self.local_to_global_map
        else:
            mapping = self.global_to_local_map

        # Check if already mapped
        target_ids = set(mapping.values())
        if current_ids.issubset(target_ids):
            warnings.warn(
                "IDs are already in target format. Returning DataFrame unchanged."
            )
            return

        # Ensure that all current IDs are in the mapping --> otherwise throw an error
        source_ids = set(mapping.keys())
        if not current_ids.issubset(source_ids):
            unmapped = current_ids - source_ids
            raise ValueError(f"Cannot map forager IDs: {unmapped}")

        # Apply the mapping to the foragersDF
        self.foragersDF = self.foragersDF.assign(
            forager=self.foragersDF.forager.map(mapping).astype(int)
        )


def foragers_to_forager_distances(obj: dataObject) -> List[List[pd.DataFrame]]:
    """
    Calculate the distances between foragers at each time frame.

    Args:
        obj (dataObject): An object containing foragers and foragersDF data.

    Returns:
        list: A nested list where each sublist contains DataFrames of distances
              between foragers at each time frame.
    """
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
