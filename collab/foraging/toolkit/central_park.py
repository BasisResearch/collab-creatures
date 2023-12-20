import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .proximity import proximity_score
from .utils import generate_grid
from .visibility import visibility_vs_distance



def cp_generate_visibility(
    foragers,
    time_shift=0,
    grid_size=90,
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

    for forager in range(len(foragers)):
        ranges = []
        times_b = foragers[forager]["time"].unique()
        for frame in times_b:
            g = grid.copy()

            temporary_x = foragers[forager]["x"][foragers[forager]["time"] == frame]
            temporary_y = foragers[forager]["y"][foragers[forager]["time"] == frame]

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
            range_df["forager"] = forager + 1
            range_df["time"] = frame

            range_df["time"] = range_df["time"] + time_shift
            ranges.append(range_df)

        visibility.append(ranges)

    foragers_visibilities = []
    for forager in range(len(foragers)):
        foragers_visibilities.append(pd.concat(visibility[forager]))

    visibility_df = pd.concat(foragers_visibilities)

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
    forager_distances=None,
):
    foragers = obj.foragers
    foragersDF = obj.foragersDF

    if end is None:
        end = len(foragers[0])

    if forager_distances is None:
        forager_distances = cp_foragers_to_forager_distances(obj)

    proximity = obj.visibility.copy()

    for forager in range(len(foragers)):
        times_b = foragers[forager]["time"].unique()

        for frame in times_b:
            frame_index = np.where(times_b == frame)[0][0]

            foragers_at_frameDF = foragersDF[foragersDF["time"] == frame]
            foragers_at_frameDF.sort_values(by="forager", inplace=True)

            foragers_at_frame = foragers_at_frameDF["forager"].unique()
            foragers_at_frame.sort()  # perhaps redundant

            # assert forager + 1 in foragers_at_frame

            proximity[forager][frame_index]["proximity"] = 0

            dist_bt = forager_distances[forager][frame_index]

            visible_foragers = dist_bt["foragers_at_frame"][
                dist_bt["distance"] <= visibility_range
            ].tolist()

            if visible_foragers:
                for vb in visible_foragers:
                    o_x = (
                        foragers[vb - 1].loc[foragers[vb - 1]["time"] == frame, "x"].values[0]
                    )
                    o_y = (
                        foragers[vb - 1].loc[foragers[vb - 1]["time"] == frame, "y"].values[0]
                    )

                proximity[forager][frame_index]["proximity"] += [
                    proximity_score(s, getting_worse, optimal, proximity_decay)
                    for s in np.sqrt(
                        (proximity[forager][frame_index]["x"] - o_x) ** 2
                        + (proximity[forager][frame_index]["y"] - o_y) ** 2
                    )
                ]

            proximity[forager][frame_index]["proximity_standardized"] = (
                proximity[forager][frame_index]["proximity"]
                - proximity[forager][frame_index]["proximity"].mean()
            ) / proximity[forager][frame_index]["proximity"].std()

            proximity[forager][frame_index]["proximity_standardized"].fillna(
                0, inplace=True
            )

            proximity[forager][frame_index]["forager"] = forager + 1
            proximity[forager][frame_index]["time"] = frame

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
    forager_distances=None,
):
    foragers = obj.foragers
    foragersDF = obj.foragersDF

    forager_map = [foragers[k]["forager"].unique().item() for k in range(len(foragers))]

    if end is None:
        end = len(foragers[0])

    if forager_distances is None:
        forager_distances = cp_foragers_to_forager_distances(obj)

    proximity = obj.visibility.copy()

    for forager in range(len(foragers)):
        times_b = foragers[forager]["time"].unique()

        for frame in times_b:
            frame_index = np.where(times_b == frame)[0][0]

            foragers_at_frameDF = foragersDF[foragersDF["time"] == frame]
            foragers_at_frameDF.sort_values(by="forager", inplace=True)

            foragers_at_frame = foragers_at_frameDF["forager"].unique()
            foragers_at_frame.sort()  # perhaps redundant

            proximity[forager][frame_index]["proximity"] = 0

            dist_bt = forager_distances[forager][frame_index]

            visible_foragers = dist_bt["foragers_at_frame"][
                dist_bt["distance"] <= visibility_range
            ].tolist()

            if visible_foragers:
                for vb in visible_foragers:
                    vb_loc = forager_map.index(vb)
                    o_x = (
                        foragers[vb_loc].loc[foragers[vb_loc]["time"] == frame, "x"].values[0]
                    )
                    o_y = (
                        foragers[vb_loc].loc[foragers[vb_loc]["time"] == frame, "y"].values[0]
                    )

                proximity[forager][frame_index]["proximity"] += [
                    proximity_score(s, getting_worse, optimal, proximity_decay)
                    for s in np.sqrt(
                        (proximity[forager][frame_index]["x"] - o_x) ** 2
                        + (proximity[forager][frame_index]["y"] - o_y) ** 2
                    )
                ]

            proximity[forager][frame_index]["proximity_standardized"] = (
                proximity[forager][frame_index]["proximity"]
                - proximity[forager][frame_index]["proximity"].mean()
            ) / proximity[forager][frame_index]["proximity"].std()

            proximity[forager][frame_index]["proximity_standardized"].fillna(
                0, inplace=True
            )

            proximity[forager][frame_index]["forager"] = forager + 1
            proximity[forager][frame_index]["time"] = frame

    proximityDF = pd.concat([pd.concat(p) for p in proximity])

    return {"proximity": proximity, "proximityDF": proximityDF}


def cp_add_how_far_squared_scaled(sim):
    foragers = sim.foragers
    step_size_max = sim.step_size_max
    visibility_range = 100
    how_far = sim.proximity.copy()

    for forager in range(sim.num_foragers):
        df = foragers[forager]
        for frame in range(len(df) - 1):
            try:
                x_new = int(df["x"][frame + 1])
                y_new = int(df["y"][frame + 1])
            except (KeyError, AttributeError):
                x_new = int(df["x"].iloc[frame + 1])
                y_new = int(df["y"].iloc[frame + 1])

            assert isinstance(x_new, int) and isinstance(y_new, int)

            _hf = how_far[forager][frame]
            _hf["how_far_squared"] = (_hf["x"] - x_new) ** 2 + (_hf["y"] - y_new) ** 2
            _hf["how_far_squared_scaled"] = (
                -_hf["how_far_squared"]
                / (2 * (sim.step_size_max + visibility_range) ** 2)
                + 1
            )

        # m maybe not needed
        # how_far[forager][len(df)]["how_far_squared"] = np.nan
        # how_far[bforager][len(df)]["how_far_squared_scaled"] = np.nan

    sim.how_far = how_far

    foragers_how_far = []
    for forager in range(sim.num_foragers):
        foragers_how_far.append(pd.concat(how_far[forager]))

    sim.how_farDF = pd.concat(foragers_how_far)
