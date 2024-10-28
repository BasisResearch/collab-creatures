import math
from itertools import product

import numpy as np
import pandas as pd


def generate_grid(grid_size):
    grid = list(product(range(0, grid_size), repeat=2))
    return pd.DataFrame(grid, columns=["x", "y"])


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
            range_df["forager"] = forager  # DB: CHECK!
            range_df["time"] = frame  # DB: CHECK!

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


def rewards_trace(distance, rewards_decay):
    return np.exp(-rewards_decay * distance)


def rewards_to_trace(
    rewards,
    grid_size,
    num_frames,
    rewards_decay=0.5,
    start=None,
    end=None,
    time_shift=0,
    grid=None,
):
    if start is None:
        start = 0

    if end is None:
        end = num_frames

    if grid is None:
        grid = generate_grid(grid_size)

    traces = []

    for t in range(start, end):
        rewt = rewards[t]
        trace = grid.copy()
        trace["trace"] = 0
        trace["time"] = t  # DB: CHECK!
        trace["trace_standardized"] = 0

        if len(rewt) > 0:
            for re in range(len(rewt)):
                trace["trace"] += rewards_trace(
                    np.sqrt(
                        (rewt["x"].iloc[re] - trace["x"]) ** 2
                        + (rewt["y"].iloc[re] - trace["y"]) ** 2
                    ),
                    rewards_decay,
                )

            trace["trace_standardized"] = (
                trace["trace"] - trace["trace"].mean()
            ) / trace["trace"].std()

            trace["time"] = trace["time"] + time_shift

        traces.append(trace)

    tracesDF = pd.concat(traces)

    return {"traces": traces, "tracesDF": tracesDF}


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

        rewards[t]["time"] = t  # DB: CHECK!

    rewardsDF = pd.concat(rewards)

    return {"rewards": rewards, "rewardsDF": rewardsDF}


def proximity_score(distance, getting_worse=1.5, optimal=4, proximity_decay=1):
    if distance <= getting_worse:
        return math.sin(math.pi / (2 * getting_worse) * (distance + 3 * getting_worse))
    elif distance <= getting_worse + 1.5 * (optimal - getting_worse):
        return math.sin(
            math.pi / (2 * (optimal - getting_worse)) * (distance - getting_worse)
        )
    else:
        return math.sin(
            math.pi
            / (2 * (optimal - getting_worse))
            * (1.5 * (optimal - getting_worse))
        ) * math.exp(1) ** (
            -proximity_decay * (distance - optimal - 0.5 * (optimal - getting_worse))
        )


def foragers_to_forager_distances(foragers):
    forager_distances = []

    for b in range(len(foragers)):
        frames = len(foragers[0])
        forager_distances.append(np.zeros((frames, len(foragers))))

        for o in range(len(foragers)):
            for t in range(frames):
                forager_distances[b][t, o] = math.sqrt(
                    (foragers[b]["x"].iloc[t] - foragers[o]["x"].iloc[t]) ** 2
                    + (foragers[b]["y"].iloc[t] - foragers[o]["y"].iloc[t]) ** 2
                )

        forager_distances[b] = pd.DataFrame(forager_distances[b])
        forager_distances[b].columns = range(1, len(foragers) + 1)

    return forager_distances


def generate_proximity_score(
    foragers,
    visibility,
    visibility_range,
    getting_worse=1.5,
    optimal=4,
    proximity_decay=1,
    start=0,
    end=None,
    time_shift=0,
):
    if end is None:
        end = len(foragers[0])

    forager_distances = foragers_to_forager_distances(foragers)

    proximity = visibility.copy()
    for b in range(len(foragers)):
        for t in range(start, end):
            proximity[b][t]["proximity"] = 0

            distbt = forager_distances[b].iloc[t].drop(b + 1)

            visible_foragers = distbt[distbt <= visibility_range].index.tolist()

            if len(visible_foragers) > 0:
                for vb in range(len(visible_foragers)):
                    o_x = foragers[visible_foragers[vb] - 1]["x"].iloc[t]
                    o_y = foragers[visible_foragers[vb] - 1]["y"].iloc[t]

                    proximity[b][t]["proximity"] += [
                        proximity_score(s, getting_worse, optimal, proximity_decay)
                        for s in np.sqrt(
                            (proximity[b][t]["x"] - o_x) ** 2
                            + (proximity[b][t]["y"] - o_y) ** 2
                        )
                    ]

            proximity[b][t]["proximity_standardized"] = (
                proximity[b][t]["proximity"] - proximity[b][t]["proximity"].mean()
            ) / proximity[b][t]["proximity"].std()

            proximity[b][t]["proximity_standardized"].fillna(0, inplace=True)

            proximity[b][t]["forager"] = b 
            proximity[b][t]["time"] = t + 1 + time_shift

    proximityDF = pd.concat([pd.concat(p) for p in proximity])

    return {"proximity": proximity, "proximityDF": proximityDF}
