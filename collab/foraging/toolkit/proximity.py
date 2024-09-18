import math

import numpy as np
import pandas as pd


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

            proximity[b][t]["forager"] = b + 1
            proximity[b][t]["time"] = t + 1 + time_shift

    proximityDF = pd.concat([pd.concat(p) for p in proximity])

    return {"proximity": proximity, "proximityDF": proximityDF}
