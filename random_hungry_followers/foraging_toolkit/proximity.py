import sys

sys.path.insert(0, "../..")

import math
import pandas as pd
import numpy as np

# import foraging_toolkit as ft


def proximity_score(distance, getting_worse=1.5, optimal=4, proximity_decay=1):
    if distance <= getting_worse:
        return math.sin(
            math.pi / (2 * getting_worse) * (distance + 3 * getting_worse)
        )
    elif distance <= getting_worse + 1.5 * (optimal - getting_worse):
        return math.sin(
            math.pi
            / (2 * (optimal - getting_worse))
            * (distance - getting_worse)
        )
    else:
        return math.sin(
            math.pi
            / (2 * (optimal - getting_worse))
            * (1.5 * (optimal - getting_worse))
        ) * math.exp(1) ** (
            -proximity_decay
            * (distance - optimal - 0.5 * (optimal - getting_worse))
        )


def birds_to_bird_distances(birds):
    bird_distances = []

    for b in range(len(birds)):
        frames = len(birds[0])
        bird_distances.append(np.zeros((frames, len(birds))))

        for o in range(len(birds)):
            for t in range(frames):
                bird_distances[b][t, o] = math.sqrt(
                    (birds[b]["x"].iloc[t] - birds[o]["x"].iloc[t]) ** 2
                    + (birds[b]["y"].iloc[t] - birds[o]["y"].iloc[t]) ** 2
                )

        bird_distances[b] = pd.DataFrame(bird_distances[b])
        bird_distances[b].columns = range(1, len(birds) + 1)

    return bird_distances


def generate_proximity_score(
    birds,
    visibility,
    visibility_range,
    getting_worse=1.5,
    optimal=4,
    proximity_decay=1,
):
    bird_distances = birds_to_bird_distances(birds)

    proximity = visibility.copy()
    for b in range(len(birds)):
        for t in range(len(birds[0])):
            proximity[b][t]["proximity"] = 0

            distbt = bird_distances[b].iloc[t].drop(b + 1)

            visible_birds = distbt[distbt <= visibility_range].index.tolist()

            if len(visible_birds) > 0:
                for vb in range(len(visible_birds)):
                    o_x = birds[visible_birds[vb] - 1]["x"].iloc[t]
                    o_y = birds[visible_birds[vb] - 1]["y"].iloc[t]

                    proximity[b][t]["proximity"] += [
                        proximity_score(
                            s, getting_worse, optimal, proximity_decay
                        )
                        for s in np.sqrt(
                            (proximity[b][t]["x"] - o_x) ** 2
                            + (proximity[b][t]["y"] - o_y) ** 2
                        )
                    ]

                    # print(proximity[b][t].head(n=1))
                    # print(distbt[visible_birds[vb]])
                    # print(proximity[b][t]["proximity"])
                    # proximity[b][t]["proximity"] += proximity_score(
                    #     distbt[visible_birds[vb]],
                    #     getting_worse,
                    #     optimal,
                    #     proximity_decay,
                    # )

            proximity[b][t]["proximity_standardized"] = (
                proximity[b][t]["proximity"]
                - proximity[b][t]["proximity"].mean()
            ) / proximity[b][t]["proximity"].std()

            proximity[b][t]["proximity_standardized"].fillna(0, inplace=True)

            proximity[b][t]["bird"] = b + 1
            proximity[b][t]["time"] = t + 1

    proximityDF = pd.concat([pd.concat(p) for p in proximity])

    return {"proximity": proximity, "proximityDF": proximityDF}


# import matplotlib.pyplot as plt

# distances = np.linspace(0, 20, 100)
# scores = [proximity_score(d) for d in distances]

# plt.plot(distances, scores)
# plt.xlabel("Distance")
# plt.ylabel("Proximity Score")
# plt.title("Proximity Score vs. Distance")
# plt.grid(True)
# plt.show()
