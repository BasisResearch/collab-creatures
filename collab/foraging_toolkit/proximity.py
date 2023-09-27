import sys

sys.path.insert(0, "../..")

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks


# import foraging_toolkit as ft


def proximity_score(distance, getting_worse=1.5, optimal=4, proximity_decay=1):
    if distance <= getting_worse:
        return math.sin(math.pi / (2 * getting_worse) * (distance + 3 * getting_worse))
    elif distance <= getting_worse + 1.5 * (optimal - getting_worse):
        return math.sin(math.pi / (2 * (optimal - getting_worse)) * (distance - getting_worse))
    else:
        return math.sin(math.pi / (2 * (optimal - getting_worse)) * (1.5 * (optimal - getting_worse))) * math.exp(
            1
        ) ** (-proximity_decay * (distance - optimal - 0.5 * (optimal - getting_worse)))


def birds_to_bird_distances(birds, joint_df=False):
    bird_positions_by_frame = []
    bird_distances = []

    if joint_df:
        birds_split = [group for _, group in birds.groupby("bird")]
        times = birds["time"].unique()
    else:
        birds_split = birds
        # TODO needs to be checked
        times = [set(bird["time"].unique()) for bird in birds].union().unique()

    for time in times:
        selected_rows = [df[df["time"] == time] for df in birds_split]
        bird_positions_by_frame.append(pd.concat(selected_rows, axis=0, ignore_index=True))

    for frame in bird_positions_by_frame:
        bird_distances.append(np.zeros((len(frame), len(frame))))
        for b in range(len(frame)):
            for o in range(len(frame)):
                bird_distances[-1][b, o] = math.sqrt(
                    (frame["x"].iloc[b] - frame["x"].iloc[o]) ** 2 + (frame["y"].iloc[b] - frame["y"].iloc[o]) ** 2
                )

    bird_distances = [pd.DataFrame(bd) for bd in bird_distances]

    return bird_distances


def distances_and_peaks(distances):
    distancesDF = pd.concat(distances)
    distances_list = distancesDF.values.ravel().tolist()
    distances_list = list(filter(lambda x: x != 0, distances_list))
    hist, bins, _ = plt.hist(distances_list, bins=40, color="blue", edgecolor="black")
    peaks, _ = find_peaks(hist)

    plt.hist(distances_list, bins=40, color="blue", edgecolor="black")
    plt.scatter(bins[peaks], hist[peaks], c="red", marker="o", s=50, label="Peaks")

    peak_values = hist[peaks]
    peak_positions = np.round(bins[peaks], 2)

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


# TODO most likely outdated
# remove when tested the new version
# on the locust data
# def birds_to_bird_distances(birds):
#     bird_distances = []

#     for b in range(len(birds)):
#         frames = len(birds[0])
#         bird_distances.append(np.zeros((frames, len(birds))))

#         for o in range(len(birds)):
#             for t in range(frames):
#                 bird_distances[b][t, o] = math.sqrt(
#                     (birds[b]["x"].iloc[t] - birds[o]["x"].iloc[t]) ** 2
#                     + (birds[b]["y"].iloc[t] - birds[o]["y"].iloc[t]) ** 2
#                 )

#         bird_distances[b] = pd.DataFrame(bird_distances[b])
#         bird_distances[b].columns = range(1, len(birds) + 1)

#     return bird_distances


def generate_proximity_score(
    birds,
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
        end = len(birds[0])

    bird_distances = birds_to_bird_distances(birds)

    proximity = visibility.copy()
    for b in range(len(birds)):
        for t in range(start, end):
            proximity[b][t]["proximity"] = 0

            distbt = bird_distances[b].iloc[t].drop(b + 1)

            visible_birds = distbt[distbt <= visibility_range].index.tolist()

            if len(visible_birds) > 0:
                for vb in range(len(visible_birds)):
                    o_x = birds[visible_birds[vb] - 1]["x"].iloc[t]
                    o_y = birds[visible_birds[vb] - 1]["y"].iloc[t]

                    proximity[b][t]["proximity"] += [
                        proximity_score(s, getting_worse, optimal, proximity_decay)
                        for s in np.sqrt((proximity[b][t]["x"] - o_x) ** 2 + (proximity[b][t]["y"] - o_y) ** 2)
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
                proximity[b][t]["proximity"] - proximity[b][t]["proximity"].mean()
            ) / proximity[b][t]["proximity"].std()

            proximity[b][t]["proximity_standardized"].fillna(0, inplace=True)

            proximity[b][t]["bird"] = b + 1
            proximity[b][t]["time"] = t + 1 + time_shift

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
