import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import foraging_toolkit as ft


def locust_object_from_data(locustDF, rewardsDF, grid_size, frames):
    class EmptyObject:
        pass

    sim = EmptyObject()

    sim.grid_size = 45
    sim.num_frames = frames
    sim.birdsDF = locustDF
    sim.rewardsDF = rewardsDF
    sim.birds = [group for _, group in locustDF.groupby("bird")]
    sim.rewards = [group for _, group in rewardsDF.groupby("time")]
    sim.num_birds = len(sim.birds)

    step_maxes = []

    for b in range(len(sim.birds)):
        step_maxes.append(
            max(
                max(
                    [
                        abs(
                            sim.birds[b]["x"].iloc[t + 1]
                            - sim.birds[b]["x"].iloc[t]
                        )
                        for t in range(sim.num_frames - 1)
                    ]
                ),
                max(
                    [
                        abs(
                            sim.birds[b]["y"].iloc[t + 1]
                            - sim.birds[b]["y"].iloc[t]
                        )
                        for t in range(sim.num_frames - 1)
                    ]
                ),
            )
        )

    sim.step_size_max = max(step_maxes)

    return sim


def load_and_clean_locust(
    path,
    desired_frames=1800,
    grid_size=45,
    rewards_x=[0.68074, -0.69292],
    rewards_y=[-0.03068, -0.03068],
    subset_starts=150,
    subset_ends=200,
):
    # loading and column names
    locust = pd.read_csv(path)
    locust.drop("cnt", axis=1, inplace=True)
    locust.rename(
        columns={"pos_x": "x", "pos_y": "y", "id": "bird", "frame": "time"},
        inplace=True,
    )
    locust = locust[["x", "y", "time", "bird"]]
    encoder = LabelEncoder()
    locust["bird"] = encoder.fit_transform(locust["bird"])
    locust["bird"] = locust["bird"] + 1

    # frame thinning
    print("original_frames:", locust["time"].max())
    print("original_shape:", locust.shape)

    locust["time"] = (
        np.round(locust["time"] / (45000 / desired_frames)).astype(int) + 1
    )
    locust = locust.drop_duplicates(subset=["time", "bird"], keep="first")
    locust = locust[locust["time"] <= desired_frames]
    print("resulting_frames:", locust["time"].max())
    print("resulting_shape:", locust.shape)

    print("min_time", locust["time"].min())
    print("max_time", locust["time"].max())

    # grid thinning half body size (4/2cm) = 2cm
    # the arena is 90cm diameter

    def rescale_to_grid(column, size):
        mapped = (column + 1) / 2
        rescaled = mapped
        rescaled = np.round(mapped * size - 1) + 1
        return rescaled

    locust["x"] = rescale_to_grid(locust["x"], grid_size)
    locust["y"] = rescale_to_grid(locust["y"], grid_size)
    locust["type"] = "locust"

    # adding constant rewards at two locations
    # specified in the metadata

    time = list(range(1, locust["time"].max() + 1))

    data = {
        "x": rewards_x * len(time),
        "y": rewards_y * len(time),
        "time": [t for t in time for _ in range(len(rewards_x))],
    }

    rewardsDF = pd.DataFrame(data)
    rewardsDF["x"] = rescale_to_grid(rewardsDF["x"], grid_size)
    rewardsDF["y"] = rescale_to_grid(rewardsDF["y"], grid_size)

    locust_subset = locust[
        (locust["time"] >= subset_starts) & (locust["time"] <= subset_ends)
    ]
    rewards_subset = rewardsDF[
        (rewardsDF["time"] >= subset_starts)
        & (rewardsDF["time"] <= subset_ends)
    ]

    loc_subset = locust_object_from_data(
        locust_subset,
        rewards_subset,
        grid_size=grid_size,
        frames=subset_ends - subset_starts,
    )

    loc = locust_object_from_data(
        locust, rewardsDF, grid_size=grid_size, frames=desired_frames
    )

    return {"subset": loc_subset, "all_frames": loc}


# most likely not needed after refactoring, remove soon:

# def derive_predictors_locust(
#     sim,
#     rewards_decay=1.4,
#     visibility_range=15,
#     getting_worse=1,
#     optimal=3,
#     proximity_decay=0.5,
# ):
#     sim.visibility_range = visibility_range

#     tr = ft.rewards_to_trace(
#         sim.rewards,
#         sim.grid_size,
#         sim.num_frames,
#         rewards_decay,
#     )

#     sim.traces = tr["traces"]
#     sim.tracesDF = tr["tracesDF"]

#     vis = ft.construct_visibility(
#         sim.birds, sim.grid_size, visibility_range=sim.visibility_range
#     )

#     sim.visibility = vis["visibility"]
#     sim.visibilityDF = vis["visibilityDF"]

#     prox = ft.generate_proximity_score(
#         sim.birds,
#         sim.visibility,
#         visibility_range=sim.visibility_range,
#         getting_worse=getting_worse,
#         optimal=optimal,
#         proximity_decay=proximity_decay,
#     )

#     sim.proximityDF = prox["proximityDF"]

#     ft.add_how_far_squared_scaled(sim)

#     sim.derivedDF = (
#         sim.tracesDF.merge(sim.visibilityDF, how="inner")
#         .merge(sim.proximityDF, how="inner")
#         .merge(sim.how_farDF, how="inner")
#     )

#     return sim
