import pandas as pd
import numpy as np
from .utils import generate_grid
from .trace import rewards_trace


def generate_communicates(sim, info_time_decay=3, info_spatial_decay=0.15):
    communicates = []

    for b in range(1, sim.num_birds + 1):
        other_birdsDF = sim.birdsDF[sim.birdsDF["bird"] != b]

        myself = sim.birdsDF[sim.birdsDF["bird"] == b]

        out_of_range_birds = []
        for t in range(1, len(sim.birds[0])):
            myself_x = myself["x"][myself["time"] == t]
            myself_y = myself["y"][myself["time"] == t]

            others_now = other_birdsDF[other_birdsDF["time"] == t].copy()

            others_now["distance"] = np.sqrt(
                (others_now["x"] - myself_x) ** 2 + (others_now["y"] - myself_y) ** 2
            )

            others_now["out_of_range"] = others_now["distance"] > sim.visibility_range

            others_now = others_now[others_now["out_of_range"]]

            on_reward = []
            for index, row in others_now.iterrows():
                others_x = row["x"]
                others_y = row["y"]

                on_reward.append(
                    any(
                        (others_x - sim.rewards[t - 1]["x"] == 0)
                        & (others_y - sim.rewards[t - 1]["y"] == 0)
                    )
                )

            others_now["on_reward"] = on_reward
            out_of_range_birds.append(others_now)
        out_of_range_birdsDF = pd.concat(out_of_range_birds)
        out_of_range_birdsDF = out_of_range_birdsDF[
            out_of_range_birdsDF["on_reward"] == True
        ]

        expansion = [
            out_of_range_birdsDF.assign(time=out_of_range_birdsDF["time"] + i)
            for i in range(1, info_time_decay + 1)
        ]

        expansion_df = pd.concat(expansion, ignore_index=True)

        callingDF = pd.concat([out_of_range_birdsDF, expansion_df])

        grid = generate_grid(sim.grid_size)
        communicates_b = []
        for t in range(1, sim.num_frames + 1):
            slice = callingDF[callingDF["time"] == t]
            communicate = grid.copy()
            communicate["bird"] = b
            communicate["time"] = t
            communicate["communicate"] = 0
            communicate["communicate_standardized"] = 0

            if slice.shape[0] > 0:
                for _step in range(slice.shape[0]):
                    communicate["communicate"] += rewards_trace(
                        np.sqrt(
                            (slice["x"].iloc[_step] - communicate["x"]) ** 2
                            + (slice["y"].iloc[_step] - communicate["y"]) ** 2
                        ),
                        info_spatial_decay,
                    )

            communicate["communicate_standardized"] = (
                communicate["communicate"] - communicate["communicate"].mean()
            ) / communicate["communicate"].std()

            communicates_b.append(communicate)

        communicates_b_df = pd.concat(communicates_b)
        communicates.append(communicates_b_df)
    communicatesDF = pd.concat(communicates)

    return {"communicates": communicates, "communicatesDF": communicatesDF}
