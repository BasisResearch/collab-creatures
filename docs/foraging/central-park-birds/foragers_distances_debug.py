import copy
import os

import dill
import matplotlib.pyplot as plt
import pandas as pd
from plotly import express as px

from collab.foraging import toolkit as ft
from collab.utils import find_repo_root

root = find_repo_root()




def object_from_data(
    foragersDF,
    grid_size=None,
    rewardsDF=None,
    frames=None,
    calculate_step_size_max=False,
):
    print('you are in the new object_from_data')
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

    print(sim.foragersDF["forager"].unique())
    sim.foragers = [group for _, group in foragersDF.groupby("forager")]
    print ("sim.foragers", [sim.foragers[k]['forager'].unique().item() for k in range(len(sim.foragers))])

    if rewardsDF is not None:
        sim.rewardsDF = rewardsDF
        sim.rewards = [group for _, group in rewardsDF.groupby("time")]

    
    sim.num_foragers = len(sim.foragers)

    # TODO: this doesn't work for foragers that run away
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
    print("forager map", forager_map)
    for forager in range(len(foragers)):
        forager_distances = []

        times_forager_present = foragers[forager]["time"].unique()

        forager_name = foragers[forager]["forager"].unique().item()

        for frame in times_forager_present:
            foragers_at_frameDF = foragersDF[foragersDF["time"] == frame].copy()
            foragers_at_frameDF.sort_values(by="forager", inplace=True)

            foragers_at_frame = foragers_at_frameDF["forager"].unique()
            foragers_at_frame.sort()
            #print(f"foragers at {frame}", foragers_at_frame)
            #if 29 in foragers_at_frame:
            #    print("Fuck!")

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
               # print(f"{other} location", other_location)
    #             df = foragers[other_location].copy()
    #             print("frame", frame)
    #             print("xs to pick", df[df["time"] == frame]["x"])
    #             other_x = df[df["time"] == frame]["x"].item()
    #             other_y = df[df["time"] == frame]["y"].item()

    #             print("other x y", other_x, other_y)
    #             assert isinstance(other_x, float) and isinstance(other_y, float)

    #             distances_now.append(
    #                 math.sqrt((forager_x - other_x) ** 2 + (forager_y - other_y) ** 2)
    #             )

    #         distances_now_df = pd.DataFrame(
    #             {"distance": distances_now, "foragers_at_frame": foragers_at_frame}
    #         )

    #         forager_distances.append(distances_now_df)

    #     distances.append(forager_distances)

    # return distances



ducks_raw = pd.read_csv(
    os.path.join(
        root,
        "data/foraging/central_park_birds_cleaned_2022/20221215122046189_-5_25_bone.avi.hand_checked_cleaned_df.csv",
    )
)
sparrows_raw = pd.read_csv(
    os.path.join(
        root,
        "data/foraging/central_park_birds_cleaned_2022/20221229124843603_n5_25_bone.avi.hand_checked_cleaned_df.csv",
    )
)

ducks_raw = ducks_raw.rename(columns={"bird": "forager"})
sparrows_raw = sparrows_raw.rename(columns={"bird": "forager"})


ducks_sub = ft.subset_frames_evenly_spaced(ducks_raw, 300)
ducks_sub = ft.rescale_to_grid(ducks_sub, 90)
ducks_obj = ft.object_from_data(
    ducks_sub, grid_size=90, frames=300, calculate_step_size_max=True
)

sps_sub = ft.subset_frames_evenly_spaced(sparrows_raw, 2300)
sps_sub = sps_sub[sps_sub["time"] <= 800]
sps_sub = ft.rescale_to_grid(sps_sub, 90)
sps_obj = ft.object_from_data(
    sps_sub, grid_size=90, frames=300, calculate_step_size_max=True
)

print("uniques in df", ducks_obj.foragersDF['forager'].unique())
duck_distances = foragers_to_forager_distances(ducks_obj)
sps_distances = foragers_to_forager_distances(sps_obj)

# foragers are not in order in sps

