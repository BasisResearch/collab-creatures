import sys
import os
import csv
import pickle
import time
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

# sys.path.insert(0, "..")


import foraging_toolkit as ft


# defs needed


def prep_data_for__communicators_inference(sim_derived):
    print("Initial dataset size:", sim_derived.derivedDF.shape[0])
    df = sim_derived.derivedDF.copy().dropna()
    print("Complete cases:", df.shape[0])
    # large drop expected as we only care about points within foragers' visibility range
    # and many communicates are outside of it

    data = torch.tensor(
        df[
            [
                "trace_standardized",
                "proximity_standardized",
                "visibility",
                "communicate_standardized",
                "how_far_squared_scaled",
            ]
        ].values,
        dtype=torch.float32,
    )

    trace, proximity, visibility, communicate, how_far = (
        data[:, 0],
        data[:, 1],
        data[:, 2],
        data[:, 3],
        data[:, 4],
    )

    print(str(len(proximity)) + " data points prepared for inference.")

    return trace, proximity, visibility, communicate, how_far


csv_file_path = os.path.join(current_dir, "15EQ20191202_tracked.csv")

locust = ft.load_and_clean_locust(
    path=csv_file_path,
    desired_frames=1350,
    grid_size=45,
    rewards_x=[0.68074, -0.69292],
    rewards_y=[-0.03068, -0.03068],
)


loc = locust["all_frames"]

start_time = time.time()
loc = ft.derive_predictors(
    loc,
    rewards_decay=1,
    visibility_range=8,
    getting_worse=0.3,
    optimal=4,
    proximity_decay=0.3,
    generate_communicates=True,
    info_time_decay=8,
    info_spatial_decay=0.1,
    finders_tolerance=2,
    time_shift=0,
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time}")


(
    proximity,
    trace,
    visibility,
    communicate,
    how_far_score,
) = prep_data_for__communicators_inference(loc)


loc_derived = (proximity, trace, visibility, communicate, how_far_score)


with open("loc_derived.pkl", "wb") as file:
    pickle.dump(loc_derived, file)
