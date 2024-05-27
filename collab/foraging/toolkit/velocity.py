from typing import List

import pandas as pd
from collab.foraging.toolkit.visibility import filter_by_visibility
from collab.foraging.toolkit.utils import generate_grid


def add_velocities_to_foragers(foragers: List[pd.DataFrame]) -> None:

    for forager in foragers:
        forager["velocity_x"] = forager["x"].diff().fillna(0)
        forager["velocity_y"] = forager["y"].diff().fillna(0)


def add_velocities_to_data_object(data_object) -> None:
    add_velocities_to_foragers(data_object.foragers)
    data_object.foragersDF = pd.concat(data_object.foragers)


def generate_velocity_scores(
    sim,
    info_time_decay=3,
    info_spatial_decay=0.15,
    finders_tolerance=2,
    time_shift=0,
    grid=None,
    visibility_restriction="visible",
    filter_by_on_reward=False,
):
    velocity_scores = []

    for subject in range(1, sim.num_foragers + 1):

        print("forager",   subject)
        callingDF = filter_by_visibility(
            sim,
            subject=subject,
            time_shift=time_shift,
            visibility_restriction=visibility_restriction,
            info_time_decay=info_time_decay,
            finders_tolerance=finders_tolerance,
            filter_by_on_reward=filter_by_on_reward,
        )

#        print("callingDF", callingDF)

        if grid is None:
            grid = generate_grid(sim.grid_size)

        velocity_scores_b = []

        for t in range(time_shift + 1, (time_shift + len(sim.foragers[0]))):
            slice_t = callingDF[callingDF["time"] == t]

            velocity_score = grid.copy()
            velocity_score["forager"] = subject
            velocity_score["time"] = t
            velocity_score["velocity_score"] = 0
            velocity_score["velocity_score_standardized"] = 0
           

            subject_at_t = sim.foragersDF[(sim.foragersDF["forager"] == subject) & (sim.foragersDF["time"] == t)]
            subject_x_at_t = subject_at_t["x"].iloc[0]
            subject_y_at_t = subject_at_t["y"].iloc[0]

            print("subject_at_t", subject_at_t)
            print("subject_x_at_t", subject_x_at_t)
            print("subject_y_at_t", subject_y_at_t)
