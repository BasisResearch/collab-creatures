from typing import List

import numpy as np
import pandas as pd

from collab.foraging.toolkit.trace import rewards_trace
from collab.foraging.toolkit.utils import generate_grid
from collab.foraging.toolkit.visibility import filter_by_visibility


def add_velocities_to_foragers(foragers: List[pd.DataFrame]) -> None:

    for forager in foragers:
        forager["velocity_x"] = forager["x"].diff().fillna(0)
        forager["velocity_y"] = forager["y"].diff().fillna(0)


def add_velocities_to_data_object(data_object) -> None:
    add_velocities_to_foragers(data_object.foragers)
    data_object.foragersDF = pd.concat(data_object.foragers)


def generate_velocity_scores(
    sim,
    velocity_time_decay=3,
    velocity_spatial_decay=0.15,
    time_shift=0,
    grid=None,
    visibility_restriction="visible",
):
    velocity_scores = []

    for subject in range(1, sim.num_foragers + 1):

        callingDF = filter_by_visibility(
            sim,
            subject=subject,
            time_shift=time_shift,
            visibility_restriction=visibility_restriction,
            info_time_decay=velocity_time_decay,
            finders_tolerance=2,  # not used in velocity calculations
            filter_by_on_reward=False,  # not used in velocity calculations
        )

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

            subject_at_t = sim.foragersDF[
                (sim.foragersDF["forager"] == subject) & (sim.foragersDF["time"] == t)
            ]
            subject_x_at_t = subject_at_t["x"].iloc[0]
            subject_y_at_t = subject_at_t["y"].iloc[0]

            if slice_t.shape[0] > 0:
                shifted_velocity_x = list(subject_x_at_t + slice_t["velocity_x"])
                shifted_velocity_y = list(subject_y_at_t + slice_t["velocity_y"])

                for _step in range(slice_t.shape[0]):
                    velocity_score["velocity_score"] += rewards_trace(
                        np.sqrt(
                            (shifted_velocity_x[_step] - velocity_score["x"]) ** 2
                            + (shifted_velocity_y[_step] - velocity_score["y"]) ** 2
                        ),
                        velocity_spatial_decay,
                    )

            if velocity_score["velocity_score"].sum() != 0:
                velocity_score["velocity_score_standardized"] = (
                    velocity_score["velocity_score"]
                    - velocity_score["velocity_score"].mean()
                ) / velocity_score["velocity_score"].std()
            else:
                velocity_score["velocity_score_standardized"] = 0

            velocity_score["time"] = velocity_score["time"]

            velocity_scores_b.append(velocity_score)

        velocity_scores_b_df = pd.concat(velocity_scores_b)
        velocity_scores.append(velocity_scores_b_df)

    velocity_scoresDF = pd.concat(velocity_scores)

    return {"velocity_scores": velocity_scores, "velocity_scoresDF": velocity_scoresDF}
