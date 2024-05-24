import numpy as np
import pandas as pd

from collab.foraging.toolkit.trace import rewards_trace
from collab.foraging.toolkit.utils import generate_grid
from collab.foraging.toolkit.visibility import filter_by_visibility


def generate_communicates(
    sim,
    info_time_decay=3,
    info_spatial_decay=0.15,
    finders_tolerance=2,
    time_shift=0,
    grid=None,
    visibility_restriction="invisible",
    filter_by_on_reward=True,
):
    communicates = []

    for b in range(1, sim.num_foragers + 1):

        callingDF = filter_by_visibility(
            sim,
            subject=b,
            time_shift=time_shift,
            visibility_restriction=visibility_restriction,
            info_time_decay=info_time_decay,
            finders_tolerance=finders_tolerance,
            filter_by_on_reward=filter_by_on_reward,
        )


        if grid is None:
            grid = generate_grid(sim.grid_size)

        communicates_b = []

        for t in range(time_shift + 1, (time_shift + len(sim.foragers[0]))):
            slice_t = callingDF[callingDF["time"] == t]

            communicate = grid.copy()
            communicate["forager"] = b
            communicate["time"] = t
            communicate["communicate"] = 0
            communicate["communicate_standardized"] = 0

            if slice_t.shape[0] > 0:
                for _step in range(slice_t.shape[0]):
                    communicate["communicate"] += rewards_trace(
                        np.sqrt(
                            (slice_t["x"].iloc[_step] - communicate["x"]) ** 2
                            + (slice_t["y"].iloc[_step] - communicate["y"]) ** 2
                        ),
                        info_spatial_decay,
                    )

            communicate["communicate_standardized"] = (
                communicate["communicate"] - communicate["communicate"].mean()
            ) / communicate["communicate"].std()

            communicate["time"] = communicate["time"]

            communicates_b.append(communicate)

        communicates_b_df = pd.concat(communicates_b)
        communicates.append(communicates_b_df)
    communicates_df = pd.concat(communicates)

    return {"communicates": communicates, "communicatesDF": communicates_df}
