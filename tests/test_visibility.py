import warnings

from collab.foraging import random_hungry_followers as rhf
from collab.foraging.toolkit import (
    construct_visibility,
    filter_by_visibility,
    generate_grid,
)
from collab.utils import find_repo_root

root = find_repo_root()
warnings.simplefilter(action="ignore", category=FutureWarning)


def test_filter_by_visibility():
    random_foragers_sim = rhf.RandomForagers(
        grid_size=40,
        probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
        num_foragers=3,
        num_frames=10,
        num_rewards=55,
        grab_range=3,
    )
    random_foragers_sim()
    sim = random_foragers_sim
    grid = generate_grid(sim.grid_size)
    grid = grid.sample(frac=1, random_state=42)
    sim.grid = grid
    vis = construct_visibility(
        sim.foragers,
        sim.grid_size,
        visibility_range=10,
        time_shift=0,
        grid=grid,
    )
    sim.visibility_range = 10
    sim.visibility = vis["visibility"]
    sim.visibilityDF = vis["visibilityDF"]

    df1 = filter_by_visibility(
        sim,
        subject=1,
        time_shift=0,
        visibility_restriction="visible",
        info_time_decay=1,
        finders_tolerance=1,
        filter_by_on_reward=False,
    )

    assert (df1["distance"] <= 10).all()

    df2 = filter_by_visibility(
        sim,
        subject=1,
        time_shift=0,
        visibility_restriction="invisible",
        info_time_decay=1,
        finders_tolerance=1,
        filter_by_on_reward=False,
    )

    assert (df2["distance"] > 10).all()
