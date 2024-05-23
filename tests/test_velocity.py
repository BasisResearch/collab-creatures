import os

import dill
import pandas as pd

from collab.foraging import random_hungry_followers as rhf
from collab.foraging.toolkit.velocity import add_velocities_to_foragers, add_velocities_to_data_object
from collab.utils import find_repo_root

root = find_repo_root()


def test_add_velocities_to_foragers():

    data1 = {"x": [0, 1, 4, 9, 16, 10], "y": [0, 1, 4, 9, 16, 10]}
    data2 = {"x": [0, 2, 6, 12, 20, 10], "y": [0, 2, 6, 12, 20, 10]}

    random_foragers_sim = rhf.RandomForagers(
        grid_size=40,
        probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
        num_foragers=3,
        num_frames=10,
        num_rewards=15,
        grab_range=3,
    )

    sampling_rate = 0.01

    if "CI" not in os.environ:
        path = os.path.join(
            root,
            f"data/foraging/central_park_birds_cleaned_2022/central_park_objects_sampling_rate_{sampling_rate}.pkl",
        )
        with open(path, "rb") as file:
            central_park_objects = dill.load(file)

        ducks_objects = central_park_objects[0]
        ducks_50 = ducks_objects[50]

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    foragers = [df1, df2]

    add_velocities_to_foragers(foragers)

    expected_velocity_x1 = [0, 1, 3, 5, 7, -6]
    expected_velocity_y1 = [0, 1, 3, 5, 7, -6]
    expected_velocity_x2 = [0, 2, 4, 6, 8, -10]
    expected_velocity_y2 = [0, 2, 4, 6, 8, -10]

    assert df1["velocity_x"].tolist() == expected_velocity_x1
    assert df1["velocity_y"].tolist() == expected_velocity_y1
    assert df2["velocity_x"].tolist() == expected_velocity_x2
    assert df2["velocity_y"].tolist() == expected_velocity_y2

    random_foragers_sim()
    foragers = random_foragers_sim.foragers
    add_velocities_to_foragers(foragers)

    # for simulated data, just check if the columns are added
    assert foragers[0].shape[1] == 7

    if "CI" not in os.environ:
        add_velocities_to_foragers(ducks_50.foragers)
        assert ducks_50.foragers[0].shape[1] == 6


def test_add_velocities_to_data_object():

    random_foragers_sim = rhf.RandomForagers(
        grid_size=40,
        probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
        num_foragers=3,
        num_frames=10,
        num_rewards=15,
        grab_range=3,
    )

    random_foragers_sim()

    add_velocities_to_data_object(random_foragers_sim)

    assert random_foragers_sim.foragers[0].shape[1] == 7
    assert random_foragers_sim.foragersDF.shape[1] == 7


def test_filter_by_visibility():
    sim = MockSim(num_foragers=3, visibility_range=5)

    result = filter_by_visibility(
        sim, 
        subject=1, 
        time_shift=0, 
        visibility_restriction="visible", 
        info_time_decay=1, 
        finders_tolerance=2.0, 
        filter_by_on_reward=False
    )

    # Assertions to validate the correctness of the result
    assert not result.empty
    assert all(result["time"] > 0)
    assert "distance" in result.columns
    assert "out_of_range" in result.columns