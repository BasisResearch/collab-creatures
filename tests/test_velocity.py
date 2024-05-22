from collab.foraging.toolkit.velocity import add_velocities_to_foragers
import pandas as pd
from collab.foraging import random_hungry_followers as rhf

def test_add_velocities_to_foragers():

    data1 = {
        'x': [0, 1, 4, 9, 16, 10],
        'y': [0, 1, 4, 9, 16, 10]
    }
    data2 = {
        'x': [0, 2, 6, 12, 20, 10],
        'y': [0, 2, 6, 12, 20, 10]
    }

    random_foragers_sim = rhf.RandomForagers(
    grid_size=40,
    probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
    num_foragers=3,
    num_frames=10,
    num_rewards=15,
    grab_range=3,
    )
    

    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    foragers = [df1, df2]

    add_velocities_to_foragers(foragers)

    expected_velocity_x1 = [0, 1, 3, 5, 7, -6]
    expected_velocity_y1 = [0, 1, 3, 5, 7, -6]
    expected_velocity_x2 = [0, 2, 4, 6, 8, -10]
    expected_velocity_y2 = [0, 2, 4, 6, 8, -10]

    assert df1['velocity_x'].tolist() == expected_velocity_x1
    assert df1['velocity_y'].tolist() == expected_velocity_y1
    assert df2['velocity_x'].tolist() == expected_velocity_x2
    assert df2['velocity_y'].tolist() == expected_velocity_y2

    random_foragers_sim()
    foragers = random_foragers_sim.foragers
    add_velocities_to_foragers(foragers)

    # for simulated data, just check if the columns are added
    assert foragers[0].shape[1] == 7

