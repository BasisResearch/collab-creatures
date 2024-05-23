# %%NBQA-CELL-SEPf56012
import random

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from collab.foraging import random_hungry_followers as rhf
from collab.foraging import toolkit as ft


# %%NBQA-CELL-SEPf56012
random.seed(22)
np.random.seed(22)

follower_sim = rhf.Foragers(
    grid_size=60, num_foragers=3, num_frames=10, num_rewards=10, grab_range=3
)
follower_sim()

follower_sim = rhf.add_follower_foragers(
    follower_sim, num_follower_foragers=3, proximity_decay=0.3, visibility_range=6
)

follower_sim_derived = ft.derive_predictors(
    follower_sim, getting_worse=0.5, optimal=3, visibility_range=6, dropna=False
)

display(follower_sim_derived.derivedDF)

follower_sim_derived.derivedDF.to_csv("followers_test_data.csv", index=False)


# %%NBQA-CELL-SEPf56012
random.seed(22)
np.random.seed(22)

hungry_sim = rhf.Foragers(
    grid_size=40, num_foragers=3, num_frames=10, num_rewards=10, grab_range=3
)

hungry_sim()

hungry_sim = rhf.add_hungry_foragers(
    hungry_sim, num_hungry_foragers=3, rewards_decay=0.3, visibility_range=6
)

hungry_sim_derived = ft.derive_predictors(hungry_sim, dropna=False)

hungry_sim_derived.derivedDF.to_csv("hungry_test_data.csv", index=False)

# random_foragers_derived.derivedDF.to_csv('rhf_test_data.csv', index=False)


# %%NBQA-CELL-SEPf56012
random.seed(22)
np.random.seed(22)

random_foragers_sim = rhf.RandomForagers(
    grid_size=40,
    probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
    num_foragers=3,
    num_frames=5,
    num_rewards=15,
    grab_range=3,
)

# run a particular simulation with these parameters
random_foragers_sim()

# you have created foragers and rewards in a space-time grid
# each row contains the x and y coordinates of a forager at a particular time

random_foragers_sim.foragersDF.head()

random_foragers_derived = ft.derive_predictors(random_foragers_sim, dropna=False)

# random_foragers_derived.derivedDF

random_foragers_derived.derivedDF.to_csv("random_test_data.csv", index=False)
r_test_data = pd.read_csv("random_test_data.csv")


display(random_foragers_derived.derivedDF)
display(r_test_data)

assert_frame_equal(random_foragers_derived.derivedDF, r_test_data)
