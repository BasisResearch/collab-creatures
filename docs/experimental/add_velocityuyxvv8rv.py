# %%NBQA-CELL-SEPf56012
# importing packages. See https://github.com/BasisResearch/collab-creatures for repo setup
import logging
import os
import random
import time
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np

from collab.utils import find_repo_root

root = find_repo_root()
from collab.foraging import random_hungry_followers as rhf
from collab.foraging import toolkit as ft
from collab.foraging.toolkit.velocity import add_velocities_to_foragers

logging.basicConfig(format="%(message)s", level=logging.INFO)

# users can ignore smoke_test -- it's for automatic testing on GitHub, to make sure the notebook runs on future updates to the repository
smoke_test = "CI" in os.environ
num_frames = 5 if smoke_test else 50
num_svi_iters = 10 if smoke_test else 1000
num_samples = 10 if smoke_test else 1000


notebook_starts = time.time()


# %%NBQA-CELL-SEPf56012
random.seed(23)
np.random.seed(23)

random_foragers_sim = rhf.RandomForagers(
    grid_size=40,
    probabilities=[1, 2, 3, 2, 1, 2, 3, 2, 1],
    num_foragers=3,
    num_frames=num_frames,
    num_rewards=15,
    grab_range=3,
)

# run a particular simulation with these parameters
random_foragers_sim()

# the results of the simulation are stored in `random_foragers_sim.foragersDF`.
# each row contains the x and y coordinates of a forager at a particular time

random_foragers_sim.foragersDF.head()

preferred_proximity = 4  # the distance at which foragers prefer to be from each other
random_foragers_derived = ft.derive_predictors(
    random_foragers_sim, optimal=preferred_proximity, dropna=False
)


# derived = random_foragers_derived.derivedDF

# display(data.head())


# %%NBQA-CELL-SEPf56012
foragers = random_foragers_sim.foragers
len(foragers)

add_velocities_to_foragers(foragers)

display(foragers[0].shape[1] == 7)

# TODO loop later
# forager = foragers[0]

# display(forager.head())

# forager['velocity_x'] = forager['x'].diff().fillna(0)
# forager['velocity_y'] = forager['y'].diff().fillna(0)


# %%NBQA-CELL-SEPf56012
# this file is generated using `central_park_birds_predictors.ipynb`
path = os.path.join(
    root,
    f"data/foraging/central_park_birds_cleaned_2022/central_park_objects_sampling_rate_{sampling_rate}.pkl",
)

with open(path, "rb") as file:
    central_park_objects = dill.load(file)
